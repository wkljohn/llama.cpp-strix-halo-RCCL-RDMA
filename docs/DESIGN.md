# Cross-node tensor parallelism for llama.cpp via NCCL/RCCL (`nccl-tp` branch)

Goal: give llama.cpp what vLLM has — an **overlapped RCCL collective** for the cross-node
all-reduce — while keeping llama.cpp's fast fused kernels. Today llama.cpp's meta-backend
does the cross-node all-reduce as a host-side "butterfly" (a GET/SET_TENSOR RPC round-trip
per sync ≈ 1 ms dispatch × ~120 syncs/token → the measured 2.2 t/s, 7× slower than
layer-split). This branch executes the all-reduce **inside the graph** so the peer ships
one `GRAPH_COMPUTE` per token and the ranks rendezvous inside RCCL — collapsing ~120 RPC
round-trips/token to 1.

## Design (implemented, committed)

- **`GGML_OP_ALLREDUCE`** — new ggml op + `ggml_allreduce()` builder (ggml.h / ggml.c).
- **Cross-process "world" communicator** (ggml-cuda.cu): `ncclCommInitRank` (NOT the
  local-only `ncclCommInitAll` llama.cpp already had), so ranks span processes/nodes.
  `ncclUniqueId` is exchanged over a plain TCP socket. Config via env:
  `GGML_NCCL_RANK` / `GGML_NCCL_WORLD` / `GGML_NCCL_MASTER=<ip:port>`.
- The op executes `ncclAllReduce` on the compute stream. `ggml_backend_cuda_world_init()`
  is exported for deterministic startup; it also inits lazily on first allreduce.
- CUDA/HIP **graphs auto-disabled** when an allreduce node is present (the collective is a
  runtime rendezvous; it can't be graph-captured — same reason vLLM forces `--enforce-eager`).
- RPC proto patch version bumped (the op-enum guard in ggml-rpc.h caught the new op — proof
  the RPC serialization is version-safe; the peer `ggml-rpc-server` ships whole graphs today).

Diff: ~230 lines across ggml.h, ggml.c, ggml-cuda.{h,cu}, ggml-rpc.h + a test.

## What is proven

1. Compiles against gfx1151 RCCL (both host ROCm 7.2 and the container's ROCm 7.14).
2. `GGML_OP_ALLREDUCE` registers; graph builds + allocates on **real gfx1151** (local smoke).
3. **The world communicator forms across BOTH physical nodes** over the Thunderbolt bond —
   `rank 0/2` + `rank 1/2`, TCP `ncclUniqueId` handshake OK.
4. Independent number: cross-node RCCL all-reduce over the bond = **~290 µs** (measured via
   torch.distributed). `GGML_OP_ALLREDUCE` wraps the same `ncclAllReduce`, so that IS its cost.

## The runtime obstacle, and why we build RCCL from source

Both ready-made RCCL libs fail to init **outside a torch-wrapped process**:
- **Container RCCL (TheRock, HIP 7.14):** `rocmwrap.cc:199 Failed to find ROCm runtime
  library` — works under vLLM only because torch pre-initializes ROCm in-process.
- **Extracted TheRock RCCL on host ROCm 7.2:** `rocmwrap.cc:88 cuMem support requires
  HIP_VERSION >= 7.12` — the host's HIP 7.2 is older than the TheRock lib demands.
  (`NCCL_CUMEM_ENABLE=0` gets past this, but it's a band-aid on a version-mismatched lib.)

Fix: **build gfx1151 RCCL from source against the host's own ROCm 7.2** (kyuz0
`rocm-systems@gfx1151-rccl`, `scripts/build_rccl_gfx1151.sh`). Result = a normal RCCL for
the normal host ROCm: right HIP version, no TheRock repackaging quirks, inits in a plain
native process. Build command (running in `/home/wkljohn/Desktop/cc/rocm-systems`):
```
ROCM_PATH=/opt/rocm-7.2.0 hipcc-configured cmake with -DGPU_TARGETS=gfx1151 ; make -j
```

## Runtime env needed (native, both nodes)

```
GGML_NCCL_RANK=<0|1> GGML_NCCL_WORLD=2 GGML_NCCL_MASTER=10.4.0.1:29500
NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1     # TCP over the bond (RDMA later)
LD_LIBRARY_PATH=<source-built-rccl>:/opt/rocm-7.2.0/lib
# with a source-matched RCCL these should be unnecessary, kept as fallback:
# NCCL_CUMEM_ENABLE=0   GGML_CUDA_DISABLE_GRAPHS=1 (now automatic)
```

## Remaining steps to a running 2-node result

1. ⏳ Finish source RCCL build vs ROCm 7.2 (in progress).
2. Relink `build-nccl` against the source RCCL; rerun local smoke, then 2-rank.
3. **Peer needs the ROCm 7.2 runtime** (it is Vulkan-only today — `/opt/rocm` absent but
   kernel `/dev/kfd` present). Either install `rocm` runtime via apt, or ship a minimal
   runtime bundle (libamdhip64, libhsa-runtime64, comgr, + the source RCCL) and run with
   `LD_LIBRARY_PATH`. Both nodes need it (symmetric).
4. Run `test-world-allreduce` rank0 (head) ↔ rank1 (peer) → confirm correctness (→1.5) and
   the in-graph allreduce latency (~290 µs expected).
5. Wire `ggml_allreduce()` into the model graph at the tensor-parallel sync points (the
   meta-backend already knows where they are) behind an env/CLI flag.

## STATUS 2026-07-21: WORKING END-TO-END ACROSS BOTH NODES ✅

**`CORRECTNESS OK: both ranks reduced to 1.50`  /  `286 us/allreduce` (60/graph, over the bond, TCP).**
- The in-graph `GGML_OP_ALLREDUCE` executes `ncclAllReduce`; the cross-process world
  communicator (`ncclCommInitRank` + TCP `ncclUniqueId` handshake) spans rank0=head,
  rank1=peer. Cross-node sum is exact; 286 us matches the independent torch.distributed
  measurement (~290 us) -> the port achieves native RCCL latency.
- **Winning runtime:** built ggml + linked against the container's ROCm 7.14 `_rocm_sdk`
  stack (`_rocm_sdk_core` + `_rocm_sdk_libraries_gfx1151` = the ABI vLLM uses). Run BOTH
  ranks in-container (`--network host`). Env: LD_LIBRARY_PATH=_rocm_sdk_core:_rocm_sdk_gfx1151:build-c,
  NCCL_SOCKET_IFNAME=bond0, NCCL_IB_DISABLE=1, NCCL_CUMEM_ENABLE=0, GGML_CUDA_DISABLE_GRAPHS
  (now automatic). The `/opt/rocm` librccl in the container segfaults; the `_rocm_sdk` one works.
- Native ROCm 7.2 on the peer crashed in hipStreamCreate (peer never ran native ROCm
  compute); the container path sidesteps that entirely.

### superseded notes (native 7.2 attempt):

- **Head rank 0: fully working** — enumerates gfx1151, opens the world-communicator
  listener, waits for the peer over bond0. Source RCCL `INIT_OK`/`ALLREDUCE_OK` (world=1).
- **Peer rank 1: segfaults in `ncclCommInitRank`**. Backtrace (gdb):
  `ncclCommInitRank → commAlloc → ncclCreateSideStream → [crash inside
  /opt/rocm-7.2.0/lib/libamdhip64.so.7 during hipStreamCreate]`.
  Both nodes are identical Ubuntu 26.04 + ROCm 7.2, but the **peer has never run native
  ROCm compute** (only Vulkan + the container's ROCm 7.14). Its native HIP stream/queue
  creation crashes where the container path (ROCm 7.14) worked (vLLM ran there fine).
- The peer's native ROCm 7.2 was hand-installed (repo copied from head; `libtinfo-dev`
  stub for the newer Ubuntu; `--no-install-recommends` subset because `rocm-hip-runtime`
  meta-package deps conflict). The crash is inside libamdhip64 itself, not a missing-lib
  link error, so it is a runtime/queue-init problem, not obviously a missing package.

**Two candidate resolutions (next session):**
1. **Build the whole stack against ROCm 7.14 and run in-container on both nodes** — the
   container's HIP is the *proven-working* peer runtime (vLLM TP=2 ran there). Rebuild
   source RCCL + ggml against 7.14 in-container (source-built RCCL avoids the rocmwrap
   issue the *prebuilt* TheRock lib had — validated on host). Cleanest path.
2. Debug the peer's native ROCm 7.2 hipStreamCreate crash (iommu=pt kernel param?
   complete the package set to exactly match head? queue/SDMA env knobs?).

Recommend #1: it reuses the container that already works and keeps one ABI (7.14) on both.

## Economics (the honest gate)

At ~290 µs/allreduce over TCP: ~120/token ≈ 35 ms/token comm — **break-even** vs the
compute TP saves. The port is correct and will work, but over Thunderbolt-TCP it does not
beat layer-split. It flips to **+50–75%** only with real RDMA (~30–100 µs/op) — i.e. the
ConnectX-4-Lx-over-USB4 NIC. So: **finish the port now (cheap), buy the NIC to make it pay.**

Commits on `nccl-tp`: see `git log`. Related: `VLLM-TP2-STRIX-HALO.md`,
memory `strix-halo-vllm-tensor-parallel-rccl-mod.md`.
