# llama.cpp-strix-halo-RCCL-RDMA

**Cross-node tensor parallelism for llama.cpp via an in-graph RCCL/NCCL all-reduce.**

`-sm tensor` normally does its per-layer all-reduce as a host-side *butterfly* of GET/SET_TENSOR
RPC round-trips. This adds a `GGML_OP_ALLREDUCE` graph op backed by a cross-process **`ncclCommInitRank`
world communicator**, so the collective rendezvouses *inside* RCCL — one `GRAPH_COMPUTE`/token instead
of ~120 per-op round-trips. Built and tested on 2× AMD Strix Halo (gfx1151) over a Thunderbolt/USB4 bond.

## What works today ✅

- **Correct cross-node RCCL all-reduce** — 2-rank proof `reduced to 1.50`; 286 µs/op, matching
  `torch.distributed` (i.e. it hits the native RCCL-over-TCP floor, no overhead added).
- **Beats the host butterfly** on the 27B: **+13–18%** across two independent sessions (RCCL 4.02 vs
  butterfly 3.56 clean). The collective is genuinely faster than GET/SET round-trips.
- **Runs models too big for one node** — a **298B hy_v3 MoE (95 GiB)** tensor-parallels across the two
  nodes at **1.78 t/s**. That's the point of cross-node TP: capacity a single 96 GB carve can't hold.
- **Two upstream bug fixes** that make `-sm tensor` over RPC work *at all* (zero-sized-slice crash;
  dispatch-order deadlock) — useful independent of the collective.
- **RDMA-ready, no code change** — the collective is transport-agnostic; RCCL auto-selects IB verbs when a
  NIC is present.

**⏳ The RDMA test is pending hardware — and it's the headline number, not a footnote.** Over TCP the
per-layer sync cost (286 µs/op) is what keeps TP from beating pipeline; RDMA drops that to **~5 µs/op**
(~50×), which is what should tip TP-RCCL *past* pipeline. So the TCP results below **prove correctness and
the collective win — they are the pre-RDMA baseline, not the ceiling.** The ceiling is a $-few NIC away.

## Why this exists (the differentiator)

Mainline llama.cpp shipped NCCL/RCCL tensor parallelism in Apr 2026 (build b8738) — but it's **local
multi-GPU only** (NVLink/PCIe in one box). This is the **cross-node** case: a world communicator spanning
*separate machines*, bootstrapped over TCP, RDMA-ready. Plus **two bug fixes that make `-sm tensor` over
RPC work at all** (see below).

## Status

| | |
|---|---|
| ✅ **Tested** | 2× Strix Halo gfx1151, Thunderbolt/USB4 bond, **TCP**. Correct (`reduced to 1.50`); RCCL beats butterfly on 27B (**+7–18%**, re-verified **+13% clean**). RCCL confirmed binding the bond — `NCCL_DEBUG=INFO`: `NET/Socket : Using [0]bond0:10.4.0.1`. |
| ⏳ **Pending hardware** | ConnectX RDMA NIC. The port is transport-agnostic — RCCL auto-uses IB verbs when present (`NCCL_IB_DISABLE=0`). RDMA (~5 µs/op vs TCP's ~286 µs) is what should tip TP past pipeline. |

## Measured (2 nodes, bond, TCP, in-container ROCm 7.14)

`-n` tokens/s (`tg`), `-sm tensor -ts 50/50 -ngl 99`, two independent A/B sessions:

| model | `-sm layer` (pipeline) | `-sm tensor` butterfly | `-sm tensor` **RCCL (this)** |
|---|---|---|---|
| Qwen3.5-2B Q8_0 | 70.98 | 19.35 | 19.28 *(break-even — 2B too small, per-sync overhead dominates)* |
| Qwen3.6-27B Q6_K | **8.87** | 3.10 / 3.56 | **3.65 / 4.02 (+13–18%)** |

*(27B shows both sessions: original / fresh post-bond-repair. Numbers move with bond contention;
the RCCL-beats-butterfly margin held both times.)*

Primitive: **286 µs/all-reduce** over the bond (== torch.distributed, i.e. native RCCL-over-TCP floor).
RCCL's transport confirmed on the bond via `NCCL_DEBUG=INFO`: `NET/Socket : Using [0]bond0:10.4.0.1`.

**Capacity proof (mainline, not this port):** a **298B hy_v3 MoE (IQ2_M, 95 GiB)** tensor-parallels across
the two nodes at **1.78 t/s** (`-sm tensor -ts 50/50 -mmp 0`) — i.e. the 2-node setup runs models far too
big for one 96 GB carve. Requires the `-mmp 0` fix below.

### Read it honestly
- The port **works end-to-end** and **beats the butterfly on the 27B** (+13–18% across sessions; bigger
  tensors let the collective's efficiency show).
- **Pipeline (`-sm layer`) is still the fastest cross-node path over TCP** — TP all-reduces *every* layer,
  and that per-sync cost dominates regardless of butterfly-vs-RCCL. The RCCL win is *within* the TP path,
  not vs pipeline. Overtaking pipeline needs the ~5 µs RDMA floor, not the ~286 µs TCP one.
- For a model that **fits on one node**, single-node still wins. TP's value is **capacity** (models too
  big for one 96 GB carve) — where the +18% is the payoff.
- Known issue: the 27B world path **intermittently hangs at startup** (rendezvous race) — re-run succeeds.

## What's inside

```
rccl-tp-port.patch        # the full code diff vs ggml-org/llama.cpp @ 635cdd5
docs/REPRODUCE.md         # authoritative: the 7 changes (by anchor, not line#), build, run, gotchas
docs/DESIGN.md            # design history / rationale
tests/test-world-allreduce.cpp   # 2-rank proof harness (fill rank r with r+1, expect 1.5)
scripts/                  # gfx1151 RCCL build, peer runtime install, rank launcher, nccl smoke test
```

## Quick start

Full recipe in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**. The gist (both ranks in the
`kyuz0/vllm-therock-gfx1151` container — the ROCm 7.14 `_rocm_sdk` stack is the only runtime that works here):

```bash
# apply to a 635cdd5 checkout (or rebase — see Roadmap)
git apply rccl-tp-port.patch
cmake -S . -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=ON -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench ggml-rpc-server -j$(nproc)

# env (both nodes): NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0
#   GGML_NCCL_WORLD=2 GGML_NCCL_MASTER=<head-ip>:29500
# peer: GGML_NCCL_RANK=1 ggml-rpc-server -H 0.0.0.0 -p 50070
# head: GGML_NCCL_RANK=0 GGML_META_WORLD_ALLREDUCE=1 llama-bench -m <model> \
#         --rpc <peer-ip>:50070 -sm tensor -ts 50/50 -ngl 99
```

For **RDMA** (when the NIC lands): add `libibverbs`/`libmlx5` (rdma-core) to the container, set
`NCCL_IB_DISABLE=0 NCCL_IB_HCA=<mlx5 dev>`, and confirm `NCCL_DEBUG=INFO` prints `NET/IB` not `NET/Socket`.
No code change — the collective is transport-agnostic.

## Gotchas that matter on Strix Halo (cost real debugging time)

- **Large models (>~100 GiB) over RPC: use `-mmp 0` / `--no-mmap`.** With mmap on, `llama-cli`/`llama-server`
  **hang at 0% GPU or crash with a HIP/HSA fault** while loading — the kernel `mmap`s the model and the HIP
  runtime's GPU-buffer allocation collides on the UMA address space (upstream [#19745](https://github.com/ggml-org/llama.cpp/issues/19745)).
  `--no-mmap` reads straight to GPU buffers and fixes it. `llama-bench` uses `-mmp 0`. (This is what made the
  298B hy_v3 run above possible — same stall hit *both* `-sm tensor` and `-sm layer` until mmap was off.)
- **Aggregated (bonded) Thunderbolt link: set `net.ipv4.tcp_reordering=127`** (+ larger `rmem_max`/`wmem_max`).
  `balance-rr` sends packets round-robin across both links → out-of-order arrival; the default `tcp_reordering=3`
  treats that as loss and throughput collapses to a single link. **It resets to 3 on reboot** — persist it in
  `/etc/sysctl.d/`. Verify aggregation with `iperf3` (should ~2× a single link) and per-slave TX counters.
- **Runtime:** the container's `/opt/rocm` librccl **segfaults standalone**; the `_rocm_sdk_libraries_gfx1151`
  one works — put it first on `LD_LIBRARY_PATH` (see `docs/REPRODUCE.md`). `NCCL_CUMEM_ENABLE=0` is required.
- **rpc-server** must bind `0.0.0.0` (not the node IP) and launch via `exec` (a `sleep infinity` container can
  orphan a `&`-backgrounded server when `podman exec -d` returns).

## The port, in one screen (details in docs/REPRODUCE.md)

1. `GGML_OP_ALLREDUCE` op — `ggml.h` / `ggml.c` (enum before `GGML_OP_COUNT`, name/symbol arrays, builder).
2. World communicator — `ggml-cuda.cu` (~180 lines): `ncclCommInitRank` (**not** `ncclCommInitAll`),
   TCP `ncclUniqueId` handshake via `GGML_NCCL_{RANK,WORLD,MASTER}`; op runs `ncclAllReduce`; auto-disables
   CUDA graphs when present.
3. Export `ggml_backend_cuda_world_init()` — `ggml-cuda.h`.
4. RPC proto guard bump — `ggml-rpc.h` (op enum grew).
5. **BUGFIX** empty-slice bounds — `ggml-rpc.cpp` `deserialize_tensor`: skip the buffer-bounds assert for
   `ggml_is_empty()` tensors. `-sm tensor` hands a rank a zero-sized slice whose ptr lands past the buffer;
   the old assert aborted the rpc-server mid-graph. **This is what unbreaks `-sm tensor` over RPC.**
6. Route the TP sync through the collective — `ggml-backend-meta.cpp` (~48 lines), gated on
   `GGML_META_WORLD_ALLREDUCE`. **Dispatch order high→low (RPC first) or deadlock.**
7. `transport.cpp` — a separate earlier Thunderbolt-ibverbs RDMA-sender experiment; not part of the
   collective port, kept for RDMA groundwork.

## Roadmap

- [ ] **RDMA A/B** over ConnectX (`ib_write_lat` first; expect ~5 µs → TP-RCCL should pass pipeline's 8.87).
- [ ] Rebase onto current mainline (gets `hy_v3` and every new arch; mainline's own local RCCL TP may overlap).
- [ ] Fix the 27B rendezvous-race startup hang.

## Provenance / license

Fork of **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** @ `635cdd5`, **MIT** (see `LICENSE`).
Modifications developed + measured on Strix Halo hardware; AI-assisted. **Not** submitted upstream
(llama.cpp `AGENTS.md` disallows AI-generated PRs — this is a personal fork). Numbers above are real
measurements over Thunderbolt-TCP; RDMA figures are targets pending hardware.
