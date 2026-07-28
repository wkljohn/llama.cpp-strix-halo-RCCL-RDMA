# llama.cpp-strix-halo-RCCL-RDMA

**Cross-node inference on AMD Strix Halo — over RDMA on the Thunderbolt cable you
already have, plus an in-graph RCCL all-reduce for tensor parallelism.**

> **Engineering by [paicon](https://paix-navigator.paicon.com)**
> This work is part of the [paix-navigator.paicon.com](https://paix-navigator.paicon.com) effort.

The world communicator is **N-rank**, not two: `GGML_NCCL_WORLD` sets the size, rank 0
serves the `ncclUniqueId` to every peer, and the only constraint the code enforces is
`world >= 2`. Everything measured here is 2 nodes, because that is the hardware on hand
— **3+ is supported by construction, untested in practice.**

Two parts, usable independently:

1. **[odinlink/](odinlink/) — RDMA over Thunderbolt/USB4.** Working today, no NIC.
   **22.0 µs** median round trip against TCP's **286 µs** on the same cable.
2. **The RCCL tensor-parallel port** — a `GGML_OP_ALLREDUCE` graph op backed by a
   cross-process `ncclCommInitRank` world communicator, so the collective rendezvouses
   *inside* RCCL: one `GRAPH_COMPUTE` per token instead of ~120 per-op round-trips.

## Measured — 27B Q6_K, 2 nodes, `-sm layer`

| transport | pp512 (t/s) | tg128 (t/s) | vs TCP |
|---|---|---|---|
| **RDMA — [OdinLink](https://github.com/wkljohn/OdinLink-Five/tree/strix-halo-verbs-fixes) + the fixes in [odinlink/](odinlink/)** | 290.68 ± 61.73 | **9.16 ± 0.02** | **+3.7 %** |
| RDMA — `thunderbolt_ibverbs` 0.3.4 | 290.23 ± 61.80 | 9.07 ± 0.01 | +2.7 % |
| TCP over bond0 | 292.11 ± 62.37 | 8.83 ± 0.03 | — |
| *single node (reference)* | *224.68* | *9.50 ± 0.05* | — |

Byte-verified, not inferred from "the benchmark finished": 21 GiB unidirectional,
86016/86016 messages. `llama-bench` measures speed, not correctness — a corrupting
transport still prints a plausible t/s, and one did.

*(An earlier "2 GiB each way bidirectional, 9.84 Gb/s" figure is retracted: it was a single
lucky run. Duplex reliability turned out to depend on cable quality — see
[odinlink/RESULTS.md](odinlink/RESULTS.md).)*

**Single node is still fastest for a model that fits.** Two nodes are for *capacity*.
RDMA recovers about half the cross-node penalty (8.83 → 9.16 against a 9.50 ceiling).

→ **[odinlink/REPRODUCE-RPC.md](odinlink/REPRODUCE-RPC.md)** to run this.
→ **[odinlink/RESULTS.md](odinlink/RESULTS.md)** for every number and its caveats.

## Status

| | |
|---|---|
| ✅ **RDMA inference** | Working end-to-end and byte-verified. Transport is [wkljohn/OdinLink-Five @ `strix-halo-verbs-fixes`](https://github.com/wkljohn/OdinLink-Five/tree/strix-halo-verbs-fixes) — clone that branch and you have all 23 fixes |
| ✅ **RCCL all-reduce, correct** | 2-rank proof `reduced to 1.50`; 286 µs/op over TCP, matching `torch.distributed` — i.e. the native RCCL-over-TCP floor, no overhead added |
| ✅ **Beats the host butterfly** | +13–18 % on the 27B across two independent sessions. The collective is genuinely faster than GET/SET round-trips |
| ✅ **Runs models too big for one node** | 298B hy_v3 MoE (95 GiB) tensor-parallels across both nodes at 1.78 t/s |
| ⚠️ **RDMA discovery is an LD_PRELOAD shim** | OdinLink registers no kernel `ib_device`; anything not inheriting the preload cannot see it |
| ❌ **RDMA does NOT help tensor parallel** | Measured: TCP **3.50 ± 0.01** beats RDMA **3.30 ± 0.01** t/s. The collective is 2.9× faster over RDMA, but TP is bound by host dispatch (~4.13 ms/sync vs a 100 µs collective), so a transport that costs CPU loses. See [odinlink/RESULTS.md](odinlink/RESULTS.md) |

## Why this exists

Mainline llama.cpp shipped NCCL/RCCL tensor parallelism in Apr 2026 (b8738) — but
**local multi-GPU only** (NVLink/PCIe in one box). This is the **cross-node** case: a
world communicator spanning separate machines. Plus **two bug fixes that make
`-sm tensor` over RPC work at all**, and an RDMA transport that needs no new hardware.

Cross-node TP was assumed to be latency-bound: at 286 µs/op the per-layer all-reduce
dominates, so a ~13× lower comm term should let TP overtake pipeline. **That measurement
has now been made, and the assumption was wrong.** TP is bound by *host dispatch*
(~4.13 ms per sync point against a 100 µs collective), so RDMA can address only ~3.7 % of
the token budget — and because the transport busy-spins, it gives back more than it gains:
TCP 3.50 ± 0.01 vs RDMA 3.30 ± 0.01 t/s, both far behind pipeline's 9.20.

The useful conclusion: **a faster interconnect does not rescue cross-node tensor
parallelism here.** Full arithmetic in [odinlink/RESULTS.md](odinlink/RESULTS.md).

## Quick start — RCCL tensor parallel

Full recipe in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**. The gist (both ranks in the
`kyuz0/vllm-therock-gfx1151` container — the ROCm 7.14 `_rocm_sdk` stack is the only
runtime that works here):

```bash
# apply to a 635cdd5 checkout (or rebase — see Roadmap)
git apply rccl-tp-port.patch
cmake -S . -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=ON -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench ggml-rpc-server -j$(nproc)

# env (every node): NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0
#   GGML_NCCL_WORLD=<N> GGML_NCCL_MASTER=<head-ip>:29500
# peers: GGML_NCCL_RANK=<1..N-1> ggml-rpc-server -H 0.0.0.0 -p 50070
# head:  GGML_NCCL_RANK=0 GGML_META_WORLD_ALLREDUCE=1 llama-bench -m <model> \
#          --rpc <peer1>:50070,<peer2>:50070 -sm tensor -ts 34/33/33 -ngl 99
```

**Scaling past two nodes.** Set `GGML_NCCL_WORLD` to the node count, give each peer a
distinct `GGML_NCCL_RANK`, pass every peer to `--rpc`, and size `-ts` to match. Rank 0
blocks until all `N-1` peers have collected the unique ID. Nothing in the port assumes
two — but nothing beyond two has been run here, so treat 3+ as untested.

The **RDMA transport does not follow automatically**: a Thunderbolt cable is
point-to-point, so 3+ nodes need a topology (daisy chain, or more ports per node) that
was never built or measured. Over TCP/IP the node count is unconstrained.

To point RCCL at the RDMA link instead of TCP, see
**[odinlink/REPRODUCE-RCCL.md](odinlink/REPRODUCE-RCCL.md)** — three separate traps there
end in a *silent* fallback to sockets.

## What's inside

```
odinlink/                 # RDMA over Thunderbolt: recipes, results, 23 defects, patch
rccl-tp-port.patch        # the full code diff vs ggml-org/llama.cpp @ 635cdd5
docs/REPRODUCE.md         # authoritative: the 7 changes (by anchor, not line#), build, run
docs/DESIGN.md            # design history / rationale
tests/test-world-allreduce.cpp   # 2-rank proof harness (fill rank r with r+1, expect 1.5)
scripts/                  # gfx1151 RCCL build, peer runtime install, rank launcher
```

## The port, in one screen (details in docs/REPRODUCE.md)

1. `GGML_OP_ALLREDUCE` op — `ggml.h` / `ggml.c`.
2. World communicator — `ggml-cuda.cu` (~180 lines): `ncclCommInitRank` (**not**
   `ncclCommInitAll`), TCP `ncclUniqueId` handshake via `GGML_NCCL_{RANK,WORLD,MASTER}`;
   auto-disables CUDA graphs when present.
3. Export `ggml_backend_cuda_world_init()` — `ggml-cuda.h`.
4. RPC proto guard bump — `ggml-rpc.h`.
5. **BUGFIX** empty-slice bounds — `ggml-rpc.cpp` `deserialize_tensor`: skip the
   buffer-bounds assert for `ggml_is_empty()` tensors. `-sm tensor` hands a rank a
   zero-sized slice whose ptr lands past the buffer; the old assert aborted the
   rpc-server mid-graph. **This is what unbreaks `-sm tensor` over RPC.**
6. Route the TP sync through the collective — `ggml-backend-meta.cpp` (~48 lines), gated
   on `GGML_META_WORLD_ALLREDUCE`. **Dispatch order high→low (RPC first) or deadlock.**
7. `transport.cpp` — an earlier Thunderbolt-ibverbs RDMA-sender experiment; not part of
   the collective port, kept for RDMA groundwork.

## Gotchas that cost real debugging time

- **Large models (>~100 GiB) over RPC: use `-mmp 0` / `--no-mmap`.** With mmap on,
  `llama-cli`/`llama-server` **hang at 0 % GPU or crash with a HIP/HSA fault** while
  loading — the kernel `mmap`s the model and the HIP runtime's GPU-buffer allocation
  collides on the UMA address space (upstream
  [#19745](https://github.com/ggml-org/llama.cpp/issues/19745)). This is what made the
  298B run possible; the same stall hit *both* `-sm tensor` and `-sm layer`.
- **Bonded Thunderbolt link: set `net.ipv4.tcp_reordering=127`** (+ larger
  `rmem_max`/`wmem_max`). `balance-rr` sends packets round-robin across both links → the
  default `tcp_reordering=3` treats that as loss and throughput collapses to one link.
  **It resets to 3 on reboot** — persist it in `/etc/sysctl.d/`.
- **Runtime:** the container's `/opt/rocm` librccl **segfaults standalone**; the
  `_rocm_sdk_libraries_gfx1151` one works — put it first on `LD_LIBRARY_PATH`.
  `NCCL_CUMEM_ENABLE=0` is required.
- **rpc-server** must bind `0.0.0.0` (not the node IP) and launch via `exec` (a
  `sleep infinity` container can orphan a `&`-backgrounded server).
- More, including a corrupt `~/.cache/comgr` that bricks HIP entirely:
  [odinlink/FINDINGS.md](odinlink/FINDINGS.md).

## Historical — TCP-era tensor-parallel A/B

Measured before the RDMA work, two nodes over the bonded IP link, `-sm tensor -ts 50/50
-ngl 99`, two independent sessions. Kept because it is what establishes that the RCCL
collective beats the host butterfly:

| model | `-sm layer` (pipeline) | `-sm tensor` butterfly | `-sm tensor` **RCCL (this)** |
|---|---|---|---|
| Qwen3.5-2B Q8_0 | 70.98 | 19.35 | 19.28 *(break-even — 2B too small)* |
| Qwen3.6-27B Q6_K | **8.87** | 3.10 / 3.56 | **3.65 / 4.02 (+13–18 %)** |

Primitive: **286 µs/all-reduce** over the bond (== `torch.distributed`). Transport
confirmed via `NCCL_DEBUG=INFO`: `NET/Socket : Using [0]bond0:10.4.0.1`.

Read honestly: the RCCL win is *within* the TP path, not against pipeline. Pipeline
remains the fastest cross-node path over TCP because TP all-reduces **every** layer, and
that per-sync cost dominates regardless of butterfly-vs-RCCL. Overtaking pipeline needs
the RDMA floor — which is exactly why [odinlink/](odinlink/) exists, and why the
TP-over-RDMA run is the measurement still worth making.

## Roadmap

- [ ] **Run `-sm tensor` over the RDMA plugin** — the one measurement that would settle
      whether TP beats pipeline here. Nothing blocks it; it just has not been done.
- [ ] Fix teardown hang (BUG 24) and the `connect`/`accept` retry contract (BUG 25),
      both of which collectives hit far harder than pipeline does.
- [ ] Attribute the rare fragment drop now that sequencing detects it
      ([odinlink/FINDINGS.md](odinlink/FINDINGS.md) BUG 22), then add NAK/retry.
- [ ] A kernel `ib_device` for OdinLink, retiring the `LD_PRELOAD` shim.
- [ ] Rebase onto current mainline (gets `hy_v3` and every new arch).

## Provenance / license

Fork of **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** @ `635cdd5`,
**MIT** (see `LICENSE`). RDMA transport is
[wkljohn/OdinLink-Five](https://github.com/wkljohn/OdinLink-Five) (driver GPL-2.0, verbs
provider MIT). Developed and measured on Strix Halo hardware; AI-assisted. **Not**
submitted upstream to llama.cpp — its `AGENTS.md` disallows AI-generated PRs, so this
stays a personal fork.
