# llama.cpp-strix-halo-RCCL-RDMA

**Cross-node tensor parallelism for llama.cpp via an in-graph RCCL/NCCL all-reduce.**

`-sm tensor` normally does its per-layer all-reduce as a host-side *butterfly* of GET/SET_TENSOR
RPC round-trips. This adds a `GGML_OP_ALLREDUCE` graph op backed by a cross-process **`ncclCommInitRank`
world communicator**, so the collective rendezvouses *inside* RCCL — one `GRAPH_COMPUTE`/token instead
of ~120 per-op round-trips. Built and tested on 2× AMD Strix Halo (gfx1151) over a Thunderbolt/USB4 bond.

## Why this exists (the differentiator)

Mainline llama.cpp shipped NCCL/RCCL tensor parallelism in Apr 2026 (build b8738) — but it's **local
multi-GPU only** (NVLink/PCIe in one box). This is the **cross-node** case: a world communicator spanning
*separate machines*, bootstrapped over TCP, RDMA-ready. Plus **two bug fixes that make `-sm tensor` over
RPC work at all** (see below).

## Status

| | |
|---|---|
| ✅ **Tested** | 2× Strix Halo gfx1151, Thunderbolt/USB4 bond, **TCP**. Correct (`reduced to 1.50`), +18% vs butterfly on 27B. |
| ⏳ **Pending hardware** | ConnectX RDMA NIC. The port is transport-agnostic — RCCL auto-uses IB verbs when present (`NCCL_IB_DISABLE=0`). RDMA (~5 µs/op vs TCP's ~286 µs) is what should tip TP past pipeline. |

## Measured (2 nodes, bond, TCP, in-container ROCm 7.14)

| model | `-sm layer` (pipeline) | `-sm tensor` butterfly | `-sm tensor` **RCCL (this)** |
|---|---|---|---|
| Qwen3.5-2B Q8_0 | 70.98 | 19.35 | 19.28 *(break-even — 2B too small, per-sync overhead dominates)* |
| Huihui-Qwen3.6-27B Q6_K | **8.87** | 3.10 | **3.65 (+18%)** |

Primitive: **286 µs/all-reduce** over the bond (== torch.distributed, i.e. native RCCL-over-TCP floor).

### Read it honestly
- The port **works end-to-end** and **beats the butterfly by +18% on the 27B** (bigger tensors let the
  collective's efficiency show).
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
