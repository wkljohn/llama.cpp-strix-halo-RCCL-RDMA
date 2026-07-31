# RDMA over Thunderbolt/USB4 on Strix Halo

Two AMD Strix Halo nodes running llama.cpp cross-node inference with tensor traffic
over **RDMA on the Thunderbolt/USB4 cable you already have** — no NIC, no new hardware.

**22.0 µs median round trip against TCP's 286 µs on the same cable, ~13× lower.**
On the 27B that is 9.16 t/s versus 8.83 over TCP — the fastest cross-node path measured
here. Full numbers: **[RESULTS.md](RESULTS.md)**.

The transport is [wkljohn/OdinLink-Five @ `strix-halo-verbs-fixes`](https://github.com/wkljohn/OdinLink-Five/tree/strix-halo-verbs-fixes) —
clone that branch and you have every fix documented here.

## Start here

```bash
git clone -b strix-halo-verbs-fixes https://github.com/wkljohn/OdinLink-Five.git
cd OdinLink-Five
cmake -B build -DBUILD_VERBS=ON -DBUILD_TRAY=OFF && cmake --build build -j$(nproc)
make -C driver
```

`patches/odinlink-verbs-and-driver-fixes.patch` is kept in sync for anyone who prefers
patching a pristine checkout. Both nodes must run the **same build** — the wire format
changed.

Then pick your path:

| you want | read |
|---|---|
| llama.cpp inference across two nodes (`-sm layer`) — **the working, fastest path** | **[REPRODUCE-RPC.md](REPRODUCE-RPC.md)** |
| speculative decoding (MTP / DFlash) on a hy_v3 target across the split | [SPECULATIVE-HY3.md](SPECULATIVE-HY3.md) |
| RCCL collectives over the same link | [REPRODUCE-RCCL.md](REPRODUCE-RCCL.md) |
| every number, with its caveats | [RESULTS.md](RESULTS.md) |
| the 23 defects found getting here, root cause + fix each | [FINDINGS.md](FINDINGS.md) |
| conclusions that were wrong and why | [APPENDIX-HISTORY.md](APPENDIX-HISTORY.md) |

## Status

| | |
|---|---|
| ✅ **Working** | 27B Q6_K across 2 nodes over OdinLink RDMA, `-sm layer`, **9.16 t/s** — faster than `thunderbolt_ibverbs` (9.07) and TCP (8.83) |
| ✅ **Byte-verified, one direction** | 21 GiB unidirectional, every byte checked |
| ❌ **Bidirectional is NOT reliable** | ~1 duplex run in 4 corrupts: two consecutive fragments lost, detected but not repaired. Reproduces on the frozen driver too. See [RESULTS.md](RESULTS.md) — **blocks collectives** |
| ⚠️ **LD_PRELOAD only** | OdinLink registers no kernel `ib_device`, so discovery is a preload shim. Anything not inheriting the preload cannot see it (BUG 11) |
| ⚠️ **Loss detected, not repaired** | Fragment sequencing makes a dropped fragment a loud error instead of silent corruption; there is no retransmit (BUG 22) |
| ⛔ **`-sm tensor` blocked over RDMA** | Not by RCCL — by the transport. Duplex loses messages (above), and RCCL assumes reliable delivery. Fix that first. BUG 25 is also still open |

## Why bother

Cross-node tensor parallelism is latency-bound. At 286 µs/op the per-layer all-reduce
dominates and pipeline (`-sm layer`) beats TP outright. At ~22 µs the comm term drops
~13×, which is the regime where TP can plausibly overtake pipeline. That is the whole
argument for this work, and it is still an argument rather than a result: the TP-over-RDMA
run has not been made. [REPRODUCE-RCCL.md](REPRODUCE-RCCL.md) says how.

Worth knowing before you assume RDMA settles it: the 2B TP result was *break-even* over
TCP, which means per-sync dispatch overhead alone roughly equals the whole butterfly cost.
Lowering the collective to 22 µs does not touch that overhead, so TP may still lose to
pipeline's 8.87. The measurement is worth making precisely because the answer is not
obvious.

For `-sm layer` the win is real but modest (+3.7 %): it crosses the wire once per token,
so ~22 µs sits against a ~110 ms/token budget. **Single node is still fastest for a model
that fits.** Two nodes are for capacity.

## Bandwidth is not the point

~9.2 Gb/s payload against a link the driver reports as 10 Gb/s × 2 lanes — roughly 46 %
of line rate, and *worse* than the ~19 Gb/s the bonded IP link gets. This is a latency
transport. The ceiling is a **USB4v1 cable**; the routers report `gen=4`, so a TB5 cable
should scale it.

## Scripts

| | |
|---|---|
| [`scripts/odl-bringup.sh`](scripts/odl-bringup.sh) | load + single-service bind + wait for READY |
| [`scripts/odl-measure.sh`](scripts/odl-measure.sh) | latency + bandwidth run |
| [`scripts/odl-reload.sh`](scripts/odl-reload.sh) | Thunderbolt stack reload — see the hazard note in [FINDINGS.md](FINDINGS.md) before using |
| [`odl_rdma_stress.c`](odl_rdma_stress.c) | byte-verifying conformance test (`--bidir`, `--latency`) |

## Before you load anything

**Keep a non-Thunderbolt path (LAN/Wi-Fi) to every node.** The IP bond shares these
cables; when the link misbehaves you will need another way in. Everything else worth
knowing before the first `insmod` is in [FINDINGS.md](FINDINGS.md) — including which
hazards were retracted and which still stand.
