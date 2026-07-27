# OdinLink RDMA on 2× Strix Halo — consolidated results

Every number in this repo lives here. All figures measured on 2× Ryzen AI MAX+ 395
(gfx1151), kernel 7.0.0-28, Ubuntu 26.04, ROCm 7.2.0, single USB4 cable, running
[wkljohn/OdinLink-Five @ `strix-halo-verbs-fixes`](https://github.com/wkljohn/OdinLink-Five/tree/strix-halo-verbs-fixes)
(equivalently, `patches/odinlink-verbs-and-driver-fixes.patch`).

Recipes: [REPRODUCE-RPC.md](REPRODUCE-RPC.md) (inference) ·
[REPRODUCE-RCCL.md](REPRODUCE-RCCL.md) (collectives). Defects:
[FINDINGS.md](FINDINGS.md).

## Inference — 27B Q6_K (20.88 GiB), 2 nodes, `-sm layer`

| transport | pp512 (t/s) | tg128 (t/s) | vs TCP |
|---|---|---|---|
| **RDMA (OdinLink + these patches)** | 290.68 ± 61.73 | **9.16 ± 0.02** | **+3.7 %** |
| RDMA (thunderbolt_ibverbs 0.3.4) | 290.23 ± 61.80 | 9.07 ± 0.01 | +2.7 % |
| TCP over bond0 | 292.11 ± 62.37 | 8.83 ± 0.03 | — |
| *single node (reference)* | *224.68* | *9.50 ± 0.05* | — |

`llama-bench -m Huihui-Qwen3.6-27B-abliterated-Q6_K.gguf --rpc <peer>:50052 -sm layer -ngl 99 -p 512 -n 128 -r 2`

**Single node is still fastest for a model that fits** in one 96 GB carve — two
nodes cost ~3.6 % because `-sm layer` crosses the wire once per token. Two nodes
are for *capacity*, not speed. RDMA recovers about half the cross-node penalty
(8.83 → 9.16 against a 9.50 ceiling).

> **Every inference figure on this page is `-sm layer` (pipeline).** Tensor
> parallel over RDMA — the case a 13× latency win should actually transform,
> since it all-reduces every layer — **has not been benchmarked**. Not because
> anything blocks it: RCCL binds the OdinLink plugin, and `-sm tensor` runs fine
> over TCP (27B at 3.65/4.02 t/s, in the repo root). The run pointing TP at the
> `ODL_TB5` plugin simply has not been done yet. See
> [REPRODUCE-RCCL.md](REPRODUCE-RCCL.md).

## Transport — byte-verified

| test | result |
|---|---|
| unidirectional bulk, 21 GiB | **86016/86016 messages verified**, 8.38 Gb/s |
| bidirectional, 2 GiB each way | **8192/8192 verified both peers**, 9.84 Gb/s full duplex |
| 256 MiB smoke | verified, 7.7 Gb/s |

Verification is byte-level with a position-dependent pattern
(`odl_rdma_stress.c`), so truncation, reordering, dropped fragments and stale
buffers all fail loudly. This matters: `llama-bench` measures speed, not
correctness, so a corrupting transport still prints a plausible t/s — and did,
before the bounce-buffer fix.

## Latency — CLI ping-pong RTT, 2000 iterations

| configuration | min | median | p95 | p99 | jitter σ |
|---|---|---|---|---|---|
| 10 µs fallback poll | 10.71 | 22.47 | 59.69 | 69.22 | 14.85 |
| **3 µs fallback poll** | 10.66 | **22.27** | **29.85** | 67.02 | **9.93** |
| + adversarial-review fixes | 10.54 | 21.97 | 43.36 | 67.61 | 11.59 |

vs **286 µs/op over TCP** on the same cable — roughly **13× lower**, and ~40×
lower than typical IP-over-Thunderbolt (~1 ms).

### Measurement caveat (read before trusting any tail number)

Across five runs in four configurations, **min and median are reproducible**
(median 22.0 ± 0.19 µs) while **p95/p99 swing 3×** under nominally identical
conditions. Single-run tail comparisons are not trustworthy. Median-based
conclusions are.

### Inline send fast path — the one change that measurably helps

Measured through the **verbs** path (`odl_rdma_stress --latency`, 1 KiB payload,
20000 iterations), A/B toggled with `ODL_VERBS_INLINE=0/1` at fixed payload size:

| metric | OFF | ON | delta |
|---|---|---|---|
| min | 15.20 | **13.27** | **−1.93 µs** |
| median | 22.58 | 22.50 | −0.08 µs |
| p95 | 33.97 | 33.78 | −0.19 µs |
| p99 | 34.66 | 34.31 | −0.35 µs |
| stddev | 12.86 | **6.22** | **−52 %** |

It improves the **floor and the jitter, not the median** — which is exactly the
expected signature. On an empty pipeline it removes a thread handoff, a
malloc+memcpy and two `poll()` syscalls; the typical case stays dominated by the
kworker hop it cannot touch. Default on.

Note the CLI latency tool cannot measure this at all — it talks straight to
ioctls and never enters the verbs shim.

### What does NOT reduce latency (all tested, median-based)

- Fallback poll timer 10 µs → 3 µs: median moved 0.2 µs. Tail only.
- Blocking CPU idle states (PM QoS, `/dev/cpu_dma_latency=0`): median unchanged
  at 22.06/22.10. Also rejected on merit — it blocks C1/C2/C3 leaving only POLL,
  so every core spins.
- Kernel busy-poll (`odl_busy_poll_us=50`): already enabled; median still 22 µs.
- NHI interrupt moderation: needs a kernel exporting `tb_ring_throttling()`;
  ours exports none, and the default is disabled anyway.
- `iommu=pt`: the TB controller is already on `DMA-FQ` (deferred invalidation)
  and the GPU on `identity`; the frame pool reuses the same buffers so IOTLB hit
  rate is ~100 %. Not worth losing the IOMMU, which is what caught a real
  use-after-free DMA bug in this driver.

The residual ~11 µs median-minus-min gap is most likely the double scheduler hop
(MSI-X → `schedule_work` → kworker → `wake_up_interruptible`), since rings are
allocated with `start_poll = NULL`. **That attribution is inference, not
measurement** — confirming it needs ftrace on `workqueue_queue_work` →
`workqueue_execute_start`. Note `thunderbolt-ibverbs` allocates its RX ring the
same way *deliberately*, to keep RX completion single-sourced for ordering.

## Reproducing

Full recipes in [REPRODUCE-RPC.md](REPRODUCE-RPC.md) and
[REPRODUCE-RCCL.md](REPRODUCE-RCCL.md). The transport check on its own:

```bash
# byte-verified bulk, both directions
gcc -O2 -o odl_rdma_stress odl_rdma_stress.c -libverbs -lpthread
# both sides: LD_PRELOAD=libodl_tb5_verbs.so ODL_RDMA_GID_IFACE=bond0
peer: ./odl_rdma_stress --server --local-ip 10.4.0.2 --total 21G --bidir
head: ./odl_rdma_stress --client 10.4.0.2 --local-ip 10.4.0.1 --total 21G --bidir
```
Exit codes: `0` ok, `2` data corruption, `3` stall.
