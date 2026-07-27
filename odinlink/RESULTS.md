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

## Transport

> ## ⚠️ Bidirectional transport is NOT reliable. Retracted 2026-07-27.
>
> This section previously reported *"bidirectional, 2 GiB each way — 8192/8192
> verified both peers, 9.84 Gb/s full duplex"* as a settled result. **It does not
> reproduce.** Re-measured across ~20 duplex runs, roughly **1 in 4 corrupts**.
>
> The original figure was a single passing run recorded as if it were a property.
> That is the same mistake this page warns about elsewhere, made here.
>
> The failure also reproduces on the frozen pre-change driver
> (`rdma-working-2026-07-27`), so it is not caused by the teardown fixes.

| test | result |
|---|---|
| unidirectional bulk, 21 GiB | **86016/86016 verified**, 8.38 Gb/s |
| unidirectional 512 MiB, 256 KiB chunk | verified, 7.91 Gb/s |
| 256 MiB smoke | verified, 7.7 Gb/s |
| **bidirectional 512 MiB, 256 KiB chunk** | **~1 run in 4 corrupts** (≈10 pass / 4 fail) |
| bidirectional 512 MiB, 64 KiB chunk | passed every run — but 64 KiB never enters batch mode (`THROUGHPUT_THRESH = 65536`), so this is a different code path, not evidence the bug is size-dependent |

**Unidirectional has not failed once.** Every failure to date needs duplex load.

### What the failure actually is

The driver's fragment sequencing detects it. Six gaps captured across both nodes:

```
lost 61,62   lost 56,57   lost 36,37   lost 23,24   lost 40,41   lost 50,51
```

**Always exactly two consecutive fragments, never one, never three** — and at
varying positions within the message, not at a fixed boundary. Independent of
`tx_depth` (1, 2 and 4 all lose exactly two) and of chunk size.

Loss is *detected* but not repaired — there is no retransmit (BUG 22) — so one
dropped message desynchronises the stream and the verifier then reports
corruption on later messages. That is why the reported offsets look erratic
(sometimes byte 0, sometimes mid-message): the first casualty is the message
that vanished, not the one that fails verification.

**Mechanism not yet established.** Two hypotheses have already been tested and
killed: chunk-size threshold (it is intermittent at every size above the batch
threshold) and batch-buffer overflow (that predicts loss at a fixed boundary;
the measured positions vary). A live candidate is that the RX callback never
examines NHI error flags — it checks only `frame->size` — so hardware-flagged
frames would be lost invisibly. Untested.

### Why this matters more for collectives than for RPC

`-sm layer` inference works over this transport because pipeline parallelism is
mostly one-directional and light per token, so it rarely meets the failure.
RCCL is far more duplex-heavy and assumes reliable delivery. **Do not treat this
transport as ready for collectives until duplex passes a real gate** — 20
consecutive clean 512 MiB duplex runs, not one lucky pass.

Verification is byte-level with a position-dependent pattern
(`odl_rdma_stress.c`), so truncation, reordering, dropped fragments and stale
buffers all fail loudly. That is what caught this. Note it proves *continuity*,
not integrity: the wire format carries no checksum.

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
