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

## Tensor parallel over RDMA — measured, and it changes nothing

The measurement this whole effort was for. `-sm tensor`, 27B Q6_K, 2 nodes, `-ts 50/50`:

| collective transport | tg128 | pp512 |
|---|---|---|
| **TCP over bond0** | **3.50 ± 0.01** | **282.71 ± 2.81** |
| RDMA (OdinLink plugin) | 3.30 ± 0.01 | 216.73 ± 0.76 |
| *`-sm layer` (pipeline) over RDMA, for scale* | *9.20* | *246.69* |

**RDMA is slower — 6 % on decode, 30 % on prompt processing** — despite the collective
itself being **2.9× faster** over RDMA (100 µs vs 286 µs). Three repetitions each; the
±0.01 error bars put this well outside noise. Transport verified, not assumed:
`NCCL INFO Using network ODL_TB5`, zero `Using network Socket`, and the plugin logged
**69120 isend / 69120 irecv / 0 failures**.

> **Correction.** An earlier revision reported 3.33 vs 3.32 and called them "identical
> within noise". Those were **single runs** (`-r 1`) against llama-bench's default of 5 —
> the same single-run-as-property error this page retracts elsewhere for the bidirectional
> throughput claim. With three repetitions the difference is unambiguous and in the
> opposite direction from the project's hypothesis.

### Why a faster wire makes the system slower

The plugin busy-spins: a `poll(POLLOUT, 2 ms)` that returns ready every time and is
immediately re-issued, plus a 20 µs `nanosleep` receive loop. That burns CPU continuously.

In a path already dominated by **host-side dispatch** (4.13 ms per sync point, below),
CPU stolen from dispatch costs more than a 186 µs faster collective wins. The transport
gets faster and the system gets slower. This also explains the prompt-processing gap,
which is the most dispatch-heavy phase.

### Why a 2.9× faster collective bought nothing — the arithmetic

Without this, "we measured no gain" reads like a measurement error. It is a ceiling:

| term | value |
|---|---|
| Token budget at 3.33 t/s | **300 ms** |
| Compute, if TP split perfectly (half of single-node's 105 ms) | ~53 ms |
| **Everything else** | **248 ms/token = 4.13 ms per sync point** |
| What RDMA can address (60 syncs × 186 µs saved) | **11.2 ms = 3.7 %** |
| Predicted best case | 3.46 t/s |
| Measured | **3.33 / 3.32** |

**RDMA can only ever touch 3.7 % of the token budget**, which is indistinguishable from
run-to-run noise — exactly what was observed. The other **4.13 ms per sync point** is graph
dispatch, RPC subgraph serialisation and backend synchronisation: roughly **40× the entire
collective cost**.

Even a *zero-latency* collective would leave ~289 ms/token — still **2.6× slower than
pipeline's 109 ms**. Cross-node tensor parallelism cannot be rescued here by improving the
interconnect.

Two independent confirmations that this is structural rather than transport:

- The TCP-era A/B: butterfly **3.10** vs RCCL **3.65**. Two completely different collective
  mechanisms, both ≈3 t/s against pipeline's 8.87. Changing *how* ranks communicate barely
  moves the total.
- Pipeline at 109 ms/token is within 4 % of single-node's 105 ms — it crosses the wire once
  per token and pays almost nothing for it.

### Why a 2.9× faster collective bought nothing

The 2B result was *break-even* over TCP, which already said per-sync **dispatch** overhead
alone roughly equals the entire butterfly cost. RDMA lowers the collective; it does not
touch dispatch. So the term that dominates was never the one being optimised.

This is the honest headline: **RDMA over Thunderbolt makes cross-node collectives 2.9×
faster and cross-node tensor-parallel inference no faster at all.** Pipeline parallelism
remains ~2.8× ahead for this model. Anyone hoping a faster interconnect rescues `-sm tensor`
on this class of hardware should read that result before buying cables.

Where RDMA *does* pay: `-sm layer` at 9.20 vs 8.83 t/s over TCP, and the collective floor
itself, which matters for any workload dominated by all-reduce rather than dispatch.

### Getting `-sm tensor` to run at all

It deadlocked before producing a token. The peer was never in the collective — it was stuck
inside `ncclCommInitRank`:

```
ggml_cuda_world_init_once -> ncclCommInitRank_impl -> initTransportsRank
  -> bootstrapAllGather -> socketRingAllGatherUnidir -> recv()   [blocked]
```

Both ranks initialised the world lazily, inside the first `GGML_OP_ALLREDUCE`, and under
`-sm tensor` those moments are unrelated. `ggml_backend_cuda_world_init()` was exported with
**zero call sites**. Fixed by calling it eagerly on both ranks (detached thread on the peer,
so RPC serving is not blocked) and widening the rendezvous retry from 60 s to 5 min — the
peer must start first, but must then survive the head's ROCm init. Proven over TCP before
RDMA, so the fix stands independent of transport.

## RCCL collectives over RDMA — working

`test-world-allreduce`, 2 ranks, 60 all-reduces/graph, 24 KB each, single channel:

| transport | per all-reduce | correctness |
|---|---|---|
| **RDMA (OdinLink + this plugin)** | **100 µs** | `reduced to 1.50` ✅ |
| TCP over bond0 | 286 µs | `reduced to 1.50` ✅ |

**2.9× lower.** Proof the payload really crossed RDMA rather than falling back to sockets:
`bond0` TX moved only **13 KiB** across the entire run (bootstrap only), while the plugin
logged **2520 isend / 2520 irecv / 0 failures**. A t/s number alone is never evidence of
transport — three separate silent-fallback modes are documented in
[REPRODUCE-RCCL.md](REPRODUCE-RCCL.md).

This is the first cross-node RCCL collective carried by OdinLink. `ncclCommInitRank`
completes in ~2.9 s over the plugin, 12 channels connect, and the ABI is v7.

> ### Trap that cost hours: a stale test binary
>
> This ran "broken" for a long time — `out=1.000000 expect=1.500000` — and the failure was
> blamed on the plugin through four wrong hypotheses (NULL-request completion, wrong size
> reporting, `iflush`/APU cache coherency, channel-stream mapping). None were the cause.
>
> ```
> test-world-allreduce   built  Jul 20 23:19
> tests/test-world-allreduce.cpp modified Jul 21 00:33   <- 74 min NEWER than the binary
> libggml-hip.so.0       rebuilt Jul 27 15:14            <- 6 days newer, loaded at runtime
> ```
>
> An old test binary dynamically loading freshly-rebuilt ggml. One `cmake --build` fixed it.
>
> **What actually found it: running the same test over TCP.** It failed identically, which
> proved the transport was never implicated. That control should have been the *first* move,
> before reading a line of plugin source. When a component is suspected, test it against a
> known-good substitute before analysing its internals.

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

### Mechanism: a marginal cable, not a software defect

**Established.** Adding NHI error-flag checking to the receive callback (BUG 28) caught it
on the first failure, on both nodes at once:

```
RX CRC error (size=3520 flags=0xb) - frame dropped, total=1
stream 20 fragment gap: got 39 expected 38 (lost 1); rx crc=1 ovr=0 cancel=2048 ok=980073
```

**The hardware was reporting CRC errors and the driver was silently accepting the corrupt
frames.** Error rate ≈ 1 per 10⁶ frames, which predicts 0.125 failures per 512 MiB duplex
run — exactly the observed 1-in-8.

The "always exactly two fragments" signature was an *artifact of the missing check*: the
corrupt frame was consumed, advancing the sequence past it while the real frame was also
absent. With the check in place the gap became 1, not 2.

Hypotheses tested and killed on the way, recorded so nobody repeats them: chunk-size
threshold, send depth, batch-buffer overflow, and frame-pool starvation (`rx_repost_starved`
never fired).

### The other cable is clean

The rig has two USB4 cables; `max_devices=1` binds one. Rebinding both nodes to the second:

| | cable A (`0-2.1`) | cable B (`1-2.1`) |
|---|---|---|
| 512 MiB duplex runs | 7 pass / 1 fail | **10 pass / 0 fail** |
| new CRC errors | 1 per ~980k frames | **0** |

Ten runs is a real signal, not proof — at cable A's rate you would expect ~1.25 failures in
10. ~5 GiB of clean duplex is also far short of a 27B run's 20 GiB. Retransmission is still
absent (BUG 22), so a single bit error anywhere in a large transfer still means a hang.

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
