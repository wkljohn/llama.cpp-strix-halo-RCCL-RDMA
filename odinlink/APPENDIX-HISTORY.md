# Appendix — retractions and superseded conclusions

Kept deliberately. Several conclusions in this investigation were wrong, published,
and later withdrawn. The corrections are how the real answers were found, so deleting
them would make the work look tidier than it was and would strip the evidence that
justifies the current claims.

[FINDINGS.md](FINDINGS.md) carries the final diagnosis for each defect. This file
carries what was believed before.

---

## Retraction 1 — "never load this driver" (BUG 9)

**What was published:** that the driver hands the NHI unmapped addresses, that ring
buffers "must be mapped through the Thunderbolt device's DMA API", that this was a
confirmed upstream defect, and that readers should never load `odl_tb5` on a machine
they cared about.

**What was actually observed** — real, and seen repeatedly:

```
thunderbolt 0000:c7:00.6: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x003f
                                                address=0xffbb8000 flags=0x0020]
amdgpu 0000:c5:00.0: probe with driver amdgpu failed with error -22
drm_buddy_fini+0x112/0x120 [drm_buddy]
BUG: kernel NULL pointer dereference, address: 0000000000000000
```

The `IO_PAGE_FAULT` on the Thunderbolt controller comes first; the USB4 router wedges;
and because the router shares a firmware/power domain with the iGPU on Strix Halo,
`amdgpu` fails to initialise — sometimes on the *following* boot too, needing a full
power cycle.

**Why the root cause was withdrawn.** Direct inspection of `odl_tb5_rings_alloc()`
shows the buffers already allocated with

```c
dma_alloc_coherent(tb_ring_dma_device(dev->tx.ring), ...)
```

i.e. the correct API against the correct device. The stated defect does not exist.

**The confound.** The crashes were produced by a **locally patched build, not stock
upstream.** While adding the BUG 1 hop-ID fix, a bad patch to `odl_tb5_remove()`
silently deleted its entire cleanup body — `odl_tb5_rings_stop()`, six
`cancel_work_sync()` calls, `synchronize_rcu()`, `list_del_rcu()`, and every
buffer/ring free — leaving:

```
  ... state bookkeeping ... -> disable_paths/release_in_hopid -> kfree(dev)
```

So `dev` and its coherent ring buffers were freed **while the NHI rings were still
armed and six work items still held pointers to them.** DMA into freed-and-reallocated
memory is an exact mechanism for `IO_PAGE_FAULT` at a stale address and the wedge that
follows. Those runs **cannot distinguish** an upstream DMA defect from this local
use-after-free.

**Retest after restoring the upstream teardown order**, same hardware:

| test | result |
|---|---|
| `rmmod` (previously: "❌ never do this") | clean, <1 s, **3/3 cycles** |
| `insmod` → `rmmod` → `insmod` reload | no `enable_paths -12`, hop-ID correctly released |
| `IO_PAGE_FAULT` count | **0** |
| kernel oops / `BUG:` | **0** |
| GPU (`/dev/kfd`, `rocm-smi`) | healthy throughout, no reboot needed |

**Lesson:** when a kernel module starts corrupting DMA after you patched it, suspect
your patch before you suspect the platform.

---

## Retraction 2 — "`rmmod odl_tb5` is unsafe even with unbind-first ordering"

This escalation was written while the BUG 9 regression was still in the build:

> A later attempt used the "safe" sequence (unbind all services, sleep, then
> `rmmod odl_tb5` only, leaving the Thunderbolt core loaded). **It still crashed the
> machine.**
>
> ```
> RIP: 0010:check_config_address+0x8d/0xb0 [thunderbolt]
> RIP: 0010:tb_cfg_read+0xa5/0xf0 [thunderbolt]
> RIP: 0010:drm_buddy_fini+0x119/0x120 [drm_buddy]
> ```
>
> **Operational rule — do not unload this module.** Load `odl_tb5` exactly once per
> boot. To test a new driver build: reboot first, then `insmod`.

Withdrawn: the crash was the use-after-free above, not a property of module removal.
`rmmod` is clean on the repaired build.

**What survives:** reloading the *whole Thunderbolt stack* — unbinding the
`thunderbolt` PCI driver, not just `odl_tb5` — did wedge the USB4 controller and the
GPU PSP, and that was seen before the regression existed. It is a different operation
and was never disproven. Treat it as risky; use it once, not as routine recovery.

---

## Retraction 3 — the `ncclCommInitRank` failure was misattributed

**What was published:** that PR #20's plugin loads and is selected but
`ncclCommInitRank` fails, with the suspected cause being `connect()`/`accept()` retry
semantics differing from the PR's Ray-launched vLLM setup:

```
[Proxy Service] .../transport/net_tmp.cc:1006 -> 2      # 2 = ncclSystemError
ggml_cuda_world_init_once: ncclCommInitRank failed: unhandled system error
```

Ruled out at the time, correctly: device exclusivity, channel count, link state at
launch, ABI/symbol resolution.

**The correction.** Checking the link state immediately *after* a failing run showed it
had already left READY *before* the benchmark started:

```
node2:  enable_paths failed (-12) -> failed to enable XDomain paths after 5 attempts
node1:  DMA ping attempt 250, still waiting for pong / peer restarted
```

That is **BUG 1**. The plugin opens and closes the device on every comm setup; each
cycle restarts the link, and after enough cycles the peer exhausts hop-IDs. With the
link not READY, `get_shared_handle()`'s `odl_tb5_wait_peer(handle, 10000)` times out
and returns `rcclSystemError` — exactly the chain observed.

**Implication that mattered:** the hop-ID leak is not only an `insmod`-time problem,
which is why BUG 1 became the highest-value fix rather than a startup annoyance.

**Second-order problem, since resolved by fixing BUG 1 properly:** the stack-reload
workaround itself disturbs `thunderbolt_net`. After several reload cycles the IP bond
stopped passing traffic in both directions (interfaces `UP`, slaves `UP`, zero
`rx_packets`) while XDomain control messages still flowed — the fabric was fine, the
network driver had desynced. Recovering that needed a reboot.

---

## Retraction 4 — "the 2B model works over OdinLink"

Published with real benchmark output:

```
| qwen35 2B Q8_0 | 1.93 GiB | 1.94 B | ROCm,RPC | 99 | pp64 | 110.10 ± 0.00 |
| qwen35 2B Q8_0 | 1.93 GiB | 1.94 B | ROCm,RPC | 99 | tg32 |  76.22 ± 0.00 |
```

Invalidated by BUG 21: sends completed before the payload reached the wire, so the
weights uploaded were corrupt. `llama-bench` measures speed, not output correctness —
a corrupting transport still prints a plausible t/s, and this one did, at an impossible
979 MiB/s.

This is why every transport claim in [RESULTS.md](RESULTS.md) is byte-verified with
`odl_rdma_stress` rather than inferred from "the benchmark finished".

---

## Superseded progress snapshots

Three intermediate states were published as "where it stands now". Each was true when
written and is now historical:

| snapshot | claim at the time | superseded by |
|---|---|---|
| after BUG 11/13/14 | "discovered, opened, QP activated — does not yet complete a benchmark" | BUG 16/17/18 |
| after BUG 16/17/18 | "frames flowing both directions at µs cadence — still does not finish a benchmark", head blocked in `odl_poll_cq -> eventfd_read` | BUG 19 |
| after BUG 19 | "past device query into `buffer_set_tensor` — uploading weights, busy-polling normally. Now a *throughput* problem, not a liveness one" | BUG 20, 21, 22, 23 |

The progression, each step verified at the driver level:

| after fixing | head reached |
|---|---|
| — (upstream) | `ibv_devices` empty; nothing could find the device |
| BUG 11 | device opened, QP created — `RDMA probed` |
| BUG 13 | `RDMA activated: qpn=20->20 mtu=4096 rx_depth=24` on **both** nodes |
| BUG 16/17/18 | frames flowing both directions, µs cadence (was 5 s/message) |
| BUG 19 | past device query + buffer alloc into `buffer_set_tensor` — uploading weights |
| BUG 20/21/22/23 | 27B completes; byte-verified at 21 GiB |

---

## Superseded latency conclusion — "blocking C-states cuts p99 by 57 %"

One run showed p99 dropping from 67 µs to 28.76 µs with CPU idle states blocked via
PM QoS. The very next run, same configuration on two nodes, gave p99 64.12 µs. The
apparent win did not exist.

| run | min | median | p95 | p99 | jitter |
|---|---|---|---|---|---|
| 10 µs poll (baseline) | 10.71 | 22.47 | 59.69 | 69.22 | 14.85 |
| 3 µs poll | 10.66 | 22.27 | 29.85 | 67.02 | 9.93 |
| + adversarial-review fixes | 10.54 | 21.97 | 43.36 | 67.61 | 11.59 |
| C-states blocked (1 node) | 10.55 | 22.06 | **22.62** | **28.76** | 3.48 |
| C-states blocked (2 nodes) | 10.53 | 22.10 | 61.57 | 64.12 | 15.21 |

**Min and median are reproducible; p95/p99 are not.** The median holds at 22.0 ± 0.19 µs
across every configuration while the tails swing 3× under nominally identical
conditions. Any conclusion drawn from a single-run tail comparison is unreliable.

The change was also rejected on merit: writing 0 to `/dev/cpu_dma_latency` blocks every
state with non-zero exit latency — C1 (1 µs), C2 (18 µs) **and** C3 (350 µs) — leaving
only POLL. Every core spins, at maximum power and heat, on a machine that also runs
inference.
