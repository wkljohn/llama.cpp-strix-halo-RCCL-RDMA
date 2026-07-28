# OdinLink-Five on Strix Halo — the bug ledger

Every defect found bringing RDMA-over-Thunderbolt up on 2× Ryzen AI MAX+ 395
(gfx1151, USB4 Host Router `1022:158d/158e`), kernel 7.0.0-28-generic, gcc 15.2.0.
All fixes are in [wkljohn/OdinLink-Five @ `strix-halo-verbs-fixes`](https://github.com/wkljohn/OdinLink-Five/tree/strix-halo-verbs-fixes)
— clone that and you have them; the credits and upstream base are stated there.

One entry per defect, each with its **final** diagnosis. Where a diagnosis was
revised mid-investigation, the entry states the conclusion and
[APPENDIX-HISTORY.md](APPENDIX-HISTORY.md) keeps the superseded reasoning —
retractions are evidence of how the answer was reached, not embarrassments to
delete.

Numbers live in [RESULTS.md](RESULTS.md). Recipes live in
[REPRODUCE-RPC.md](REPRODUCE-RPC.md) and [REPRODUCE-RCCL.md](REPRODUCE-RCCL.md).

## Index

| # | defect | effect | status |
|---|---|---|---|
| 1 | XDomain hop-ID / path leak on teardown | `enable_paths -12` on every load after the first | **fixed here** |
| 2 | Two cables ⇒ both services at `route=2` | login handshake never completes | workaround `max_devices=1` |
| 3 | `e2e=0` honoured but logged "enabled" | cosmetic; costs debugging time | reported |
| 4 | Plugins hardcode 80 Gb/s link speed | bad RCCL topology/chunking decisions | reported |
| 5 | Service unbind sends a peer logout | handshake churn | reported |
| 6 | RCCL plugin exports `rcclNetPlugin_v7` | plugin invisible; silent TCP fallback | fixed; also in PR #20 |
| 7 | Vendored `net_v7.h` matches no real ABI | segfault as soon as RCCL calls in | PR #20 |
| 8 | RCCL plugin data path is a stub | cannot move correct data | PR #20 |
| 9 | *(withdrawn)* IOMMU fault blamed on upstream | was a local use-after-free | **retracted** |
| 10 | DMA-verify handshake is an unwinnable race | second node **always** fails to reach READY | **fixed here** |
| 11 | No discovery path at all | `ibv_devices` empty — invisible to every RDMA app | **fixed here** |
| 12 | RCCL world-communicator rendezvous deadlock | rank 0 blocks in `accept()` before the peer is dispatched | **fixed** (dispatch order, in the patch) |
| 13 | `ibv_post_recv` performed a blocking receive | pre-posting always failed → `RDMA activate failed` | **fixed here** |
| 14 | `ibv_post_send` kept the caller's stack WR | worker read freed stack; completions unmatchable | **fixed here** |
| 15 | `modify_qp` discarded `IBV_QP_DEST_QPN` | sends addressed to `dst_id 0` | **fixed here** |
| 16 | `cmd_fd` assigned then overwritten with `-1` | `poll()` always `-EBADF`; worker busy-spun | **fixed here** |
| 17 | `POLLOUT` required a *previous* TX to complete | 5 s per message, measured | **fixed here** |
| 18 | Verify clears the proof a peer PING provided | stuck `CONNECTED`; every send `-EAGAIN` | **fixed here** |
| 19 | `ibv_poll_cq` blocked holding the CQ lock | self-deadlock against its own producer | **fixed here** |
| 20 | Full-size frames exactly equal the RX buffer | **every multi-frame message lost all full fragments** | **fixed here** |
| 21 | Send completed before the payload was on the wire | silent data corruption at 979 MiB/s | **fixed here** |
| 22 | No fragment sequencing in the wire header | a dropped fragment becomes a silent short message | **fixed here** |
| 23 | Bidirectional traffic deadlocks (three causes) | the 27B blocker | **fixed here** |
| 24 | Destroying a stream never wakes its receive waiters | closing an RCCL comm hung forever | **fixed here** |
| 25 | Plugin `connect`/`accept` return errors instead of retry | violates the RCCL v7 non-blocking contract | **open** |
| 26 | `remove()` never returns the `max_devices` slot | one peer restart wedges the link until a module reload | **fixed here** |
| 27 | Duplex traffic loses two consecutive fragments | ~1 run in 4 corrupts; **blocks collectives** | **open** |
| 28 | RX callback ignores NHI error flags | hardware-flagged frames accepted as valid | **open** |
| 29 | Verbs TX worker busy-spins when idle | steals CPU; makes RDMA *slower* than TCP for dispatch-bound workloads | **open** |

"Fixed here" means the patch is on the
[`strix-halo-verbs-fixes` branch](https://github.com/wkljohn/OdinLink-Five/tree/strix-halo-verbs-fixes)
and in `patches/odinlink-verbs-and-driver-fixes.patch`. "PR #20" refers to
[OdinLink PR #20](https://github.com/Geramy/OdinLink-Five/pull/20), an independent
fix for the same defects, cited for attribution.

---

# BUG 1 — XDomain hop-ID / path leak → `ENOMEM` on every load after the first

**Severity: high.** This is the one that used to force reboots.

The first `insmod` after a boot reaches `entering READY state`. Every subsequent
load — or service re-probe, e.g. after the peer reboots — fails:

```
OdinLink: enable_paths failed (-12), retry 1..5
OdinLink: failed to enable XDomain paths after 5 attempts: -12
```

`-12` = `ENOMEM` from `tb_xdomain_enable_paths()`: the Thunderbolt core is out of
hop-ID/tunnel resources because previous instances never released theirs.

**Root cause A — `remove()` releases only in two states** (`odl_tb5_service.c:209`):

```c
if (saved_state == ODL_TB5_STATE_CONNECTED ||
    saved_state == ODL_TB5_STATE_READY) {
        tb_xdomain_disable_paths(...);
        tb_xdomain_release_in_hopid(dev->xd, dev->remote_tx_hopid);
}
```

`tb_xdomain_alloc_in_hopid()` is called in `odl_tb5_complete_connection()`. If the
device is removed while in **HANDSHAKE**, **DISCONNECTED** or **ERROR** — exactly
what happens when a handshake is retrying, the peer disappears, or a service is
unbound — the hop ID and paths are **never released**.

**Root cause B — the restart path releases a possibly-stale hop ID.**
`odl_tb5_proto.c:601-608` releases `dev->stale_remote_tx_hopid`, but that field is
only assigned in the peer-login handler. A restart triggered by DMA-verify failure
does not pass through those lines, so the wrong ID is released and the live one leaks.

**Root cause C — `complete_connection()` re-entry.** The function calls
`tb_xdomain_alloc_in_hopid()` unconditionally at its head and overwrites
`dev->in_hopid`. It is re-entered on **every** handshake restart, so each retry
orphans the previous allocation. Tracking `in_hopid_valid` alone was not enough —
a node still hit `enable_paths -12` after a handful of restarts.

**Fix.** Track the allocation explicitly, release on every teardown path, and
release *before* re-allocating on re-entry:

```c
/* odl_tb5_core.h */
bool in_hopid_valid;
int  in_hopid;            /* the value actually passed to alloc_in_hopid() */

/* one helper used by remove(), restart, and every error path */
static void odl_tb5_release_paths(struct odl_tb5_device *dev)
{
        if (!dev->in_hopid_valid)
                return;
        if (dev->tx.started)
                tb_xdomain_disable_paths(dev->xd, dev->local_tx_hopid,
                                         dev->tx.ring ? dev->tx.ring->hop : -1,
                                         dev->in_hopid,
                                         dev->rx.ring ? dev->rx.ring->hop : -1);
        odl_tb5_rings_stop(dev);
        tb_xdomain_release_in_hopid(dev->xd, dev->in_hopid);
        dev->in_hopid_valid = false;
}

/* complete_connection(), at the head — re-entry drops the old one first */
if (dev->in_hopid_valid)
        odl_tb5_release_paths(dev);
ret = tb_xdomain_alloc_in_hopid(dev->xd, dev->remote_tx_hopid);
```

Releasing when nothing is allocated is a no-op, so the helper is safe everywhere.
Apply it **after** the existing teardown body, never in place of it — replacing it
is what produced the retracted BUG 9 (see [APPENDIX-HISTORY.md](APPENDIX-HISTORY.md)).

**Scope note.** The leak is not only an `insmod`-time problem. *Any* workload that
opens and closes the device repeatedly exhausts hop-IDs — which is what made a
long-running RCCL job fail with `ncclSystemError` before this was fixed.

---

# BUG 2 — multi-domain route collision breaks the login handshake

**Severity: high** on any host with two or more Thunderbolt cables to the same peer.

With both USB4 cables connected each side *receives* the peer's login and *sends* a
response with `ret=0`, but its own request always times out:

```
OdinLink: received login from peer (version=1, tx_path=9, size=48)
OdinLink: sent login response (ret=0, route=2, sn=0, tx_hopid=9)
OdinLink: login request failed: -110          <-- ETIMEDOUT, forever
```

`odl_tb5_find_device_by_route(u64 route)` (`odl_tb5_proto.c:78`) resolves the target
device by route **alone**. With two USB4 host routers both peer services sit at
`route=2`:

```
0-2.1  prtcid=20300  route=2      <- domain 0
1-2.1  prtcid=20300  route=2      <- domain 1
```

so the lookup returns whichever matches first and the response is dispatched onto
the wrong xdomain.

**Suggested fix** — match on the `tb_xdomain` pointer, which the XDomain callback
already has:

```c
static struct odl_tb5_device *
odl_tb5_find_device(const struct tb_xdomain *xd)
{
        list_for_each_entry(dev, &odl_tb5_devices, list)
                if (dev->xd == xd)
                        return dev;
        return NULL;
}
```

If route matching must be kept, key on `(tb_xdomain_parent(xd)->index, route)`.

**Workaround.** Bind exactly one service *from the start*, via the module parameter
added here — `max_devices=1`. Unbinding afterwards works but triggers BUG 5.

---

# BUG 3 — `e2e=0` is honoured but logged as "enabled" (cosmetic)

`driver/odl_tb5_ring_dma.c:476` prints `(E2E enabled, e2e_tx_hop=%d)`
unconditionally. The parameter itself works correctly, but the log claims otherwise
— which costs debugging time when following the module's own `e2e=0` advice for
TB3-class controllers.

```c
"local_tx_hopid=%d (E2E %s, e2e_tx_hop=%d)\n", ..., odl_e2e ? "enabled" : "disabled", ...
```

---

# BUG 4 — RCCL/NCCL plugins report a hardcoded 80 Gb/s

`rccl/src/odl_tb5_plugin.c:184` and `nccl/src/odl_tb5_nccl_plugin.c:343`:

```c
props->speed = 80000;      /* TB5 spec number, not measured */
```

The driver itself reports the true link as **10 Gb/s × 2 lanes** and the measured
payload rate is **~9.2 Gb/s** — off by ~9×. RCCL feeds `props->speed` into topology
and cost estimation (ring vs tree, chunk sizing), so an inflated value produces bad
algorithm decisions.

**Fix:** query the real speed, already exposed through the `GET_PEER` ioctl
(`struct odl_tb5_peer_info.speed`), and fall back to a conservative default only if
the ioctl returns 0.

---

# BUG 5 — unbinding a service triggers a peer logout (minor)

Unbinding one service (the old BUG 2 workaround) sends a logout that resets the
peer's state machine, so both sides re-run the handshake. Combined with BUG 1 this
leaked another hop ID each time. Using `max_devices=1` avoids the unbind entirely.

---

# BUG 6 — RCCL net plugin exports the wrong symbol name (FIXED)

**Severity: high.** Makes the plugin invisible; RCCL silently falls back to TCP.

```
NCCL INFO External network plugin /path/librccl-net-ODL_TB5.so is unsupported
NCCL INFO NET/Socket : Using [0]bond0:10.4.0.1<0>      <-- silent TCP fallback
```

`rccl/src/odl_tb5_plugin.c:375` defines the entry point as **`rcclNetPlugin_v7`**.
RCCL — like NCCL — resolves net plugins by **`ncclNetPlugin_v<N>`** (`strings
librccl.so.1` lists `ncclNetPlugin_v6..v11`, no `rccl*` variants). The symbol never
matches and the plugin is rejected **without an error**.

```c
extern rcclNet_v7_t ncclNetPlugin_v7 __attribute__((alias("rcclNetPlugin_v7")));
```

See [REPRODUCE-RCCL.md](REPRODUCE-RCCL.md) for the two *further* discovery traps
(filename form, and `NCCL_PLUGIN_DIR` not being honoured) that also end in silent TCP.

---

# BUG 7 — the vendored net API does not match the real NCCL/RCCL ABI

**Severity: critical.** Once BUG 6 is fixed and the plugin actually loads, both ends
**segfault** as soon as RCCL calls into it.

`third_party/rccl/net_v7.h` defines a struct matching **no** real NCCL/RCCL version.
Against the genuine `ncclNet_v7_t`:

| slot | OdinLink `rcclNet_v7_t` | real `ncclNet_v7_t` |
|---|---|---|
| 7 | `closeListen` | **`regMr`** |
| 8 | `isend` | **`regMrDmaBuf`** |
| 9 | `irecv` | **`deregMr`** |
| — | *(absent)* | `getDeviceMr`, `irecvConsumed` |

Signatures differ too: real `isend` takes an `mhandle`; real `irecv`/`iflush` are
multi-buffer. So RCCL calls `regMr` at the offset holding `closeListen` → immediate
crash. The NCCL-side plugin declares only `_v4`/`_v5`, which modern RCCL will not load.

**Fix:** rebuild against the official headers from `NVIDIA/nccl`
(`plugins/net/example/nccl/net_v*.h`), targeting v7 or newer. Upstream **PR #20**
does this.

---

# BUG 8 — the RCCL plugin's data path is a non-functional stub

**Severity: critical.** Even with BUG 6 + 7 fixed the plugin cannot move correct data.

1. **`isend`/`irecv` treat the data pointer as a file descriptor**
   (`odl_tb5_plugin.c:290,315`):
   ```c
   odl_tb5_send_dmabuf(comm->handle, (int)(intptr_t)data, 0, size);
   ```
   `odl_tb5_send_dmabuf()` expects a dmabuf FD; NCCL passes a memory pointer.
2. **`test()` never polls for completion** — it unconditionally sets `*done = 1`, so
   NCCL believes every transfer finished instantly.
3. `iflush` is a no-op with a non-standard signature.

The suggested alternative in upstream's docs — `NCCL_NET_PLUGIN=IB` with
`NCCL_IB_HCA=odl_tb5` — does not work either: the driver registers no kernel RDMA
device, and substituting `libodl_tb5_verbs.so` for `libibverbs.so.1` fails because
RCCL requires versioned symbols (`IBVERBS_1.1/1.8/1.10/1.12`) it does not export.

**Superseded by PR #20**, found open on the same hardware class. It rewrites the plugin
host-staged (`NCCL_PTR_HOST`) with a per-connection FIFO worker, fixes the ABI
including a `char[128]`-vs-`char*` properties defect not identified here, and raises
the driver's per-stream `rx_queue_max` from 256 to 65536, so a backlog no longer makes
`odl_tb5_rx_callback()` **silently drop** work under duplex load.

*Correction:* earlier revisions here called that a **fragment** limit and repeated PR #20's
"a ~1 MB message chunks into ~264 frames" reasoning. In this tree the limit applies to
**completed messages** enqueued after reassembly, not to individual fragments. The old 256
cap could still drop completed messages under backlog, but the per-fragment framing of the
explanation was wrong for this source.
PR #20 reports TP=2 Llama-3.1-8B beating TCP by ~17–26 % on decode.

---

# BUG 9 — WITHDRAWN: the IOMMU fault was a local use-after-free

Earlier revisions of this file asserted that the driver hands the NHI unmapped
addresses, and told readers never to load it. **Both claims were wrong.**
`odl_tb5_rings_alloc()` uses `dma_alloc_coherent(tb_ring_dma_device(...))` — the
correct API against the correct device.

The crashes were produced by a **locally patched build**: while adding the BUG 1 fix,
a bad patch to `odl_tb5_remove()` silently deleted its entire cleanup body, freeing
the device and its coherent DMA rings while the NHI was still transferring into them.
That is an exact mechanism for `IO_PAGE_FAULT` at a stale address, the USB4 router
wedge that follows, and — because the router shares a firmware/power domain with the
iGPU on Strix Halo — the `amdgpu` probe failure on the next boot.

With the upstream teardown order restored, **3/3 `rmmod` + `insmod` cycles**: no
`IO_PAGE_FAULT`, no oops, no `enable_paths -12`, GPU healthy, no reboot needed.

The correct order — apply any hop-ID fix **after** the rings are stopped:

```
send_logout -> cancel_work_sync x6 -> rings_stop -> synchronize_rcu -> list_del_rcu
   -> dma_bufs_free -> rings_free -> disable_paths -> release_in_hopid -> kfree
```

Full retraction, the original claim, and the hazards that survive it are in
[APPENDIX-HISTORY.md](APPENDIX-HISTORY.md). **Lesson worth keeping:** when a kernel
module starts corrupting DMA after you patched it, suspect your patch before you
suspect the platform.

---

# BUG 10 — DMA verify handshake is an unwinnable race (FIXED)

**Severity: blocker.** This is why two nodes never both reached READY, and unlike
BUG 9 it is a real upstream bug.

One node reaches READY instantly. The other spins ~31 s and dies:

```
OdinLink: DMA ping attempt 300, still waiting for pong
OdinLink: DMA verify failed after 300 attempts
```

The link is *perfectly healthy* — the failing node has already received and answered
the peer's PING. It just never gets one back.

`odl_tb5_verify_work_fn()` succeeds on the first node, which then does:

```c
odl_tb5_rings_reset(dev);          /* drops every posted RX frame */
dev->state = ODL_TB5_STATE_READY;
dev->rx_target = 0;
atomic_set(&dev->rx_posted, 0);    /* and posts NO new ones */
```

Pool RX repost does not restart until userspace issues a `STREAM_OPEN` ioctl. So from
the moment the first node goes READY it has **zero RX frames posted** and silently
drops everything the peer sends — 100 % of the time, because the winner always resets
before the loser's verify loop completes. Measured asymmetry that identified it: the
winner received **0** frames after READY while the loser sent **300**; before the flip,
delivery was 300/300.

**Fix.** A node that *answers* a PING has already proven the path both ways — it
received a frame and successfully submitted one. It must not wait for a PONG that by
construction will never arrive:

```c
if (type == ODL_TB5_DMA_PING) {
        ret = odl_tb5_send_dma_msg(dev, ODL_TB5_DMA_PONG);
        /* Answering a PING proves RX and TX both work. */
        if (!ret && !dev->pong_received) {
                dev->pong_received = true;
                wake_up_all(&dev->verify_waitq);
        }
}
```

Both nodes now reach READY in **~0.4 ms**, first attempt. See BUG 18 for the
regression this fix introduced and how it was closed.

*Alternative for upstream:* keep a few RX frames posted in READY so ctrl frames stay
serviceable. Cleaner, but it changes buffer accounting the legacy daemon/CLI depend on.

---

# BUG 11 — no discovery path; OdinLink is invisible to every RDMA app (FIXED)

**Severity: blocker for any real use.** Two independent defects.

**1. Stale ABI symbol.** `verbs/src/odl_tb5_provider_plugin.c` resolves
`verbs_register_driver_34`. rdma-core 61 (Ubuntu 26.04) exports only
`verbs_register_driver_59@@IBVERBS_PRIVATE_59`. The `dlsym` returns NULL, the
constructor warns to stderr where nobody sees it, and the provider silently never
registers. The build also emits `libodl_tb5-rdmav34.so` while every system provider
is `-rdmav59`, so rdma-core's directory scan skips it regardless.

**2. No kernel RDMA device.** rdma-core enumerates from `/sys/class/infiniband/*` +
`/dev/infiniband/uverbsN`. OdinLink's module creates only a char device
(`/dev/odl_tb5_0`) and registers nothing with `ib_core`, so `ibv_devices` stays empty
and `match_device`/`alloc_device` are never called.

Every consumer — llama.cpp's `ggml-rpc` RDMA transport, RCCL's IB transport, every
`perftest` tool — starts with `ibv_get_device_list()` → `ibv_get_device_name()` →
`ibv_open_device()` → `ibv_query_port()` → `ibv_query_gid_ex()` to match a GID
against the local IP. None of that could resolve.

**Fix here:** supply the discovery half of the API in the existing LD_PRELOAD shim —
new file `verbs/src/odl_tb5_verbs_discovery.c`:

- `ibv_get_device_list` / `ibv_free_device_list` / `ibv_get_device_name` — advertise
  `odl_tb5_N` **alongside** (not instead of) real adapters, which are forwarded to the
  real libibverbs so a Mellanox card in the same box keeps working.
- `ibv_query_gid` **and** `_ibv_query_gid_ex` — synthesise a RoCE v2 GID as the
  IPv4-mapped local address (`::ffff:a.b.c.d`, from `ODL_RDMA_GID_IFACE`, default
  `bond0`). Both forms must be interposed: `ibv_query_gid_ex` is `static inline` and
  forwards to the exported `_ibv_query_gid_ex`, while plain `ibv_query_gid` is called
  separately by `rdma_probe()` — and the real libibverbs implementation **segfaults**
  on an OdinLink context, since it walks provider-private state that does not exist.

The data-path verbs (`ibv_post_send`, `ibv_poll_cq`) are `static inline` and dispatch
through `context->ops`, which `odl_init_context_ops()` already fills — no interposer
needed.

`ibv_devices` now lists `odl_tb5_0` on both nodes. **This is a bridge, not a complete
integration:** the proper fix is `ib_register_device()` in the kernel module, and until
that exists every consumer needs the `LD_PRELOAD`.

---

# BUG 12 — RCCL world-communicator rendezvous deadlock (FIXED)

This one is in **this repo's llama.cpp port**, not in OdinLink.

**The mechanism.** `ggml_cuda_world_init_once()` blocks in `accept()` on rank 0. Backend 0
is the local CUDA backend, whose `graph_compute_async` runs the allreduce **inline** on the
host thread; the RPC backends dispatch fire-and-forget. So if the local backend is
dispatched first, rank 0 blocks in `accept()` *before* the peer's graph has been sent — the
peer never executes an allreduce, never calls `world_init`, never connects back. Structural
and deterministic at any world size, not a race.

**The fix is in `rccl-tp-port.patch`** — `allreduce_world()` in `ggml-backend-meta.cpp`
iterates backends high→low so the RPC ranks ship the peer its allreduce *before* the
blocking local rank 0 runs:

```c
// Iterate high->low (n-1 .. 0) so RPC ranks go before the blocking local rank 0.
for (size_t jj = n_backends; jj-- > 0; ) {
    ggml_backend_graph_compute_async(backend_ctx->backend_configs[jj].backend, step_cgraphs[jj]);
}
```

> **Correction.** Earlier revisions of this file called BUG 12 open and said it "reproduces
> every time", and other pages here cited it as the reason no `-sm tensor` RDMA numbers
> exist. That was wrong on both counts, and self-contradictory: this repo publishes
> **measured** TP-over-TCP results (27B at 3.65/4.02 t/s), which could only come from
> completed TP runs. Two independent source reviews caught it.
>
> **`-sm tensor` over RDMA has since been run.** It needed one more fix — eager world init,
> because both ranks called `ncclCommInitRank` lazily at unrelated moments and deadlocked in
> RCCL's bootstrap all-gather. Result: TCP 3.50 ± 0.01 beats RDMA 3.30 ± 0.01 t/s. See
> [RESULTS.md](RESULTS.md).

**What remains.** `docs/REPRODUCE.md` records an *intermittent* 27B startup hang. Its cause
is not determined from source. The likely hardening is eager world init: the exported
`ggml_backend_cuda_world_init()` currently has **zero call sites**, so initialisation is
lazy on first allreduce. Calling it at startup — in `rpc-server.cpp` `main`, and on the head
when `GGML_NCCL_RANK` is set — makes both ranks arrive at the rendezvous deterministically.

---

# BUG 13 — `ibv_post_recv` had inverted semantics (FIXED)

`odl_post_recv()` did not post a buffer; it **performed a receive inline**, blocking
in `poll(fd, 5000)`:

```c
int pr = poll(&pfd, 1, 5000);
if (pr <= 0 || !(pfd.revents & POLLIN)) { *bad_wr = wr; return pr == 0 ? -ETIMEDOUT : -EAGAIN; }
```

Every verbs consumer **pre-posts** receive buffers before any traffic exists —
llama.cpp posts `RDMA_RX_DEPTH` (24) between the INIT and RTR transitions, and
RCCL/perftest do the same. So the first post always failed and setup aborted with
`RDMA activate failed, staying on TCP`. A design-level incompatibility, not a tuning
issue.

**Fix:** a real receive queue. `post_recv` copies the WR into a ring and returns
immediately; a worker drains inbound stream data into posted buffers and posts the
completions (`odl_rq_drain()`).

---

# BUG 14 — send path stored pointers to caller stack memory (FIXED)

`odl_post_send()` stored the caller's `struct ibv_send_wr *` in `qp->sq[]` and the
worker dereferenced it later. Callers legitimately use stack storage, because
`ibv_post_send` is defined to consume the WR before returning:

```c
struct ibv_sge sge = {};
struct ibv_send_wr wr = {}, *bad = nullptr;
if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;   // wr dies at scope exit
```

The worker was reading freed stack frames.

**Fix:** copy `wr_id`/`addr`/`length`/`lkey`/`num_sge` at post time. The same change
sets `wc.wr_id` on send completions, which was never populated — consumers that match
completions by `wr_id` (llama.cpp uses the chunk sequence number) could not have worked.

---

# BUG 15 — `modify_qp` discarded `IBV_QP_DEST_QPN` (FIXED)

The RTR transition dropped the destination QP number, so every send was addressed to
`dst_id 0` and landed nowhere. Fixed by honouring the attribute and plumbing it to the
stream's destination ID.

---

# BUG 16 — `cmd_fd` clobbered immediately after being set (FIXED)

`odl_ibv_open_device()` set up the device fd; the "Initialize context fields" block
four lines later overwrote it:

```c
    ctx->base.cmd_fd = dev_fd;     /* fd stored */
}
/* Initialize context fields */
ctx->base.cmd_fd        = -1;      /* ...and immediately thrown away */
```

`odl_worker_poll_fd()` therefore always returned `-EBADF`, the worker never waited for
TX readiness, and it busy-spun on `send EAGAIN, re-queueing` at ~1000 log lines/second.
**Fix:** assign the fd *after* the defaults.

---

# BUG 17 — `poll()` POLLOUT was a chicken-and-egg (FIXED)

```c
/* Writable if TX has room */
if (atomic_read(&dev->tx.completed) > 0)
    mask |= EPOLLOUT | EPOLLWRNORM;
```

TX counted as writable only once a previous TX had *completed* — never true before the
first send. Every non-blocking sender burned the full 5 s poll timeout per message,
visible in the driver log as TX submissions exactly 5 s apart:

```
[11909.495468] TX submit stream=20 dst=20 len=1
[11914.501045] TX submit stream=20 dst=20 len=8     <- +5.0 s
```

**Fix:** POLLOUT must mean "a send would succeed now" — the same condition
`odl_tb5_stream_can_send()` applies. After the fix the same trace shows microsecond
spacing (`+10 µs`, `+4 µs`).

---

# BUG 18 — verify clears the proof a peer PING already provided (FIXED)

A regression introduced by this repo's own BUG 10 fix.
`odl_tb5_ctrl_reply_work_fn()` sets `pong_received = true` when it answers a peer PING,
but if that PING arrives *before* the local `odl_tb5_verify_work_fn()` starts — the
common case when the peer loaded first — verify's first statement wipes it:

```c
dev->pong_received = false;   /* discards the proof we already had */
```

Verify then times out, the device stays `CONNECTED` (state 2) instead of reaching
`READY` (state 4), and since `odl_tb5_stream_can_send()` requires READY, **every send
returns `-EAGAIN` forever**:

```
odl_tb5: can_send=0 state=2 free=1024 reserve=64 rx_target=512 rx_posted=0
```

Note `free=1024` — the frame pool was completely idle; state was the only failing term.

**Fix:** a separate `peer_ping_answered` flag that verify does not clear (reset only
when a new connection begins), honoured by the wait condition and the final check.

---

# BUG 19 — `ibv_poll_cq` deadlocked against its own completion producer (FIXED)

```c
/* Clear eventfd if we drained the ring */
if (ocq->head == ocq->tail) {
    eventfd_t val;
    eventfd_read(ocq->eventfd_fd, &val);
}
pthread_mutex_unlock(&ocq->lock);
```

`ibv_poll_cq()` is defined to be non-blocking — consumers busy-poll it. This
implementation blocked in `eventfd_read()` on an empty CQ **while holding `ocq->lock`**,
and `odl_cq_post()` needs that same lock to deliver a completion. The only thread that
could wake the poller was locked out of doing so:

```
odl_poll_cq -> eventfd_read (blocked)
  <- rdma_poll <- rdma_recv <- send_rpc_cmd <- ggml_backend_rpc_get_device_memory
```

**Fix:** drain the eventfd outside the lock, only when completions were actually
consumed, and force `O_NONBLOCK` at CQ creation rather than trusting the flag. The
empty-CQ path must do **no** syscalls at all — an `fcntl` + `eventfd_read` per poll
iteration dominates the transfer and is itself a (softer) failure mode.

---

# BUG 20 — full-size frames silently dropped (FIXED — the bulk blocker)

```c
#define ODL_TB5_STREAM_PAYLOAD_MAX   (ODL_TB5_FRAME_SIZE - ODL_TB5_STREAM_HDR_SIZE)
```

A maximally-filled frame is therefore **exactly** `ODL_TB5_FRAME_SIZE` (4096) bytes.
The RX ring buffers are the same 4096 bytes, and a Thunderbolt ring frame carries
framing/CRC overhead beyond the payload the driver writes — so full frames do not fit
and the NHI drops them **silently**, with no error at any layer.

Only sub-maximal frames were ever delivered. That is why the link looked perfectly
healthy: the RPC handshake's 1–16 byte messages are single frames and worked, both
peers reached `RDMA activated`, and `ibv_devinfo` was happy. But every *multi-frame*
message lost all of its full fragments and kept only the short tail, so reassembly
never completed and the receiver waited forever.

Captured directly — an 8496-byte message sends as `4091 + 4091 + 314`, and the peer
logs only the tail:

```
odl_tb5: RXF-BIG dst=20 src=20 plen=314 flags=0x02 stream=yes    <- nothing else
```

**Fix:** reserve `ODL_TB5_FRAME_TAIL_RESERVE` (64 B) so header + payload stays strictly
below the buffer size, applied to **both** the driver uapi and the userspace ioctl
header so fragmentation matches on each side. Both nodes must run the same build.

---

# BUG 21 — send completed before the payload reached the wire (FIXED)

The first byte-verifying stress run exposed a bug every `llama-bench` run had hidden:

```
client: PASS: 1024 msgs, 256.0 MiB in 0.26s = 979.2 MiB/s
server: CORRUPTION at msg 0, first bad word 0 (byte 0 of 262144)
```

`ibv_post_send()` must consume the payload before returning — callers reuse the buffer
immediately. `odl_tb5_stream_send()` only queued it, so the worker DMA'd whatever the
caller had since written, and the completion fired before the bytes reached the wire
(hence the impossible 979 MiB/s).

**Fix:** bounce-copy the payload at post time.

**This invalidated an earlier "2B model works" result.** `llama-bench` measures speed,
not output correctness, so a corrupting transport still prints a plausible t/s — and
did. After the fix, 256 MiB verifies clean at 7.7 Gb/s.

---

# BUG 22 — no fragment sequencing in the wire header (FIXED)

At 27B scale the transport survived **18.5 GiB** and then failed:

```
recv+verified 73728/86016 (18432.0 MiB)          <- 18.5 GiB byte-perfect
SHORT msg 73956: got 44686 bytes, expected 262144
CORRUPTION at msg 73956, first bad word 1006 (byte 8048 of 44686)
```

Byte 8048 is almost exactly two payload fragments (2 × 4027), so the first two arrive
correct and reassembly then desyncs — a dropped fragment. The root cause is the wire
format:

```c
struct odl_tb5_stream_hdr {
    __u8 src_id; __u8 dst_id; __u8 flags; __le16 payload_len;
} __packed;                      /* flags: MSG_START | MSG_END only */
```

A multi-frame message is reassembled by **blind concatenation** gated only by
START/END. No fragment index, no offset, no message length, no checksum — so if any
fragment is lost the receiver cannot tell. It concatenates fewer bytes and delivers a
short, silently corrupt message at MSG_END.

**Fix:** a fragment index in the stream header (grown to 8 bytes), contiguity checked
at reassembly, and a gap reported as an error with the damaged message dropped. This
**detects** loss; it does not retransmit. A 20.88 GiB model transfer needs ~86 000
messages and ~5.5 million fragments, so undetected loss was a coin flip before this.

Remaining work, in order: (1) attribute the drop source — 18.5 GiB clean implies a
rare condition (RX repost starvation, frame-pool exhaustion, NHI ring wrap), not a
systematic bug; (2) then consider NAK/retry, which sequencing now makes possible.

---

# BUG 23 — bidirectional traffic deadlocks (FIXED — the 27B blocker)

Unidirectional bulk ran 21 GiB flawlessly while llama.cpp still died on the 27B. The
distinguishing property: ggml-rpc is **request/response on one QP**, so both peers send
and receive concurrently. `odl_rdma_stress --bidir` reproduces it in ~25 s and found
three separate causes.

1. **ABBA lock inversion.** Draining the receive queue from `ibv_poll_cq()` meant two
   threads polling *different* CQs both entered `odl_rq_drain`, which posts completions
   into the *other* CQ — opposite lock orders. Receive progress now belongs to a
   **dedicated per-QP RX thread**, independent of who polls what. TX and RX are separate
   engines, as on real hardware.

2. **One stream served both directions.** The RCCL plugin — the only consumer known to
   work — opens a *separate* stream per direction and never shares one. Each QP now
   opens two: `rx_stream` (advertised as `qp_num`, so the peer's `IBV_QP_DEST_QPN` names
   where we receive) and `tx_stream`.

3. **Reserve sized as if it were the RX ring.** `TX_POOL_RESERVE=2048` against
   `rx_target=2048` of a 4096-frame pool left `free_count == reserve` exactly, and
   `can_send()` tests `free > reserve` — TX blocked forever on a healthy link. The
   driver diagnostic printed it verbatim: `can_send=0 state=4 free=2048 reserve=2048`.
   The reserve is a *floor* for RX repost, not the ring size; 256 leaves TX ~1700 frames.

Teardown was fixed alongside: `destroy_qp` now joins the RX thread (it kept draining
into a freed QP) and frees bounce buffers for untransmitted WRs.

**Method note.** Three of the changes made while chasing this were hypothesis-driven and
two made things *worse* — the oversized reserve deadlocked TX outright, and draining from
`poll_cq` introduced the ABBA. Reliable progress came from two things only: a test that
reproduces the real access pattern in seconds and verifies every byte, and a driver
diagnostic that prints the failing arithmetic. Build those first.

---

# BUG 24 — destroying a stream never wakes its receive waiters (FIXED)

**Severity: high for RCCL.** Found by two independent source reviews, then reproduced and
fixed. `--bidir` exercises steady-state traffic, not teardown, which is why it survived;
`tests/odl_lifecycle_test.c` now covers it. Before the fix a blocked receive never
returned; after it, it returns `-ESHUTDOWN` in 1.00 s.

The RCCL plugin's shutdown sets a stop flag, closes the stream, and joins the receive
worker — the close is *meant* to make the blocked receive fail so the worker observes the
flag. It does not. `odl_tb5_stream_destroy()` (`driver/odl_tb5_ring_dma.c:1429`) unhashes
the stream, unlinks it from its owner, frees the ID and drops the reference — and never
touches the wait queue:

```c
hash_del_rcu(&stream->node);
list_del(&stream->owner_list);
ida_free(&dev->stream_ida, stream->id);
kref_put(&stream->refcount, odl_tb5_stream_free);
/* no wake_up on stream->rx_waitq */
```

The **only** `wake_up` on `rx_waitq` is on data arrival (`:387`). Three waiters sleep on it
(`:2020`, `:2057`, `:2060`). So a worker parked in `STREAM_RECV` with no inbound data sleeps
until a signal arrives, and `pthread_join` blocks behind it — indefinitely.

Consequence: every RCCL error and abort path can wedge the process rather than tearing down.
That matters more for collectives than for ggml-rpc, because RCCL closes and reopens
communicators far more often.

**Fix:** wake the queue in `odl_tb5_stream_destroy()` (and on any close/abort path) with a
`dying` flag included in each `wait_event_interruptible` condition, so waiters re-check and
return `-ESHUTDOWN` instead of sleeping on a stream that no longer exists.

---

# BUG 25 — plugin `connect`/`accept` violate the RCCL v7 non-blocking contract (OPEN)

RCCL requires `connect()`/`accept()` to return `ncclSuccess` with `*sendComm`/`*recvComm`
set to `NULL` when the peer is not ready, so it can retry. The plugin returns an error
instead, which RCCL surfaces as a fatal `ncclSystemError` rather than retrying.

This matters exactly when the two ranks start at different times — which is the normal case
here, since rank 1 initialises lazily inside `ggml-rpc-server`. Related: `irecv` advertises a
single receive buffer but does not reject `n > 1`, so a multi-buffer receive would be
silently mishandled rather than refused.

Neither is yet reproduced on hardware; both are source findings.

---

# BUG 27 — duplex traffic loses two consecutive fragments (OPEN)

**Severity: blocks collectives.** This is why `-sm tensor` over RDMA cannot be attempted.

Roughly **1 in 4** bidirectional 512 MiB runs corrupts. Unidirectional has never failed.
Reproduces on the frozen `rdma-working-2026-07-27` driver, so it predates the teardown work.

The fragment sequencing added in BUG 22 detects it. Six gaps captured across both nodes:

```
lost 61,62   lost 56,57   lost 36,37   lost 23,24   lost 40,41   lost 50,51
```

**Always exactly two consecutive fragments** — never one, never three — at *varying*
positions. Independent of `tx_depth` (1, 2 and 4 all lose exactly two) and of chunk size
above the batch threshold.

Loss is detected but not repaired (no retransmit), so a dropped message desynchronises the
stream and the verifier reports corruption on a *later* message. That is why the offsets
look erratic — sometimes byte 0, sometimes mid-message. The first casualty is the message
that vanished, not the one that fails verification.

### Hypotheses already tested and killed

Recorded so nobody re-runs them:

| hypothesis | why it died |
|---|---|
| Chunk-size threshold | Intermittent at every size above `THROUGHPUT_THRESH`: 256 KiB failed 2/2 in one sitting, passed 5/6 in another |
| Send depth | `tx_depth` 1, 2 and 4 all lose exactly two; depth 1 still corrupts |
| Batch-buffer overflow | A 256 KiB message needs 66 frames and a batch holds 64 (payload is 4024/frame after the BUG 20 tail reserve, not 4096). The arithmetic matches "exactly two" — but it predicts loss at a **fixed boundary**, and the measured positions vary. Killed by position data |
| `odl_tb5_frame_pool_get_batch()` partial allocation | The function has **no callers**; dead code |

64 KiB chunks pass consistently, but `ODL_TB5_THROUGHPUT_THRESH = 65536` means they never
enter batch mode — a different code path, not evidence about size.

### Live candidate

See BUG 28. The RX callback never examines NHI error flags, so hardware-dropped or
error-flagged frames would be invisible — which fits pair-loss at arbitrary positions far
better than any fixed-boundary mechanism. **Untested.** The next step is to count the flags
rather than form a fifth hypothesis.

---

# BUG 28 — RX callback ignores NHI error flags (OPEN)

The receive path validates only `frame->size` before consuming a frame. It never inspects
hardware CRC, overrun or descriptor-validity flags, and keeps no counters for them.

Two consequences. A frame the hardware knows is bad is accepted as good. And when frames go
missing there is no evidence to inspect, because the one place that could report it is
silent — which is what makes BUG 27 hard to attribute.

The wire format also carries **no checksum** (`driver/uapi/odl_tb5_uapi.h`), so fragment
sequencing proves *continuity*, not *integrity*.

**Fix:** check the flags before copying; count CRC errors, overruns, short descriptors,
`payload_len` exceeding the descriptor, fragment gaps and assembly failures; expose them
read-only. Do not rate-limit away the only evidence.

---

# BUG 29 — the verbs TX worker busy-spins when idle (OPEN)

**Severity: it inverts the entire value proposition.** This is why RDMA measured *slower*
than TCP for cross-node tensor parallelism.

`verbs/src/odl_tb5_verbs_qp.c`, worker loop, idle branch:

```c
if (!have_wr) {
    /* No send work: make receive progress, then yield briefly. */
    odl_worker_poll_fd(qp, 2);      /* polls POLLOUT, 2 ms timeout */
    continue;
}
```

The intent is to yield. It never does. The device is writable essentially always, so
`poll(POLLOUT)` returns **immediately** rather than waiting out its timeout, and the loop
runs flat out for as long as the send queue is empty — one core, continuously, per QP.
Confirmed by strace: `poll([{fd=9, events=POLLOUT}], 1, 2) = 1` repeating back-to-back.

### Why it matters more than it looks

Cross-node tensor parallelism is bound by **host-side dispatch** (~4.13 ms per sync point
against a 100 µs collective — see [RESULTS.md](RESULTS.md)). CPU taken from that dispatch
costs more than a 2.9× faster collective returns:

| transport | tg128 |
|---|---|
| TCP | **3.50 ± 0.01** |
| RDMA | 3.30 ± 0.01 |

**A faster wire that costs CPU is a net loss when the bottleneck is the host.** Any
transport competing against sockets on a dispatch-bound workload must be judged on CPU
cost, not just latency and bandwidth.

### The fix is NOT a different poll mask — that was tried and it broke

Switching the idle wait to `POLLIN` looks obvious: it genuinely blocks, since inbound data
is the only thing that creates receive work. It is wrong. This worker must also wake when
the **caller posts a new send**, and waiting on RX means a freshly posted send is not
noticed until the 2 ms timeout expires or unrelated traffic arrives. Under the ggml-rpc
bulk-upload pattern that is enough to abort:

```
ggml_backend_rpc_buffer_set_tensor -> ggml_abort   (during load_all_data)
```

**Note the trap:** `odl_rdma_stress --bidir` **passed at 9.06 Gb/s with that broken
change in place.** The stress test's access pattern does not exercise the failing case, so
it is not sufficient validation here — a full tensor-parallel run is required, which costs
~10 minutes per attempt.

### The correct fix

A real wakeup channel, not a poll-mask change. Add a condition variable (or eventfd)
alongside `sq_lock`, signal it from `post_send` after enqueueing, and have the idle branch
`pthread_cond_timedwait()` on it. Then the worker sleeps until there is genuinely work,
wakes immediately on a new send, and keeps a bounded timeout as a safety net.

Not attempted here: the payoff is capped at ~6 % on a path that is structurally 2.6×
behind pipeline anyway, the validation loop is ~10 minutes, and the fast regression test
demonstrably does not catch breakage in this path. Worth doing deliberately rather than
opportunistically.

---

# Operational notes — Thunderbolt bond recovery

Reloading the Thunderbolt stack disturbs the **IP bond** that shares the same cables.

1. **Stale ARP after a `thunderbolt_net` reload.** New interfaces get new MACs; the
   peer's ARP entry goes stale and traffic blackholes in one direction only.
   `sudo ip neigh flush dev bond0`.
2. **A slave can come back "up" but carry no traffic.** `carrier=1`, `operstate=up`, yet
   one link receives nothing. With `balance-rr` the bond keeps striping onto the dead
   slave — ~50–66 % loss while ping *partially* works, easy to misread as "peer is down".
   Find it with per-interface `rx_packets` during a ping, then
   `sudo ip link set thunderbolt1 down`.
3. **Keep a non-Thunderbolt path to every node.** None of the above is debuggable
   otherwise.
4. **`odl_tb5` and `thunderbolt_net` coexist** (`prtcid=20300` vs `1`), so RDMA and IP
   share the cables — but a stack reload resets both.
5. **`odl_tb5` and `thunderbolt_ibverbs` do *not* coexist** — both drive the same NHI DMA.
6. **Never probe the RPC port with `/dev/tcp`.** `ggml-rpc-server` rejects the non-HELLO
   bytes and *exits*; if it is the container's PID 1 the container dies with it.

## Still cautionary

Each of these predates the BUG 9 regression and is **not** retracted, though each
deserves re-testing against the repaired build:

| ⚠️ | why |
|---|---|
| Reloading the **whole Thunderbolt stack** (unbinding the `thunderbolt` PCI driver, not just `odl_tb5`) | Wedged the USB4 controller and the GPU PSP; recovery needed a full power cycle. A different operation from `rmmod odl_tb5`, and never disproven |
| Unbounded login retries against a wedged router | Fixed by `login_max_retries` (default 20) |
| A boot logging `PSP firmware loading failed` | Power off fully; a warm reboot will not clear it |

## Also worth knowing

- **A corrupt `~/.cache/comgr` bricks HIP entirely.** One node segfaulted inside
  `libamdhip64` on *every* HIP program, including a 20-line `hipMalloc` test.
  `AMD_LOG_LEVEL=4` gave the real cause:
  ```
  ld.lld: error: undefined hidden symbol: __ockl_dm_init_v1
  rocdevice.cpp:730 : Couldn't create blit kernels!
  ```
  HIP JIT-links its blit kernels at init; a poisoned cache makes that link fail and the
  runtime dereferences NULL. Every on-disk file was byte-identical to the working node.
  Fix: `rm -rf ~/.cache/comgr` — try it **before** reinstalling ROCm. This cost a full
  ROCm reinstall to find.
- **AMD's apt repo needs pin priority 600** (`/etc/apt/preferences.d/repo-radeon-pin-600`),
  or Ubuntu's own `rocminfo`/`rocm-smi` win and block `rocm` with unsatisfiable conflicts.
- The driver logs under **two** prefixes: `odl_tb5:` (load/probe) and `OdinLink:`
  (handshake/DMA). Grepping only the first hides every interesting failure.
- `/dev/odl_tb5_N` indices are assigned by probe order and **differ between nodes** —
  read them, don't assume 0.
- Start the RPC server or CLI server over ssh with `exec` in a held session;
  `setsid … &` gets orphaned and dies silently.
