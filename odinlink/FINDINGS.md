# OdinLink-Five on AMD Strix Halo — measurements, bugs, and fixes

Tested 2026-07-25..27 on 2× Ryzen AI MAX+ 395 (gfx1151, USB4 Host Router `1022:158d/158e`),
kernel 7.0.0-28-generic, gcc 15.2.0, `github.com/Geramy/OdinLink-Five` @ `ed60505`.
Both nodes joined by 2× USB4 cables (normally bonded as `bond0`, balance-rr).

## TL;DR

RDMA-over-Thunderbolt **works on AMD USB4 hardware** (the project targets Intel TB5).
Measured **22.4 µs median** round-trip vs **286 µs/op** for the same collective over TCP —
a **~13× latency improvement** on the existing cables, no new NIC.
Three real bugs block a smooth bring-up; all have workarounds, two have concrete patches.

## Measurements (node1 ↔ node2, single USB4 link, `e2e=0`)

Latency (`odl_tb5_cli client -t latency -i 1000`):

| metric | value |
|---|---|
| min | 13.60 µs |
| **median (p50)** | **22.42 µs** |
| avg | 23.59 µs |
| p95 / p99 | 33.52 / 34.62 µs |
| p99.9 / max | 68.55 µs |
| jitter (stddev) | 4.01 µs |

Distribution: 96.2 % of samples in 20–50 µs, 3.7 % in 10–20 µs, 0.1 % > 50 µs.

Bandwidth (`-t bandwidth`): **8.38–9.22 Gb/s (1.05–1.15 GB/s)** on one link.
Driver reports the true link as `10 Gb/s (x2 lanes)`. So ~46 % of line rate — bandwidth is
*not* this transport's strength; **latency is**.

Reference points on the same hardware:
- RCCL all-reduce over TCP/bond0: **286 µs/op**
- TCP throughput over bonded 2× links (iperf3): ~19 Gb/s
- Thunderbolt ping (ICMP, IP stack): ~0.5–0.8 ms

## Why this matters

Cross-node tensor parallelism here is latency-bound. At 286 µs/op the per-layer all-reduce
dominates and pipeline (`-sm layer`) beats TP. At ~22 µs the comm term drops ~13×, which is
the regime where TP-RCCL can plausibly overtake pipeline. The RCCL net plugin
(`librccl_net_odl_tb5.so`) builds and its plugin API tests pass, so the path exists —
but see BUG 4 before benchmarking.

---

# BUG 1 — XDomain hop-ID / path leak → `ENOMEM` on every load after the first

**Severity: high.** This is the one that forces reboots.

### Symptom
The first `insmod` after a boot reaches `entering READY state`. Every subsequent load (or
service re-probe, e.g. after the peer reboots) fails:

```
OdinLink: enable_paths failed (-12), retry 1..5
OdinLink: failed to enable XDomain paths after 5 attempts: -12
OdinLink: connection completion failed (-12), retrying handshake
```

`-12` = `ENOMEM` from `tb_xdomain_enable_paths()` — the Thunderbolt core is out of
hop-ID/tunnel resources because previous instances never released theirs.

### Root cause A — `remove()` releases only in two states
`driver/odl_tb5_service.c:209`:

```c
if (saved_state == ODL_TB5_STATE_CONNECTED ||
    saved_state == ODL_TB5_STATE_READY) {
        tb_xdomain_disable_paths(...);
        tb_xdomain_release_in_hopid(dev->xd, dev->remote_tx_hopid);
}
```

`tb_xdomain_alloc_in_hopid()` is called in `odl_tb5_complete_connection()`
(`odl_tb5_proto.c:293`). If the device is removed/unbound while in **HANDSHAKE**,
**DISCONNECTED** or **ERROR** — which is exactly what happens when a handshake is retrying,
when the peer disappears, or when the user unbinds a service — the hop ID and paths are
**never released**.

### Root cause B — restart path releases a possibly-stale hop ID
`driver/odl_tb5_proto.c:601-608` releases `dev->stale_remote_tx_hopid`, but that field is
only assigned in the peer-login handler (lines 165, 191). A restart triggered by
**DMA-verify failure** (`DMA verify failed after 300 attempts`) does not pass through those
lines, so `stale_remote_tx_hopid` can differ from the currently allocated
`remote_tx_hopid` → the wrong ID is released and the live one leaks.

### Suggested fix
Track the allocation explicitly instead of inferring it from connection state:

```c
/* odl_tb5_core.h */
bool in_hopid_valid;
int  in_hopid;            /* the value actually passed to alloc_in_hopid() */

/* complete_connection(), after a successful alloc */
dev->in_hopid       = dev->remote_tx_hopid;
dev->in_hopid_valid = true;

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
```

Call it unconditionally in `odl_tb5_remove()` (drop the state check) and in the restart
path (replacing the `stale_remote_tx_hopid` logic). Releasing when nothing is allocated
becomes a no-op, so it is safe on all paths.

### Workaround — **no reboot required** (verified 2026-07-27)
Reloading the whole Thunderbolt stack tears down the domain and frees every hop ID:

```bash
sudo rmmod odl_tb5
sudo rmmod thunderbolt_net      # bond0 drops briefly
sudo rmmod thunderbolt          # frees all XDomain tunnels/hop IDs
sudo modprobe thunderbolt
sudo modprobe thunderbolt_net   # thunderboltN reappear, re-enslaved to bond0 automatically
sudo insmod odl_tb5.ko e2e=0
```

Verified: after this, `enable_paths` succeeded and both nodes reached READY without a
reboot. Peer rediscovery took ~5 s; `bond0` came back up at 20 Gb/s on its own.
Local GPU/inference workloads are untouched (they do not use Thunderbolt).

---

# BUG 2 — multi-domain route collision breaks the login handshake

**Severity: high** on any host with two or more Thunderbolt cables to the same peer.

### Symptom
With both USB4 cables connected, each side *receives* the peer's login and *sends* a
response with `ret=0`, but its own request always times out:

```
OdinLink: received login from peer (version=1, tx_path=9, size=48)
OdinLink: sent login response (ret=0, route=2, sn=0, tx_hopid=9)
OdinLink: login request failed: -110          <-- ETIMEDOUT, forever
```

The handshake never completes. Zero progress until one service is unbound.

### Root cause
`odl_tb5_find_device_by_route(u64 route)` (`odl_tb5_proto.c:78`) resolves the target device
by route **alone**. With two USB4 host routers (two domains), *both* peer services sit at
`route=2`:

```
0-2.1  prtcid=20300  route=2      <- domain 0
1-2.1  prtcid=20300  route=2      <- domain 1
```

so the lookup returns whichever matches first and the response is dispatched onto the wrong
xdomain — it never reaches the requester.

### Suggested fix
Disambiguate by domain as well as route, or match on the `tb_xdomain` pointer directly:

```c
static struct odl_tb5_device *
odl_tb5_find_device(const struct tb_xdomain *xd)          /* preferred */
{
        list_for_each_entry(dev, &odl_tb5_devices, list)
                if (dev->xd == xd)
                        return dev;
        return NULL;
}
```

The XDomain callback already has the originating `tb_xdomain`, so no route matching is
needed. If route matching must be kept, key on
`(tb_xdomain_parent(xd)->index, route)` rather than `route`.

### Workaround
Bind only one service per node:

```bash
echo 1-2.1 | sudo tee /sys/bus/thunderbolt/drivers/odl_tb5/unbind
```

(or physically unplug one of the two cables — cleaner, because it also avoids BUG 5).

---

# BUG 3 — `e2e=0` is honoured but logged as "enabled" (cosmetic)

`driver/odl_tb5_ring_dma.c:476` prints `(E2E enabled, e2e_tx_hop=%d)` unconditionally.
The parameter itself works correctly — `/sys/module/odl_tb5/parameters/e2e` reads `N` and
`RING_FLAG_E2E` is not set (`odl_tb5_ring_dma.c:448`) — but the log claims otherwise, which
costs debugging time when following the module's own `e2e=0` advice for TB3-class
controllers.

### Suggested fix
```c
"local_tx_hopid=%d (E2E %s, e2e_tx_hop=%d)\n", ..., odl_e2e ? "enabled" : "disabled", ...
```

---

# BUG 4 — RCCL/NCCL plugins report a hardcoded 80 Gb/s

`rccl/src/odl_tb5_plugin.c:184` and `nccl/src/odl_tb5_nccl_plugin.c:343`:

```c
props->speed = 80000;      /* TB5 spec number, not measured */
```

On this hardware the driver itself reports the true link as **10 Gb/s × 2 lanes**, and the
measured payload rate is **~9.2 Gb/s** — off by ~9×. NCCL/RCCL feed `props->speed` into
topology/cost estimation (ring vs tree, chunk sizing), so an inflated value can produce bad
algorithm and chunking decisions.

### Suggested fix
Report the real speed; the driver already exposes it through the `GET_PEER` ioctl
(`struct odl_tb5_peer_info.speed`, used by the CLI to print "Link speed: 10 Gb/s (x2
lanes)"). Query it in `getProperties()` and fall back to a conservative default only if the
ioctl returns 0.

---

# BUG 5 — unbinding a service triggers a peer logout → handshake churn (minor)

Unbinding one service (the BUG 2 workaround) sends a logout that resets the peer's state
machine (`received logout from peer` → `connection restarted, beginning handshake`), so both
sides re-run the handshake. Combined with BUG 1 this used to leak another hop ID each time.
Fixing BUG 2 removes the need to unbind at all; otherwise consider suppressing logout when
the removal is a local administrative unbind rather than a link loss.

---

# Working bring-up procedure (with current upstream code)

1. Ensure a clean Thunderbolt state — either fresh boot, or the BUG 1 stack-reload above.
2. `sudo insmod odl_tb5.ko e2e=0` on **both** nodes (peer must be present).
3. Unbind the second service on both nodes (BUG 2), or run with a single cable.
4. Wait for `OdinLink: entering READY state` in `dmesg` on **both** sides.
5. Server: `./build/cli/odl_tb5_cli server -d <N> -v`
   Client: `./build/cli/odl_tb5_cli client -d <N> -t latency|bandwidth`

Notes:
- The `/dev/odl_tb5_N` index is assigned by probe order and **differs between nodes** —
  read it from `ls /dev/odl_tb5_*` rather than assuming 0.
- The driver logs under **two** prefixes: `odl_tb5:` (load/probe) and `OdinLink:`
  (handshake/DMA). Grepping only the first hides every interesting failure.
- Start the server over ssh with `exec` in a held session; `setsid ... &` gets orphaned and
  dies silently (empty log).
- `odl_tb5` coexists with `thunderbolt_net`: it is a separate XDomain service
  (`prtcid=20300` vs `1`), so `bond0` and IP traffic keep working while RDMA is active.

---

# BUG 6 — RCCL net plugin exports the wrong symbol name (FIXED)

**Severity: high.** Makes the plugin invisible; RCCL silently falls back to TCP.

### Symptom
```
NCCL INFO NCCL_NET_PLUGIN set by environment to ODL_TB5
NCCL INFO External network plugin /path/librccl-net-ODL_TB5.so is unsupported
NCCL INFO NET/Socket : Using [0]bond0:10.4.0.1<0>      <-- silent TCP fallback
```

### Root cause
`rccl/src/odl_tb5_plugin.c:375` defines the entry point as **`rcclNetPlugin_v7`**.
RCCL — like NCCL — resolves net plugins by the symbol **`ncclNetPlugin_v<N>`**
(confirmed: `strings librccl.so.1` lists `ncclNetPlugin_v6..v11`, no `rccl*` variants).
The symbol never matches, so the plugin is rejected and RCCL falls back to sockets
**without an error** — users get zero RDMA benefit and no indication why.

### Fix (applied and verified)
```c
extern rcclNet_v7_t ncclNetPlugin_v7 __attribute__((alias("rcclNetPlugin_v7")));
```
After rebuilding, `nm -D` shows both `ncclNetPlugin_v7` and `rcclNetPlugin_v7`, and RCCL
loads the plugin instead of rejecting it.

---

# BUG 7 — the vendored net API does not match the real NCCL/RCCL ABI

**Severity: critical.** Once BUG 6 is fixed and the plugin actually loads, both ends
**segfault** (SIGSEGV/139) as soon as RCCL calls into it.

### Root cause
`third_party/rccl/net_v7.h` defines a struct that matches **no** real NCCL/RCCL version.
Compared against the genuine `ncclNet_v7_t` (NVIDIA nccl `plugins/net/example/nccl/net_v7.h`):

| slot | OdinLink `rcclNet_v7_t` | real `ncclNet_v7_t` |
|---|---|---|
| 7 | `closeListen` | **`regMr`** |
| 8 | `isend` | **`regMrDmaBuf`** |
| 9 | `irecv` | **`deregMr`** |
| — | *(absent)* | `getDeviceMr`, `irecvConsumed` |

Signatures also differ: real `isend` takes an `mhandle`; real `irecv`/`iflush` are
**multi-buffer** (`int n, void** data, int* sizes, int* tags, void** mhandles`), OdinLink's
are single-buffer. So RCCL calls `regMr` at the offset holding `closeListen`, with
mismatched arguments → immediate crash.

The NCCL-side plugin (`nccl/src/odl_tb5_nccl_plugin.c`) is closer to the real API but
declares only `ncclNetPlugin_v4`/`_v5`, which modern RCCL (v6+ only) will not load, and its
`connect`/`accept`/`irecv` signatures are likewise non-standard.

### Suggested fix
Rebuild both plugins against the **official** headers from
`github.com/NVIDIA/nccl` → `plugins/net/example/nccl/net_v*.h`, using
`plugins/net/example/plugin.c` as the reference skeleton (it also shows how to layer old
API versions on top of a newer implementation). Target v7 or newer.

---

# BUG 8 — the RCCL plugin's data path is a non-functional stub

**Severity: critical.** Even with BUG 6 + BUG 7 fixed, the plugin cannot move correct data.

1. **`isend`/`irecv` treat the data pointer as a file descriptor**
   (`odl_tb5_plugin.c:290,315`):
   ```c
   odl_tb5_send_dmabuf(comm->handle, (int)(intptr_t)data, 0, size);
   ```
   `odl_tb5_send_dmabuf()` expects a **dmabuf FD**, but NCCL passes a **memory pointer**.
   Casting the pointer to `int` yields a meaningless FD.
2. **`test()` never polls for completion** — zero calls to `odl_tb5_poll` /
   `odl_tb5_wait_tx` / `odl_tb5_wait_rx` anywhere in the plugin. It unconditionally sets
   `*done = 1`, so NCCL believes every transfer finished instantly.
3. `iflush` is a no-op with a non-standard signature (`(void*, int dev, void**)`).

### Consequence
The RCCL plugin is a skeleton, not an implementation. A working version needs real buffer
handling (`odl_tb5_tx_buffer`/`rx_buffer` + copy, or genuine dmabuf registration through
`regMr`), real completion tracking via `odl_tb5_poll`, tag matching, and multi-receive
support.

### Status of the verbs alternative (also closed)
OdinLink's docs suggest `NCCL_NET_PLUGIN=IB` + `NCCL_IB_HCA=odl_tb5` instead. That does not
work here either:
- the driver is a **Thunderbolt service driver**, not a kernel RDMA driver, so nothing
  appears under `/sys/class/infiniband` and `ibv_devinfo` reports "No IB devices found";
- substituting `libodl_tb5_verbs.so` for `libibverbs.so.1` fails because RCCL requires
  **versioned** symbols (`IBVERBS_1.1/1.8/1.10/1.12`) that the OdinLink library does not
  export.

**Net effect:** RDMA-over-Thunderbolt is real and fast at the driver level (22.4 µs), but
there is currently **no working path from RCCL to it** — neither plugin nor verbs.

---

# UPDATE 2026-07-27 — upstream PR #20 supersedes BUG 6/7/8

After these findings were written, **[OdinLink PR #20](https://github.com/Geramy/OdinLink-Five/pull/20)**
("Strix Halo (gfx1151) over Thunderbolt: fix RX-queue overflow hang + ncclNet_v7 ABI") was found
open against the same repo, on the **same hardware class**. It fixes BUG 6/7/8 and one further
defect not identified here, and reports TP=2 Llama-3.1-8B serving over OdinLink **beating
TCP-over-`thunderbolt0` by ~17–26 % on decode**.

Its independent measurements agree with ours: ~9 Gbit/s bulk bandwidth, attributed to the
**USB4v1 cable** (the routers are `gen=4`/v2-capable, so a TB5 cable should scale it).

### What PR #20 changes
1. **driver — RX-queue overflow (the real hang blocker).** Per-stream `rx_queue_max` was 256,
   but one ~1 MB message chunks into ~264 frames. Under sustained duplex load the queue fills
   and `odl_tb5_rx_callback()` **silently drops** frames — there is no backpressure — so any
   consumer assuming lossless in-order delivery desyncs permanently and RCCL's recv blocks
   forever. Raised to 65536, plus RCU-safe stream lookup
   (`hash_*_rcu` + `kref_get_unless_zero`).
   *This was not diagnosed in our own investigation and is the reason collectives hang rather
   than merely crash.*
2. **rccl — `ncclNet_v7` ABI + host-staged plugin rewrite.** Same defects as BUG 7/8 here,
   plus one we missed: properties used `char[128]` where the real ABI has `char*`, so RCCL
   dereferenced a string as a pointer. Rewritten host-staged (`NCCL_PTR_HOST`) with a
   per-connection FIFO worker, single-frame chunking and a 4-byte length header.
3. **`odl_stress.c`** — an RCCL-free reproducer for the desync.

### Applying it
```bash
gh pr diff 20 --repo Geramy/OdinLink-Five > pr20.diff   # or curl .../pull/20.diff
git apply pr20.diff
cd build && make                 # rebuilds librccl_net_odl_tb5.so
cd ../driver && make             # rebuilds odl_tb5.ko with the queue fix
```
Verify: `nm -D --defined-only build/rccl/librccl_net_odl_tb5.so | grep NetPlugin`
should list **`ncclNetPlugin_v7`** (the name RCCL resolves).

---

# OPERATIONAL NOTES — Thunderbolt bond recovery (learned the hard way)

Reloading the Thunderbolt stack (the BUG 1 no-reboot workaround) disturbs the **IP bond** that
shares the same cables. Symptoms and fixes:

1. **Stale ARP after `thunderbolt_net` reload.** New interfaces get new MACs; the peer's ARP
   entry goes stale/`INCOMPLETE` and traffic blackholes in one direction only.
   ```bash
   sudo ip neigh flush dev bond0
   ```
2. **A slave can come back "up" but carry no traffic.** `carrier=1` and `operstate=up` on both,
   yet one link receives nothing. With `balance-rr` the bond keeps striping onto the dead slave,
   producing ~50–66 % packet loss (SSH dies, RPC connects fail) while ping *partially* works —
   an easy symptom to misread as "peer is down".
   Identify the dead slave by per-interface counters during a ping:
   ```bash
   for i in thunderbolt0 thunderbolt1; do
     echo "$i $(cat /sys/class/net/$i/statistics/rx_packets)"; done
   ```
   Then drop it: `sudo ip link set thunderbolt1 down` → bond runs single-link, 0 % loss.
3. **Keep a non-Thunderbolt path to the peer.** All of the above is only debuggable if you can
   still reach the other node — use the LAN/Wi-Fi address, not the bond, for recovery.
4. **`odl_tb5` and `thunderbolt_net` coexist** (different XDomain services, `prtcid=20300` vs
   `1`), so RDMA and IP run over the same cables simultaneously — but a stack reload resets
   *both*, so expect to re-check the bond every time.

---

# OPEN — PR #20 plugin loads and is selected, but `ncclCommInitRank` fails

Status 2026-07-27 with PR #20 applied, both nodes READY, single USB4 link, one channel.

**The plugin is genuinely in use** — this is the first configuration where RCCL does not fall
back to sockets:

```
NCCL INFO Successfully loaded external network plugin .../librccl-net-ODL_TB5.so
NCCL INFO Initialized NET plugin ODL_TB5
NCCL INFO Assigned NET plugin ODL_TB5 to comm
NCCL INFO Using network ODL_TB5
NCCL INFO Channel 00/01 : 0 1
```

**But communicator creation then fails** on both ranks:

```
[Proxy Service] .../transport/net_tmp.cc:1006 -> 2      # 2 = ncclSystemError
                .../transport/net_tmp.cc:540  -> 2
                .../transport.cc:47 -> 2 ; transport.cc:196 -> 2
                .../transport/generic.cc:25 -> 2
                .../init.cc:2050 -> 2 ; 2413 -> 2 ; 2970 -> 2 ; 3001 -> 2
ggml_cuda_world_init_once: ncclCommInitRank failed: unhandled system error
```

i.e. the failure is in the **net-transport proxy connection setup**, not in plugin discovery,
symbol resolution, or the ABI.

### Ruled out
- **Device exclusivity** — `/dev/odl_tb5_0` opens twice concurrently without error, and the
  PR #20 plugin already refcounts a single global handle (`g_handle` / `g_handle_refs`).
- **Channel count** — same failure with `NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1`
  (`Channel 00/01`), so it is not TX/RX buffer contention across channels.
- **Link state** — both nodes reached `entering READY state` immediately before the run, and
  `odl_tb5_cli` latency/bandwidth tests pass on the same link.
- **ABI/symbol** — `ncclNetPlugin_v7` resolves; RCCL logs "Using network ODL_TB5".

### Suspected difference from the PR's tested setup
PR #20 was validated with **vLLM TP=2 (Ray)**. This workload is
[llama.cpp-strix-halo-RCCL-RDMA](https://github.com/wkljohn/llama.cpp-strix-halo-RCCL-RDMA),
which builds the communicator with **`ncclCommInitRank` from two independent processes on two
machines** (rank 0 = `llama-bench`, rank 1 = `ggml-rpc-server`), rendezvousing via a TCP
`ncclUniqueId` exchange. That drives a different `listen`/`connect`/`accept` ordering through
the plugin than a Ray-launched SPMD job, and the two sides initialise at different times
(rank 1 lazily, on its first `GGML_OP_ALLREDUCE`).

Worth checking in the plugin: whether `connect()`/`accept()` tolerate being called before the
peer's corresponding call (NCCL requires returning `ncclSuccess` with `*sendComm == NULL` /
`*recvComm == NULL` so it can be retried, rather than returning an error), and whether the
5 s `odl_tb5_wait_peer()` inside `connect()` can trip when the peer process has not yet
reached its own init.

### CORRECTION — the `ncclCommInitRank` failure is BUG 1, not a plugin defect

The "suspected difference from the PR's tested setup" above (connect/accept retry semantics)
is **probably wrong**. Checking the link state immediately after a failing run showed it had
already left READY *before* the benchmark started:

```
node2:  enable_paths failed (-12) -> failed to enable XDomain paths after 5 attempts
node1:  DMA ping attempt 250, still waiting for pong / peer restarted
```

That is **BUG 1 (the XDomain hop-ID leak)**. The plugin opens and closes the device on every
comm setup; each cycle restarts the link, and after enough cycles the peer exhausts hop-IDs.
With the link not READY, `get_shared_handle()`'s `odl_tb5_wait_peer(handle, 10000)` times out
and returns `rcclSystemError`, so `listen`/`connect` fail and RCCL's net proxy surfaces
`ncclSystemError` from `net_tmp.cc` — which is exactly the chain observed.

**Implication:** the hop-ID leak is not only an `insmod`-time problem. *Any* workload that
opens/closes the device repeatedly will eventually exhaust hop-IDs. Until BUG 1 is fixed, a
long-running RCCL job on this transport is not reliable, and the TB-stack-reload workaround
must be run immediately before each attempt.

**Second-order problem:** the stack-reload workaround itself disturbs `thunderbolt_net`. After
several reload cycles the IP bond between the nodes stopped passing traffic in both directions
(interfaces `UP`, slaves `UP`, zero `rx_packets`) while XDomain control messages still flowed —
i.e. the Thunderbolt fabric was fine but the network driver had desynced. Recovering that
needed a reboot. Fixing BUG 1 properly (patch above) removes the need for the reload entirely
and is the highest-value change for anyone trying to use this transport in anger.
