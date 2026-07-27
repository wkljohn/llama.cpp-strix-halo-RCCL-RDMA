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

---

# ⚠ HAZARD — repeated module reloads can wedge the USB4 controller *and* the GPU

**Severity: critical.** This is a hardware-integrity issue, not an inconvenience. Observed on
both nodes after several `rmmod odl_tb5 / thunderbolt_net / thunderbolt` + `modprobe` cycles
(the BUG 1 workaround). On the *following* boot:

```
amdgpu 0000:c5:00.0: psp reg (0x16080) wait timed out
amdgpu 0000:c5:00.0: PSP create ring failed! / PSP firmware loading failed
amdgpu 0000:c5:00.0: hw_init of IP block <psp> failed -22
amdgpu 0000:c5:00.0: Fatal error during GPU init
amdgpu 0000:c5:00.0: probe with driver amdgpu failed with error -22
thunderbolt 0000:c7:00.6: probe with driver thunderbolt failed with error -110
BUG: kernel NULL pointer dereference
```

and on the peer, the mechanism is explicit:

```
thunderbolt 0000:c8:00.6: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0043
                                                address=0xecce8000 flags=0x0020]
```

**An IOMMU page fault on the Thunderbolt controller** — i.e. the NHI DMA engine was still
armed and wrote into unmapped memory while the controller was being torn down. That wedges
the USB4 router, and (because they share the same power/firmware domain on Strix Halo) the
GPU's PSP fails to load its firmware on the next boot, so **amdgpu does not come up at all**.
Recovery required a full power cycle.

### Why this happens
`odl_tb5_remove()` cancels work items and calls `odl_tb5_rings_stop()`, but the ring
teardown is not ordered against in-flight NHI DMA, and (BUG 1) the XDomain paths are not
always disabled first. Unloading `thunderbolt` underneath a driver whose rings may still be
live is therefore unsafe.

### Practical rules until this is fixed upstream
1. **Do not loop `rmmod`/`modprobe` on this stack.** Treat each reload as risky.
2. **Always unbind the odl services and let the link go DISCONNECTED before unloading**, so
   paths are torn down while the Thunderbolt core is still present:
   ```bash
   for s in $(ls /sys/bus/thunderbolt/drivers/odl_tb5/ | grep -E '^[01]-'); do
       echo $s | sudo tee /sys/bus/thunderbolt/drivers/odl_tb5/unbind >/dev/null; done
   sleep 2 && sudo rmmod odl_tb5      # only then
   ```
3. **Prefer a clean reboot over a stack reload** when the link is already unhealthy — the
   reload is only safe from a *good* state, which is precisely when you do not need it.
4. If a boot shows `PSP firmware loading failed` / `amdgpu probe failed -22`, the machine
   needs a **full power cycle** (not a warm reboot) to reset the firmware domain.

### Correction to BUG 1's "no-reboot workaround"
The workaround does free leaked hop-IDs and *does* work from a healthy state, but repeating
it is what produced the wedge above. It should be used **once**, not as a routine recovery
step. The real fix is the BUG 1 patch (release paths on every teardown path) plus ordering
ring teardown against in-flight DMA.

### Escalation: `rmmod odl_tb5` is unsafe *even with* unbind-first ordering

A later attempt used the "safe" sequence (unbind all services, sleep, then `rmmod odl_tb5`
only, leaving the Thunderbolt core loaded). **It still crashed the machine.** Kernel log from
the crashed boot:

```
RIP: 0010:check_config_address+0x8d/0xb0 [thunderbolt]
RIP: 0010:tb_cfg_read+0xa5/0xf0 [thunderbolt]
RIP: 0010:drm_buddy_fini+0x119/0x120 [drm_buddy]
RIP: 0010:notifier_chain_register+0x45/0xe0
```

i.e. after the module went away the Thunderbolt core oopsed doing ordinary config-space
reads against a router the NHI had wedged, and `amdgpu`'s buddy allocator went down with it
(shared firmware/power domain on Strix Halo). Result: panic and reboot, repeatedly.

**Operational rule — do not unload this module.**
- Load `odl_tb5` **exactly once per boot**. Never `rmmod` it, with or without unbinding.
- To test a new driver build: **reboot first**, then `insmod` the new `.ko`. A reboot is
  cheap; a wedged USB4 router plus a dead GPU is not.
- Unbinding an individual *service* (`.../drivers/odl_tb5/unbind`) is fine and is still
  required for BUG 2 — it is module removal that is dangerous.

This supersedes the "safe unload ordering" suggested above and makes the BUG 1 patch more
important, not less: if hop-IDs never leak, you never need to reload in the first place.

### Also: BUG 1's first patch was incomplete — `complete_connection()` re-entry leaks

Tracking `in_hopid`/`in_hopid_valid` and releasing on every teardown path was **not
sufficient**. `odl_tb5_complete_connection()` calls `tb_xdomain_alloc_in_hopid()`
unconditionally at its head and then overwrites `dev->in_hopid`. Because the function is
re-entered on **every handshake restart** (peer restart, DMA-verify failure), each retry
orphans the previously allocated hop-ID. With a fresh boot and the first patch applied, a
node still reached `enable_paths failed (-12)` after a handful of restarts.

Additional fix — release before re-allocating:

```c
static int odl_tb5_complete_connection(struct odl_tb5_device *dev)
{
        if (dev->in_hopid_valid) {          /* re-entry: drop the old one first */
                if (dev->tx.started)
                        tb_xdomain_disable_paths(dev->xd, dev->local_tx_hopid,
                                                 dev->tx.ring ? dev->tx.ring->hop : -1,
                                                 dev->in_hopid,
                                                 dev->rx.ring ? dev->rx.ring->hop : -1);
                tb_xdomain_release_in_hopid(dev->xd, dev->in_hopid);
                dev->in_hopid_valid = false;
        }
        ret = tb_xdomain_alloc_in_hopid(dev->xd, dev->remote_tx_hopid);
        ...
}
```

---

# BUG 9 — IOMMU fault + USB4/GPU wedge — CAUSE NOT ISOLATED (retracted attribution)

**Severity: unresolved. Treat the driver as unsafe on AMD Strix Halo, but do _not_ read this
as a confirmed upstream defect — the earlier version of this section made that claim and it
was not supported by the evidence.**

### What was actually observed
```
thunderbolt 0000:c7:00.6: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x003f
                                                address=0xffbb8000 flags=0x0020]
amdgpu 0000:c5:00.0: probe with driver amdgpu failed with error -22
amdgpu_irq_put+0xc4/0xe0 [amdgpu]        (repeated)
drm_buddy_fini+0x112/0x120 [drm_buddy]
BUG: kernel NULL pointer dereference, address: 0000000000000000
```
The `IO_PAGE_FAULT` on the Thunderbolt controller comes first; the USB4 router then wedges,
and because the router shares a firmware/power domain with the iGPU on Strix Halo, `amdgpu`
fails to initialise — sometimes on the *following* boot too, needing a full power cycle.
That sequence is real and was seen repeatedly.

### Why the previous root cause was withdrawn
This section previously asserted the driver hands the NHI unmapped addresses and that the
ring buffers "must be mapped through the Thunderbolt device's DMA API". **That is wrong.**
Direct inspection of `odl_tb5_rings_alloc()` shows the buffers are already allocated with

```c
dma_alloc_coherent(tb_ring_dma_device(dev->tx.ring), ...)
```

i.e. the correct API against the correct device. The stated defect does not exist.

### The confound
The crashes used to justify this bug were produced by a **locally patched build, not stock
upstream.** While adding the BUG 1 hop-ID fix, a bad patch to `odl_tb5_remove()` silently
deleted its entire cleanup body — `odl_tb5_rings_stop()`, six `cancel_work_sync()` calls,
`synchronize_rcu()`, `list_del_rcu()`, and the buffer/ring frees — leaving:

```
  ... state bookkeeping ... -> disable_paths/release_in_hopid -> kfree(dev)
```

So `dev` and its coherent ring buffers were freed **while the NHI rings were still armed and
work items still held pointers to them.** DMA into freed-and-reallocated memory is an exact
mechanism for `IO_PAGE_FAULT` at a stale address, and for the wedge that follows. That build
was loaded during the crash runs quoted above, so those runs **cannot distinguish** an
upstream DMA defect from this local use-after-free.

`odl_tb5_remove()` has since been restored to the upstream teardown order, with the BUG 1 fix
re-applied *after* `rings_stop()`/`bufs_free()` rather than in place of them:

```
send_logout -> cancel_work_sync x6 -> rings_stop -> synchronize_rcu -> list_del_rcu
   -> dma_bufs_free -> rings_free -> disable_paths -> release_in_hopid -> kfree
```

### What is still true regardless
These were reproduced against **stock** code and are unaffected by the retraction:
- `rmmod` / service `unbind` leave rings armed and can wedge the router (BUG 1 teardown path).
- Repeated whole-TB-stack reloads wedge the USB4 controller and the GPU PSP.
- An unbounded login-retry loop hammers a wedged router (fixed by `login_max_retries`).

### Status
**Not retested yet.** The corrected driver has not been run long enough to say whether the
IOMMU fault recurs. Until it has:
- keep treating a load of `odl_tb5` on a machine you care about as risky;
- do not cite this section as evidence of an upstream bug;
- if you reproduce `IO_PAGE_FAULT` on a **clean upstream checkout** with no local patches,
  that would be a genuine finding worth reporting — this one was not.
