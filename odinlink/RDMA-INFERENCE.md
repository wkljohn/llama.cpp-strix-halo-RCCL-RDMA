# RDMA-over-Thunderbolt for llama.cpp inference — working recipe

**Status: WORKING.** Two Strix Halo nodes running llama.cpp cross-node inference with the
RPC transport carrying tensor traffic over **RDMA on the Thunderbolt/USB4 cable**, no NIC.

```
RDMA probed:    dev=usb4_rdma0 gid=1 RoCEv2 qpn=2304 inline=0
RDMA activated: qpn=2304->2304 mtu=4096 rx_depth=24
```

Measured fresh on 2× Ryzen AI MAX+ 395 (gfx1151), kernel 7.0.0-28, Ubuntu 26.04, ROCm 7.2.0,
Huihui-Qwen3.6-27B-abliterated Q6_K (20.88 GiB), `llama-bench -p 512 -n 128 -r 2`.

| config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 2 nodes, `-sm layer`, **RDMA** | 290.23 ± 61.80 | **9.07 ± 0.01** |
| 2 nodes, `-sm layer`, TCP | 292.11 ± 62.37 | 8.83 ± 0.03 |

RDMA is **+2.7 % on decode** and within noise on prompt processing. That is a real but modest
win, and it is the honest number — see "Why the win is small" below.

---

## Which RDMA stack

Two candidates exist on this hardware. **Only one currently works with llama.cpp.**

| stack | kernel `ib_device`? | rdma-core ABI | works with llama.cpp RPC? |
|---|---|---|---|
| [`thunderbolt-ibverbs`](https://github.com/) 0.3.4 (DKMS) | ✅ yes (`usb4_rdma0`) | v59 (matches rdma-core 61) | ✅ **yes** |
| [OdinLink-Five](https://github.com/Geramy/OdinLink-Five) | ❌ no — char dev only (`/dev/odl_tb5_0`) | v34 (stale) | ⚠️ **partly — discovery + QP activation now work, data path still stalls** (BUG 11/13/14) |

llama.cpp's RDMA transport (`ggml/src/ggml-rpc/transport.cpp`, `GGML_RPC_RDMA=ON`) discovers
devices the standard way — `ibv_get_device_list()` → `ibv_get_device_name()` →
`ibv_query_port()` → match a GID against the local TCP address. That requires a **real kernel
RDMA device**. OdinLink does not register one, so it cannot drive this path today.

OdinLink is still the better *latency* transport in isolation (22.9 µs vs 286 µs TCP,
measured — see [FINDINGS.md](FINDINGS.md)); what it lacks is the verbs plumbing. BUG 11 below
documents exactly what it needs.

---

## Recipe (both nodes)

### 1. Install the DKMS module

```bash
sudo apt install thunderbolt-ibverbs-dkms   # needs Linux 6.14+
dkms status | grep thunderbolt              # thunderbolt-ibverbs/0.3.4 ... installed
```

If `odl_tb5` is loaded, unload it first — both drive the same Thunderbolt NHI DMA and they
conflict:

```bash
sudo rmmod odl_tb5
```

### 2. Set module options in `modprobe.d`, NOT on the command line

**This is the step that wastes the most time.** udev auto-loads `thunderbolt_ibverbs` when the
Thunderbolt service appears, *before* any manual `modprobe`, so a later
`modprobe thunderbolt_ibverbs profile=...` is a silent no-op — the module is already loaded
with defaults (`profile=auto`, `bind_services=N`, `register_verbs=N`) and registers nothing.

```bash
sudo tee /etc/modprobe.d/thunderbolt-ibverbs.conf <<'EOF'
options thunderbolt_ibverbs profile=linux_perf bind_services=1 allocate_rings=1 \
        start_rings=1 negotiate_native=1 enable_tunnels=1 register_verbs=1 roce_netdev=bond0
EOF
```

`roce_netdev` must be the interface whose IP the RPC server binds to (here `bond0` =
10.4.0.x over the Thunderbolt cables). The transport matches a GID against that address; get
it wrong and RDMA silently falls back to TCP.

### 3. Disable rdma-core persistent naming

**The other big time sink.** The module registers its device as `usb4_rdma0`, but rdma-core's
udev rule renames RDMA devices by PCI path — `usb4_rdma0` becomes `rocep199s0f5`. The
userspace provider matches on modalias `rdma_device:*Nusb4_rdma*`, i.e. **on the name**, so
after the rename nothing matches and you get:

```
libibverbs: Warning: no userspace device-specific driver found for rocep199s0f5
```

with `ibv_devices` empty even though `rdma link` shows the device ACTIVE. Mask the rule:

```bash
sudo ln -sf /dev/null /etc/udev/rules.d/60-rdma-persistent-naming.rules
sudo udevadm control --reload
```

### 4. Load and verify

```bash
sudo modprobe -r thunderbolt_ibverbs; sudo modprobe thunderbolt_ibverbs
ibv_devices        # must list usb4_rdma0 -- if empty, revisit steps 2 and 3
rdma link          # link usb4_rdma0/1 state ACTIVE physical_state LINK_UP netdev bond0
```

The peer handshake is **racy**: `native HELLO ... failed after 5 attempts: -110` happens often
when both nodes load simultaneously. Load the peer first, wait ~5 s, then the head; retry if
no `registered native ib_device` line appears in `dmesg`.

### 5. Run inference

```bash
# peer
export GGML_RDMA_DEV=usb4_rdma0
./ggml-rpc-server -H 10.4.0.2 -p 50052

# head
export GGML_RDMA_DEV=usb4_rdma0
./llama-bench -m model.gguf --rpc 10.4.0.2:50052 -sm layer -ngl 99 -p 512 -n 128 -r 2
```

Confirm from the head's output that RDMA actually engaged:

```
RDMA probed: dev=usb4_rdma0 gid=1 RoCEv2 ...     <- good
RDMA activated: qpn=...->... mtu=4096 ...         <- carrying traffic
```

No `RDMA activated` line means it fell back to TCP and your "RDMA" numbers are TCP numbers.

---

## Why the win is small (+2.7 %)

`-sm layer` (pipeline) crosses the wire **once per token**, so the per-transfer latency the
RDMA path improves is amortised away. RDMA's advantage is latency (22–23 µs vs 286 µs), not
bandwidth — and the Thunderbolt link is bandwidth-limited (~9–19 Gb/s), which is what pp512
is bound by. Hence prompt processing is unchanged and decode gains a few percent.

The configuration that *should* benefit far more is `-sm tensor`, which does an all-reduce
**every layer** — the documented 27B numbers are 3.10 t/s (butterfly) and 3.65 t/s (RCCL) over
TCP, both bottlenecked by the 286 µs/op collective. That is the case RDMA is built for, and it
is not yet reproduced here (the RCCL world-communicator rendezvous deadlocks — see BUG 12).

Note also that **single-node still beats both**: the 27B fits in one 96 GB carve, so
cross-node only makes sense for models that do not fit.

---

## BUG 11 — OdinLink's verbs provider can never load on modern rdma-core

**Severity: blocker for inference use.** Two independent defects:

1. **Stale ABI symbol.** `verbs/src/odl_tb5_provider_plugin.c` resolves
   `verbs_register_driver_34`:
   ```c
   verbs_register_driver = dlsym(h, "verbs_register_driver_34");
   ```
   rdma-core 61 (Ubuntu 26.04) exports only `verbs_register_driver_59@@IBVERBS_PRIVATE_59`.
   The `dlsym` returns NULL, the constructor prints its warning to stderr where nobody sees
   it, and the provider silently never registers. The build also emits
   `libodl_tb5-rdmav34.so` while every system provider is `-rdmav59`, so rdma-core's directory
   scan skips it regardless.

   *Fix:* resolve the symbol for the installed ABI (try `_59`, fall back to `_34`), and derive
   the filename suffix from the rdma-core found at configure time instead of hardcoding 34.

2. **No kernel RDMA device.** Even once registered, rdma-core enumerates devices from
   `/sys/class/infiniband/*` + `/dev/infiniband/uverbsN`. OdinLink's kernel module creates only
   a char device (`/dev/odl_tb5_0`) and registers nothing with `ib_core`, so there is nothing
   to enumerate — `ibv_devices` stays empty and `match_device`/`alloc_device` are never called.

   *Fix:* register an `ib_device` with `ib_register_device()` (as `thunderbolt_ibverbs` does),
   or document the LD_PRELOAD shim as the only supported path — but note the shim exports
   `ibv_open_device`/`post_send`/`poll_cq` and **not** `ibv_get_device_list`/
   `ibv_get_device_name`, so applications that enumerate devices (llama.cpp, NCCL/RCCL, and
   every `perftest` tool) cannot find it either.

Until one of these is addressed, OdinLink cannot carry llama.cpp (or RCCL) traffic no matter
how good its raw transport is.

## BUG 12 — RCCL world-communicator rendezvous deadlock (llama.cpp port)

`ggml_cuda_world_init_once()` blocks in `accept()` on rank 0 during the local backend's
allreduce, while the RPC peer only calls the same function when it *executes* an allreduce
node. With `test-world-allreduce` the head reaches `world run: layers=60 iters=30`, prints
`rank 0 waiting for 1 peer(s) on port 29500`, and hangs; the peer accepts the RPC connection,
never executes the allreduce, never connects back. Observed: head has 0 established
connections to the peer while blocked, peer GPU at 0 %. The port's own `REPRODUCE.md` flags
this as an intermittent startup hang; here it reproduces every time.

## Operational gotchas that cost real time

- **A corrupt `~/.cache/comgr` bricks HIP entirely.** One node segfaulted inside
  `libamdhip64` on *every* HIP program — including a 20-line `hipMalloc`+`hipStreamSynchronize`
  test, and even `llama-bench -ngl 0`. `AMD_LOG_LEVEL=4` revealed the real cause:
  ```
  ld.lld: error: undefined hidden symbol: __ockl_dm_init_v1
  Error: Creating the executable from LLVM IRs failed.
  rocdevice.cpp:730 : Couldn't create blit kernels!
  ```
  HIP JIT-links its internal blit kernels at init; a poisoned comgr cache makes that link fail
  and the runtime dereferences NULL. Every on-disk file (bitcode, comgr, HIP runtime) was
  byte-identical to the working node. Fix: `rm -rf ~/.cache/comgr`. Worth trying **before**
  reinstalling ROCm — this cost an entire ROCm reinstall to find.
- **AMD's apt repo needs pin priority 600** (`/etc/apt/preferences.d/repo-radeon-pin-600`),
  otherwise Ubuntu's own `rocminfo`/`rocm-smi` packages win and block `rocm` with unsatisfiable
  dependency conflicts.
- The RPC server must be started with `exec` over ssh in a held session; backgrounded/orphaned
  it dies silently.


---

# Making OdinLink itself carry the traffic — progress and remaining gap

`patches/odinlink-verbs-and-driver-fixes.patch` (apply to a clean
[OdinLink-Five](https://github.com/Geramy/OdinLink-Five) checkout) takes OdinLink from
"completely invisible to every RDMA application" to "discovered, opened, and QP activated by
llama.cpp on both nodes":

```
RDMA probed:    dev=odl_tb5_0 gid=0 RoCEv2 qpn=20 inline=256
RDMA activated: qpn=20->20 mtu=4096 rx_depth=24        <- on BOTH nodes
```

**It does not yet complete a benchmark** — after activation the first payload exchange stalls
and the run times out. What follows is what was fixed and what is left, so the remaining work
is a short list rather than a rediscovery exercise.

## BUG 11 — no discovery path (FIXED here)

Covered above: the rdma-core provider resolves `verbs_register_driver_34` against an
rdma-core that only exports `_59`, and no kernel `ib_device` exists to enumerate anyway.

Rather than add a kernel uverbs device, the fix supplies the discovery half of the API in the
existing LD_PRELOAD shim — new file `verbs/src/odl_tb5_verbs_discovery.c`:

- `ibv_get_device_list` / `ibv_free_device_list` / `ibv_get_device_name` — advertise
  `odl_tb5_N` alongside (not instead of) any real adapters, which are forwarded to the real
  libibverbs so a Mellanox card in the same box keeps working.
- `ibv_query_gid` **and** `_ibv_query_gid_ex` — synthesise a RoCE v2 GID as the IPv4-mapped
  form of the local address (`::ffff:a.b.c.d`, from `ODL_RDMA_GID_IFACE`, default `bond0`),
  which is what consumers match against their TCP address. Both forms must be interposed:
  `ibv_query_gid_ex` is `static inline` and forwards to the exported `_ibv_query_gid_ex`,
  while the plain `ibv_query_gid` is called separately by `rdma_probe()` — and the real
  libibverbs implementation **segfaults** on an OdinLink context, since it walks
  provider-private state that does not exist.

The data-path verbs (`ibv_post_send`, `ibv_poll_cq`) are `static inline` and dispatch through
`context->ops`, which `odl_init_context_ops()` already fills — so those need no interposer.

`ibv_devices` now lists `odl_tb5_0` on both nodes.

## BUG 13 — `ibv_post_recv` had inverted semantics (FIXED here)

`odl_post_recv()` did not post a buffer; it **performed a receive inline**, blocking in
`poll(fd, 5000)` and returning `-ETIMEDOUT`/`-EAGAIN` when no data had arrived:

```c
int pr = poll(&pfd, 1, 5000);
if (pr <= 0 || !(pfd.revents & POLLIN)) { *bad_wr = wr; return pr == 0 ? -ETIMEDOUT : -EAGAIN; }
```

Every verbs consumer **pre-posts** receive buffers before any traffic exists — llama.cpp posts
`RDMA_RX_DEPTH` (24) of them between the INIT and RTR transitions, and RCCL/perftest do the
same. So the first post always failed and setup aborted with `RDMA activate failed, staying on
TCP`. This is a design-level incompatibility, not a tuning issue.

Fix: a real receive queue. `post_recv` copies the WR into a ring and returns immediately; the
existing QP worker thread drains inbound stream data into posted buffers and posts the
completions (`odl_rq_drain()`).

## BUG 14 — send path stored pointers to caller stack memory (FIXED here)

`odl_post_send()` stored the caller's `struct ibv_send_wr *` in `qp->sq[]` and the worker
thread dereferenced it later. Callers legitimately use stack storage —

```c
struct ibv_sge sge = {};
struct ibv_send_wr wr = {}, * bad = nullptr;
...
if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;   // wr dies at scope exit
```

— because `ibv_post_send` is defined to consume the WR before returning. The worker was
therefore reading freed stack frames. Fix: copy `wr_id`/`addr`/`length`/`lkey`/`num_sge` at
post time. The same commit sets `wc.wr_id` on send completions, which was never populated —
consumers that match completions by `wr_id` (llama.cpp uses the chunk sequence number) could
not have worked.

## What is still broken

After `RDMA activated` on both nodes the first payload exchange does not complete: the head
blocks and the run times out, while the driver shows the stream created and torn down
(`odl_tb5: stream 20 created` / `destroying`). Both peers independently chose stream id 20, so
that coincidence has not been ruled out as load-bearing. Suspects, in order:

1. ~~**Stream addressing.**~~ **Tried, did not fix.** `odl_modify_qp()` ignored
   `IBV_QP_DEST_QPN` entirely, so the worker sent everything to `dst_id = 0`. The patch now
   captures `attr->dest_qp_num` at the RTR transition and addresses sends with it
   (`odl_tb5_stream_send(h, stream_id, dest_qp, ...)`). Correct and worth keeping, but the
   stall persists — so `dst_id` is evidently not simply "the remote stream id", and the real
   local↔remote pairing semantics of `odl_tb5_stream_open(filter_id)` need to be established
   from the driver side. **This is the open question.**
2. ~~**RX drain scheduling.**~~ **Tried, did not fix.** `odl_rq_drain()` now runs on every
   worker iteration rather than only when idle, so send traffic cannot starve receives.
   Correct, but not the blocker.
3. **Flow control / chunking.** llama.cpp sends 256 KiB chunks with `RDMA_TX_DEPTH=2` in
   flight; OdinLink's frame pool and E2E credits may need explicit backpressure rather than
   `-EAGAIN` re-queueing. **Not yet investigated.**
4. **Stream pairing across nodes.** Both peers independently auto-assigned stream id 20. If
   `odl_tb5_stream_open()` filters are node-local, two peers can hold "stream 20" that are not
   connected to each other at all, and no amount of userspace addressing will join them —
   the CLI's own client/server test does its own pairing handshake that this path skips.
   **Most likely culprit now; start here.**
