> ## 📦 Ready-to-use fork: [wkljohn/OdinLink-Five @ strix-halo-verbs-fixes](https://github.com/wkljohn/OdinLink-Five/tree/strix-halo-verbs-fixes)
>
> All fixes below are pushed to a fork so you can clone and build directly —
> no patch application needed:
>
> ```bash
> git clone -b strix-halo-verbs-fixes https://github.com/wkljohn/OdinLink-Five.git
> cd OdinLink-Five
> cmake -B build -DBUILD_VERBS=ON -DBUILD_TRAY=OFF && cmake --build build -j$(nproc)
> make -C driver
> ```
>
> Upstream is [Geramy/OdinLink-Five](https://github.com/Geramy/OdinLink-Five)
> (driver GPL-2.0, verbs provider MIT). The branch sits on upstream `ed60505` —
> see the [full diff](https://github.com/Geramy/OdinLink-Five/compare/ed60505...wkljohn:strix-halo-verbs-fixes). `patches/odinlink-verbs-and-driver-fixes.patch` is kept in
> sync for anyone who prefers patching a pristine checkout.
>
> Measured results: [RESULTS.md](RESULTS.md).

> ## ✅ RDMA INFERENCE IS WORKING — see [RDMA-INFERENCE.md](RDMA-INFERENCE.md)
>
> Two Strix Halo nodes now run llama.cpp cross-node inference with tensor traffic over
> **RDMA on the Thunderbolt/USB4 cable**, no NIC:
>
> ```
> RDMA activated: qpn=2304->2304 mtu=4096 rx_depth=24
> ```
>
> | 27B Q6_K, 2 nodes, `-sm layer` | pp512 | tg128 |
> |---|---|---|
> | **RDMA** | 290.23 ± 61.80 | **9.07 ± 0.01** |
> | TCP | 292.11 ± 62.37 | 8.83 ± 0.03 |
>
> **Important:** that works via the `thunderbolt-ibverbs` DKMS module, **not** OdinLink.
> OdinLink has the faster raw transport (22.9 µs vs 286 µs) but registers no kernel RDMA
> device and resolves a stale rdma-core ABI symbol, so llama.cpp/RCCL cannot discover it.
> See BUG 11 in [RDMA-INFERENCE.md](RDMA-INFERENCE.md) for exactly what it needs.

# RDMA over Thunderbolt/USB4 on Strix Halo — OdinLink bring-up

> ## ✅ STATUS: the crash was a local regression, not the upstream driver — now fixed
>
> An earlier revision of this page told you never to load this driver. **That was wrong, and the
> cause was our own patch.** Recorded here in full because the retraction matters more than the
> original claim.
>
> While adding the BUG 1 hop-ID fix, a bad patch to `odl_tb5_remove()` silently deleted the
> function's entire cleanup body — `odl_tb5_rings_stop()`, six `cancel_work_sync()` calls,
> `synchronize_rcu()`, `list_del_rcu()`, and every buffer/ring free. The driver then did:
>
> ```
> ... state bookkeeping ...  ->  release hop-ID  ->  kfree(dev)
> ```
>
> freeing the device and its coherent DMA rings **while the NHI was still transferring into
> them and six work items still held pointers.** That is a textbook use-after-free, and it
> explains the whole cascade we had blamed on AMD: `AMD-Vi IO_PAGE_FAULT` at a stale address →
> USB4 router wedge → (shared firmware domain) `amdgpu` probe -22 → `drm_buddy_fini` NULL
> deref → hard freeze.
>
> **After restoring the upstream teardown order**, on the same 2× Ryzen AI MAX+ 395 hardware:
>
> | test | result |
> |---|---|
> | `rmmod` (was: "❌ never do this") | clean, <1 s, **3/3 cycles** |
> | `insmod` → `rmmod` → `insmod` reload | no `enable_paths -12/ENOMEM`, hop-ID correctly released |
> | `IO_PAGE_FAULT` count | **0** |
> | kernel oops / `BUG:` | **0** |
> | GPU (`/dev/kfd`, `rocm-smi`) | healthy throughout, no reboot needed |
>
> The correct teardown order — apply any hop-ID fix **after** the rings are stopped, never in
> place of them:
>
> ```
> send_logout -> cancel_work_sync x6 -> rings_stop -> synchronize_rcu -> list_del_rcu
>    -> dma_bufs_free -> rings_free -> disable_paths -> release_in_hopid -> kfree
> ```
>
> ### What remains genuinely cautionary
>
> These were seen **before** the regression existed and are not retracted, though they are now
> the *only* open hazards and each deserves re-testing against the repaired build:
>
> | ⚠️ Still be careful | Why |
> |---|---|
> | Reloading the **whole Thunderbolt stack** (unbind `thunderbolt` PCI driver, not just `odl_tb5`) | Wedged the USB4 controller and the GPU PSP; recovery needed a full power cycle. This is a different operation from `rmmod odl_tb5` and was never disproven |
> | Unbounded login retries against a wedged router | Fixed by `login_max_retries` (default 20); a give-up path is strictly better than hammering |
> | Two cables / two bound services | Both land on `route=2` (BUG 2) and the handshake cannot complete. Use `max_devices=1` — a config fix, not a safety one |
> | A boot logging `PSP firmware loading failed` | Power off fully; a warm reboot will not clear it |
>
> **Keep a non-Thunderbolt path (LAN/Wi-Fi) to every node** — good practice regardless.
>
> Lesson worth keeping: when a kernel module starts corrupting DMA after you patched it, suspect
> your patch before you suspect the platform. See [FINDINGS.md](FINDINGS.md) BUG 9.

The RCCL port in this repo is **transport-agnostic**: it calls `ncclAllReduce`, and RCCL picks
the wire. Over TCP that wire costs **286 µs/op**, which is what keeps cross-node tensor
parallelism behind pipeline. This directory covers replacing that wire with **RDMA over the
Thunderbolt/USB4 cables you already have**, using
[OdinLink-Five](https://github.com/Geramy/OdinLink-Five) — no new NIC.

**Measured on 2× Ryzen AI MAX+ 395 (gfx1151), kernel 7.0.0-28, single USB4 cable:**

| metric | RDMA (OdinLink) | TCP (bond0) |
|---|---|---|
| **latency, median** | **22.42 µs** | 286 µs/op |
| min / p95 / p99 | 13.60 / 33.52 / 34.62 µs | — |
| jitter (stddev) | 4.01 µs | — |
| bandwidth | 9.2 Gb/s | ~19 Gb/s (2 links) |

**~13× lower latency.** Bandwidth is *worse* — this is a latency win, which is exactly what a
per-layer all-reduce needs. The ~9 Gb/s ceiling is a **USB4v1 cable** limit; the routers report
`gen=4` (v2-capable), so a TB5 cable should scale it.

> **Status:** the link and the measurement are reproducible. Getting RCCL itself to *use* it
> requires upstream **[PR #20](https://github.com/Geramy/OdinLink-Five/pull/20)** — the plugin
> shipped in `main` cannot work (see [FINDINGS.md](FINDINGS.md), BUG 6/7/8). PR #20 reports
> TP=2 Llama-3.1-8B beating TCP by ~17–26 % on decode on this same hardware class.

## Prerequisites

Both nodes, same kernel:

```bash
sudo apt install build-essential cmake linux-headers-$(uname -r) pkg-config \
                 libibverbs-dev rdma-core libglib2.0-dev
gcc --version   # MUST match the compiler the kernel was built with (check /proc/version)
```

## 1. Build (both nodes)

```bash
git clone https://github.com/Geramy/OdinLink-Five.git && cd OdinLink-Five

# REQUIRED: main's RCCL plugin is unusable. Apply PR #20.
gh pr diff 20 --repo Geramy/OdinLink-Five > pr20.diff     # or curl .../pull/20.diff
git apply pr20.diff

cmake -B build -DBUILD_VERBS=ON -DBUILD_TRAY=OFF && cmake --build build -j$(nproc)
make -C driver                                            # builds odl_tb5.ko separately
```

Verify the plugin exports the symbol RCCL actually resolves:

```bash
nm -D --defined-only build/rccl/librccl_net_odl_tb5.so | grep NetPlugin
# must show ncclNetPlugin_v7   (main exports only rcclNetPlugin_v7 -> silently ignored)
```

## 2. Bring up the link (both nodes)

Use [`scripts/odl-bringup.sh`](scripts/odl-bringup.sh), or by hand:

```bash
# With 2 cables both services land on route=2 and the handshake never completes (BUG 2).
# Bind exactly one -- via the module parameter, NOT by unbinding afterwards
# (unbinding runs the same unsafe teardown path as rmmod; see the warning above):
sudo insmod driver/odl_tb5.ko e2e=0 max_devices=1 login_max_retries=20
```

> **You need the BUG 10 patch for two nodes to work at all.** Without it the second node
> always dies with `DMA verify failed after 300 attempts`, because the first node to verify
> resets its rings and posts zero RX frames. See [FINDINGS.md](FINDINGS.md) BUG 10.

Both nodes must reach READY — this is the only success signal that matters:

```bash
sudo dmesg | grep OdinLink | tail
# ... DMA pong received / DMA path verified / entering READY state
```

> The driver logs under **two** prefixes: `odl_tb5:` (load/probe) and `OdinLink:`
> (handshake/DMA). Grepping only the first hides every interesting failure.

## 3. Measure

```bash
# node A
./build/cli/odl_tb5_cli server -d $(ls /dev/odl_tb5_* | head -1 | grep -oE '[0-9]+$') -v
# node B
./build/cli/odl_tb5_cli client -d <N> -t latency -i 1000
./build/cli/odl_tb5_cli client -d <N> -t bandwidth -b 4K,64K,1M,4M
```

`/dev/odl_tb5_N` indices are assigned by probe order and **differ between nodes** — read them,
don't assume 0. Start the server over ssh with `exec` in a held session; `setsid … &` gets
orphaned and dies silently.

## 4. Point RCCL at it

```bash
export NCCL_NET_PLUGIN=ODL_TB5 NCCL_PLUGIN_DIR=/path/to/build/rccl
export NCCL_SOCKET_IFNAME=bond0        # bootstrap only; payload goes over RDMA
export NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=INFO
```

Confirm from the log that it actually bound:

```
NCCL INFO Initialized NET plugin ODL_TB5      <- good
NCCL INFO ... is unsupported                  <- plugin rejected, fell back to TCP silently
NCCL INFO NET/Socket : Using [0]bond0         <- you are NOT on RDMA
```

That silent fallback is the single easiest way to "measure RDMA" and actually be measuring TCP.

## Gotchas that cost real time

- **One `insmod` per boot — and never unload.** Unpatched, the driver leaks XDomain hop-IDs on
  unload/re-probe and the second load fails `enable_paths failed (-12)` (ENOMEM). The obvious
  workaround (reloading the whole TB stack) **is unsafe — see the warning at the top**; it
  wedged the USB4 controller and the GPU here. Apply the BUG 1 patches instead so you never
  need to reload, and **reboot** if you must load a different build.
- **If you ever do reload the stack, it disturbs your IP bond**, which shares the same cables:
  - flush stale ARP: `sudo ip neigh flush dev bond0`
  - a slave can show `carrier=1 operstate=up` and carry **zero** traffic; `balance-rr` keeps
    striping onto it → ~50–66 % loss that looks like "the peer is down". Find it with
    per-interface `rx_packets` during a ping, then `ip link set thunderboltN down`.
  - **keep a non-Thunderbolt path (LAN/Wi-Fi) to the peer** or you cannot debug any of this.
- `odl_tb5` and `thunderbolt_net` coexist (different XDomain services, `prtcid` 20300 vs 1), so
  RDMA and IP share the cables — but a stack reload resets both.
- **Never probe the RPC port with `/dev/tcp`.** `ggml-rpc-server` rejects the non-HELLO bytes
  and *exits*; if it is the container's PID 1 the container dies with it.

## Files

| | |
|---|---|
| [`RDMA-INFERENCE.md`](RDMA-INFERENCE.md) | **working RDMA inference recipe + measured data** |
| [`FINDINGS.md`](FINDINGS.md) | 12 bugs, root cause + patch for each, with dmesg evidence |
| [`scripts/odl-bringup.sh`](scripts/odl-bringup.sh) | load + single-service bind + wait for READY |
| [`scripts/odl-reload.sh`](scripts/odl-reload.sh) | no-reboot TB stack reload (clears the hop-ID leak) |
| [`scripts/odl-measure.sh`](scripts/odl-measure.sh) | latency + bandwidth run |
