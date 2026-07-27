# RDMA over Thunderbolt/USB4 on Strix Halo — OdinLink bring-up

> ## ⚠️ STATUS: TREAT AS UNSAFE ON AMD STRIX HALO — CAUSE NOT YET ISOLATED
>
> Testing here ended after ~6 hard freezes across two machines. The chain always started with
> an IOMMU fault on the Thunderbolt controller and took the GPU down with it:
>
> ```
> thunderbolt 0000:c7:00.6: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x003f
>                                                 address=0xffbb8000 flags=0x0020]
> amdgpu 0000:c5:00.0: probe with driver amdgpu failed with error -22
> drm_buddy_fini+0x112/0x120 [drm_buddy]
> BUG: kernel NULL pointer dereference   ->   hard freeze, manual power cycle
> ```
>
> The USB4 router shares a firmware/power domain with the iGPU on Strix Halo, so once the router
> wedges, `amdgpu` can fail to initialise — sometimes on the *next boot* as well.
>
> **Honesty note:** an earlier version of this page blamed the driver's ring DMA mapping. That
> claim has been **withdrawn**. The rings do use `dma_alloc_coherent(tb_ring_dma_device(...))`
> — the correct API — and the crash runs were made with a **locally patched build whose
> `odl_tb5_remove()` had been accidentally gutted**, freeing the DMA buffers while the rings
> were still armed. That is its own sufficient explanation for the fault, so those runs prove
> nothing about upstream. The teardown has been repaired; **the corrected build has not yet been
> retested.** See [FINDINGS.md](FINDINGS.md) BUG 9.
>
> Until someone reproduces this on a **clean upstream checkout**, assume the risk is real but
> the cause is unknown. Don't load `odl_tb5` on a machine you cannot afford to power-cycle.
>
> ## ⛔ Individually dangerous paths (all still true)
>
> Every item below was reproduced repeatedly on 2× Ryzen AI MAX+ 395. These are **not**
> theoretical. Several ended in a hard freeze needing a manual power cycle, and twice the
> **GPU failed to initialise on the following boot** (`amdgpu ... PSP firmware loading
> failed` → `probe with driver amdgpu failed with error -22`).
>
> | ❌ Never do this | What happens |
> |---|---|
> | `rmmod odl_tb5` | NHI DMA rings stay armed → USB4 router wedges → `thunderbolt` core oopses in `tb_cfg_read`/`check_config_address` → GPU dies with it (shared firmware domain) → panic |
> | Unbind a service to "fix" BUG 2 | **Same teardown path as rmmod.** Crashed a node with no `rmmod` involved. Use `max_devices=1` instead so the extra service never binds |
> | Leave the module loaded while the handshake fails | The login retry loop **never gives up**. Hammering a wedged router ends in `amdgpu_irq_put` / `drm_buddy_fini` → `BUG: kernel NULL pointer dereference` → freeze. Use `login_max_retries` |
> | Reload the whole TB stack repeatedly | Wedges the USB4 controller *and* the GPU PSP. Recovery needs a **full power cycle**, not a warm reboot |
>
> **Safe operating rules**
> 1. **Load `odl_tb5` exactly once per boot. Never unload it.** To test a new build, reboot first.
> 2. **One cable, or `max_devices=1`.** With two cables both services sit at `route=2` (BUG 2) and the handshake cannot complete — and the unbind workaround is itself unsafe.
> 3. **Bound the retries**: `login_max_retries=20` (default in the patched driver).
> 4. **Keep a non-Thunderbolt path (LAN/Wi-Fi) to every node.** You will need it.
> 5. If a boot logs `PSP firmware loading failed`, **power off fully** — a warm reboot will not clear it.
>
> The patched driver in this repo adds `max_devices`, `only_domain` and `login_max_retries`
> precisely so none of the dangerous manual steps are needed. See
> [FINDINGS.md](FINDINGS.md) for root causes and patches.

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
| [`FINDINGS.md`](FINDINGS.md) | 9 bugs, root cause + patch for each, with dmesg evidence |
| [`scripts/odl-bringup.sh`](scripts/odl-bringup.sh) | load + single-service bind + wait for READY |
| [`scripts/odl-reload.sh`](scripts/odl-reload.sh) | no-reboot TB stack reload (clears the hop-ID leak) |
| [`scripts/odl-measure.sh`](scripts/odl-measure.sh) | latency + bandwidth run |
