# Reproducible: llama.cpp cross-node inference over RDMA (`-sm layer`)

The fastest cross-node path measured here — **9.16 t/s** on the 27B versus 8.83 over
TCP. Two Strix Halo nodes, tensor traffic over RDMA on the Thunderbolt/USB4 cable,
**no NIC**. Numbers in [RESULTS.md](RESULTS.md).

For RCCL collectives (`-sm tensor`) see [REPRODUCE-RCCL.md](REPRODUCE-RCCL.md) instead.

## 0. What you need

- 2 nodes, one USB4/Thunderbolt cable between them
- An IP link over the same cables for the RPC control channel (here `bond0`,
  10.4.0.1/2) — the transport matches an RDMA GID against this address
- ROCm 7.2.0; llama.cpp built with `-DGGML_RPC=ON -DGGML_RPC_RDMA=ON`

## 1. Build OdinLink (both nodes)

```bash
git clone -b strix-halo-verbs-fixes https://github.com/wkljohn/OdinLink-Five.git
cd OdinLink-Five
cmake -B build -DBUILD_VERBS=ON -DBUILD_TRAY=OFF && cmake --build build -j$(nproc)
make -C driver
```

Both nodes must run the **same build** — the wire format carries a fragment index and
an 8-byte stream header, so mismatched builds will not interoperate.

## 2. Bring up the link (both nodes)

```bash
sudo rmmod thunderbolt_ibverbs 2>/dev/null   # conflicts: same NHI DMA
sudo insmod driver/odl_tb5.ko e2e=0 max_devices=1 login_max_retries=20 odl_busy_poll_us=50
```

`max_devices=1` is required with two cables: both XDomain services report `route=2` and
the driver's route lookup cannot tell them apart, so binding both makes the handshake
fail. Bind one from the start rather than unbinding afterwards.

Both nodes must reach READY — the only success signal that matters:

```bash
sudo dmesg | grep OdinLink | tail
#   DMA path verified, resetting rings for userspace
#   entering READY state
```

The login handshake is racy. If one side reports `login request failed: -110`
repeatedly, unload on both and reload within a few seconds of each other; it usually
succeeds within 1–3 attempts.

## 3. Verify the transport before involving llama.cpp

Do this first. It isolates transport faults from inference faults, and it checks every
byte — `llama-bench` measures speed, not correctness, so a corrupting transport still
prints a plausible t/s. One did (see BUG 21 in [FINDINGS.md](FINDINGS.md)).

```bash
gcc -O2 -o odl_rdma_stress tests/odl_rdma_stress.c -libverbs -lpthread
export LD_PRELOAD=$PWD/build/verbs/libodl_tb5_verbs.so
export LD_LIBRARY_PATH=$PWD/build/verbs:$PWD/build/lib
export ODL_RDMA_GID_IFACE=bond0

# peer
./odl_rdma_stress --server --local-ip 10.4.0.2 --total 2G --bidir
# head
./odl_rdma_stress --client 10.4.0.2 --local-ip 10.4.0.1 --total 2G --bidir
```

Expect `recv+verified 8192/8192` on **both** peers. Exit codes: `0` ok, `2` data
corruption, `3` stall. `--bidir` matters — the request/response pattern on one QP is
what broke the 27B when unidirectional bulk was already running 21 GiB clean.

## 4. Run inference

`ibv_devices` is empty without the preload — OdinLink registers no kernel `ib_device`,
so the discovery shim has to be in the process. Both sides need it:

```bash
export LD_PRELOAD=/path/to/OdinLink-Five/build/verbs/libodl_tb5_verbs.so
export ODL_RDMA_GID_IFACE=bond0
export GGML_RDMA_DEV=odl_tb5_0        # check: ibv_devices

# peer
./ggml-rpc-server -H 10.4.0.2 -p 50052

# head
./llama-bench -m model.gguf --rpc 10.4.0.2:50052 -sm layer -ngl 99 -p 512 -n 128 -r 2
```

Confirm RDMA actually engaged:

```
RDMA probed:    dev=odl_tb5_0 gid=0 RoCEv2 qpn=20 inline=256   <- good
RDMA activated: qpn=20->20 mtu=4096 rx_depth=24                <- carrying traffic
```

**No `RDMA activated` line means it fell back to TCP and your "RDMA" numbers are TCP
numbers.** That silent fallback is the single easiest way to measure the wrong thing.

## Tuning

`ODL_VERBS_INLINE=1` (default) takes small sends down the inline fast path, skipping a
thread handoff, a malloc+memcpy and two `poll()` syscalls on an empty pipeline. It
improves the floor and the jitter, not the median — see [RESULTS.md](RESULTS.md).
Set `0` to A/B it.

## Alternative: `thunderbolt-ibverbs`

A DKMS module that registers a real kernel `ib_device` (`usb4_rdma0`), so no preload is
needed. Slightly slower here (9.07 vs 9.16 t/s) but a cleaner integration. It cannot run
at the same time as `odl_tb5` — both drive the same NHI DMA.

```bash
sudo apt install thunderbolt-ibverbs-dkms   # needs Linux 6.14+
sudo rmmod odl_tb5
```

Two traps, both of which cost real time:

**Set module options in `modprobe.d`, not on the command line.** udev auto-loads the
module when the Thunderbolt service appears, *before* any manual `modprobe`, so a later
`modprobe thunderbolt_ibverbs profile=...` is a silent no-op — it is already loaded with
defaults (`register_verbs=N`) and registers nothing.

```bash
sudo tee /etc/modprobe.d/thunderbolt-ibverbs.conf <<'EOF'
options thunderbolt_ibverbs profile=linux_perf bind_services=1 allocate_rings=1 \
        start_rings=1 negotiate_native=1 enable_tunnels=1 register_verbs=1 roce_netdev=bond0
EOF
```

`roce_netdev` must be the interface whose IP the RPC server binds to.

**Disable rdma-core persistent naming.** The module registers `usb4_rdma0`, but
rdma-core's udev rule renames RDMA devices by PCI path — `usb4_rdma0` becomes
`rocep199s0f5`. The userspace provider matches on modalias `rdma_device:*Nusb4_rdma*`,
i.e. **on the name**, so after the rename nothing matches:

```
libibverbs: Warning: no userspace device-specific driver found for rocep199s0f5
```

with `ibv_devices` empty even though `rdma link` shows the device ACTIVE.

```bash
sudo ln -sf /dev/null /etc/udev/rules.d/60-rdma-persistent-naming.rules
sudo udevadm control --reload
sudo modprobe -r thunderbolt_ibverbs; sudo modprobe thunderbolt_ibverbs
ibv_devices        # must list usb4_rdma0
```

Its peer handshake is racy too: `native HELLO ... failed after 5 attempts: -110` is
common when both nodes load simultaneously. Load the peer first, wait ~5 s, then the
head. Then run as above with `GGML_RDMA_DEV=usb4_rdma0` and no `LD_PRELOAD`.

## Why the win is only ~4 %

`-sm layer` crosses the wire **once per token**, so at ~9 t/s the transport contributes
~22 µs against a ~110 ms/token budget. Even halving RTT would be invisible. RDMA's
advantage is latency, not bandwidth, and prompt processing is bandwidth-bound — hence
pp512 is unchanged and only decode moves.

The configuration that *should* benefit far more is `-sm tensor`, which all-reduces every
layer and is crushed by the 286 µs TCP floor. That case is blocked on BUG 12, the RCCL
rendezvous deadlock.

And **single-node is still fastest** for a model that fits in one 96 GB carve. Two nodes
are for capacity, not speed.

## Known limits

- The verbs path is an `LD_PRELOAD` shim. Anything that does not inherit the preload
  cannot see the device; a kernel `ib_device` is the proper fix (BUG 11).
- Loss is detected, not retransmitted (BUG 22). A gap fails the message loudly.
- `odl_tb5` coexists with `thunderbolt_net`, so the IP bond keeps working — but a
  Thunderbolt stack reload resets both. See [FINDINGS.md](FINDINGS.md) for bond recovery.
