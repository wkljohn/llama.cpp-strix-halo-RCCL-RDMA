# Reproducible: RCCL over OdinLink RDMA (Thunderbolt/USB4)

End-to-end recipe for running RCCL collectives over Thunderbolt RDMA on two
AMD Strix Halo (gfx1151) nodes, with no NIC.

## 0. What you need

- 2 nodes, one USB4/Thunderbolt cable between them
- An IP link over the same cables for bootstrap (we use `bond0`, 10.4.0.1/2)
- ROCm 7.2.0, and RCCL built for gfx1151

## 1. Build OdinLink (both nodes)

```bash
git clone -b strix-halo-verbs-fixes https://github.com/wkljohn/OdinLink-Five.git
cd OdinLink-Five
cmake -B build -DBUILD_VERBS=ON -DBUILD_TRAY=OFF && cmake --build build -j$(nproc)
make -C driver
```

Both nodes must run the **same build** — the wire format carries a fragment
index and an 8-byte stream header, so mismatched builds will not interoperate.

## 2. Build RCCL for gfx1151

Stock RCCL does not target gfx1151. Use the patched tree:

```bash
git clone --depth 1 -b gfx1151-rccl https://github.com/kyuz0/rocm-systems.git
cd rocm-systems/projects/rccl && mkdir -p build && cd build
CXX=/opt/rocm/bin/hipcc cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DDEFAULT_GPUS="gfx1151" -DGPU_TARGETS="gfx1151" -DAMDGPU_TARGETS="gfx1151" \
  -DBUILD_TESTS=OFF -DGENERATE_SYM_KERNELS=OFF -DENABLE_AMDSMI=OFF \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 3. Bring up the link (both nodes)

```bash
sudo rmmod thunderbolt_ibverbs 2>/dev/null   # conflicts: same NHI DMA
sudo insmod driver/odl_tb5.ko e2e=0 max_devices=1 login_max_retries=20 odl_busy_poll_us=50
```

`max_devices=1` is required with two cables: both XDomain services report
`route=2` and the driver's route lookup cannot tell them apart (upstream defect),
so binding both makes the handshake fail. Bind one from the start rather than
unbinding afterwards.

Both nodes must reach READY — this is the only success signal:

```bash
sudo dmesg | grep OdinLink | tail
#   DMA path verified, resetting rings for userspace
#   entering READY state
```

The login handshake is racy. If one side reports `login request failed: -110`
repeatedly, unload on both and reload them within a few seconds of each other;
it usually succeeds within 1-3 attempts.

## 4. Verify the transport before involving RCCL

Do this first — it isolates transport faults from RCCL faults, and it verifies
every byte (a corrupting transport still produces plausible benchmark numbers):

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

Expect `PASS: 8192 msgs ... recv+verified 8192/8192` on **both** peers.
Exit codes: `0` ok, `2` data corruption, `3` stall.

## 5. Point RCCL at OdinLink

**Two gotchas here, both of which silently drop you onto TCP.**

**(a) The built filename is not what RCCL looks for.** The build produces
`librccl_net_odl_tb5.so` (underscores, lowercase) but RCCL `dlopen`s
`librccl-net-<PLUGIN_NAME>.so` — hyphens, and the plugin name's exact case:

```bash
cd OdinLink-Five/build/rccl
ln -sf librccl_net_odl_tb5.so librccl-net-ODL_TB5.so
ln -sf librccl_net_odl_tb5.so libnccl-net-ODL_TB5.so   # for NCCL-named lookups
```

**(b) `NCCL_PLUGIN_DIR` is NOT honoured** by this RCCL build — the plugin is
resolved through the dynamic loader, so the directory must be on
`LD_LIBRARY_PATH`:

```bash
export NCCL_NET_PLUGIN=ODL_TB5
export LD_LIBRARY_PATH=/path/to/OdinLink-Five/build/rccl:$LD_LIBRARY_PATH
export NCCL_SOCKET_IFNAME=bond0     # bootstrap only; payload goes over RDMA
export NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

Confirm from the log that it actually bound:

```
NCCL INFO NET/Plugin: Loaded net plugin ODL_TB5 (v7)          <- good
NCCL INFO Successfully loaded external network plugin .../librccl-net-ODL_TB5.so
NCCL INFO Initialized NET plugin ODL_TB5
NCCL INFO Using network ODL_TB5                               <- good

NCCL INFO NET/Plugin: Could not find: ODL_TB5 librccl-net-ODL_TB5.so   <- (a) or (b)
NCCL INFO Using network Socket                                <- you are on TCP
```

Verified working output on this rig is the first block. If you see
`Could not find`, you hit gotcha (a) or (b) above.

That silent fallback is the easiest way to "measure RDMA" and actually measure
TCP. Always check.

## How far this actually gets — read before quoting any number

**Plugin binding is verified. A tensor-parallel benchmark over it is not.**

RCCL selects `ODL_TB5` instead of sockets — the log block above is real output from this
rig. What has *not* been done is a completed `-sm tensor` run over that link. Nothing is
known to block it: `-sm tensor` runs over TCP (27B at 3.65/4.02 t/s, in the repo root),
and the rendezvous deadlock that once stopped it is fixed
([FINDINGS.md](FINDINGS.md) BUG 12). The run has simply not been made.

Everything in [RESULTS.md](RESULTS.md) with a t/s figure is `-sm layer` (pipeline) over
the RPC transport, not RCCL tensor parallel. The TP numbers in the repo root are TCP-era.

### If you are the one to run it

Go in this order — each stage isolates a different failure:

1. `odl_rdma_stress --bidir` — transport is byte-clean. Skip and you cannot tell a
   transport fault from an RCCL fault.
2. `tests/test-world-allreduce` with `NCCL_NET_PLUGIN=ODL_TB5` — the collective alone,
   no llama.cpp. Expect `reduced to 1.50`.
3. 2B model, `-sm tensor`. 4. 27B.

**Prove you are on RDMA at every stage.** Three separate traps above drop you onto TCP
without an error. Watch `bond0` byte counters across a run — flat means RDMA carried it —
and do not trust a t/s number as evidence of transport.

Two open defects are likely to bite collectives specifically: teardown can hang
(BUG 24) and `connect`/`accept` return errors where RCCL expects a retry (BUG 25). Both
are source findings, not yet reproduced.

## Notes / known limits

- `odl_tb5` and `thunderbolt_ibverbs` cannot coexist — both drive the same NHI.
- `odl_tb5` coexists fine with `thunderbolt_net`, so your IP bond keeps working.
- The verbs path is an LD_PRELOAD shim: OdinLink registers no kernel `ib_device`,
  so rdma-core cannot discover it and `ibv_devices` is empty without the preload.
- Reloading the whole Thunderbolt stack disturbs the IP bond on the same cables
  (`ip neigh flush dev bond0`, and watch for a slave with carrier but no traffic).
