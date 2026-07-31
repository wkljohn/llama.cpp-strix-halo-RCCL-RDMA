# Setup gotchas: getting RPC-over-OdinLink actually running

Everything here cost a debugging round on 2026-07-30. `REPRODUCE-RPC.md` documents
the *commands*; this documents the environment they need, which is where the time
actually goes. Symptoms are listed first because that is how you will meet them.

---

## 1. `ibv_devices` is empty — OdinLink registers no kernel `ib_device`

**Symptom:** `ibv_devices` lists nothing, `GGML_RDMA_DEV=odl_tb5_0` silently falls
back to TCP, and your "RDMA" numbers are TCP numbers.

**Cause:** OdinLink is a userspace verbs provider. There is no kernel `ib_device`
to enumerate, so discovery only works with the shim loaded *into the process*.

**Fix — required on BOTH sides:**
```bash
export LD_PRELOAD=<OdinLink>/build/verbs/libodl_tb5_verbs.so
export ODL_RDMA_GID_IFACE=bond0
export GGML_RDMA_DEV=odl_tb5_0
ibv_devices        # must now list odl_tb5_0
```

---

## 2. The preload itself fails to load on a node without the system-wide lib

**Symptom:**
```
timeout: error while loading shared libraries: libodl_tb5.so.0: cannot open shared object file
```
(the failing binary is whatever ran under `LD_PRELOAD`, not OdinLink itself — the
preload applies to every process in the chain, which makes the error look unrelated)

**Cause:** `libodl_tb5_verbs.so` links `libodl_tb5.so.0`. On the head that lives in
`/usr/lib/x86_64-linux-gnu/`; on a node where OdinLink was only *built*, never
*installed*, it exists solely in the build tree.

**Fix:** add the build lib dir to the loader path on the affected node:
```bash
export LD_LIBRARY_PATH=<OdinLink>/build/lib:$LD_LIBRARY_PATH
```
Check which case you are in with `ldd <OdinLink>/build/verbs/libodl_tb5_verbs.so | grep odl`.

---

## 3. `ggml-rpc-server` dies the moment the ssh session ends

**Symptom:** the server logs its banner, then `ss -tln | grep 50052` shows nothing
seconds later. `setsid`, `nohup`, `disown`, and redirecting all three fds do **not**
save it.

**Cause:** the process is in the ssh session's cgroup; systemd tears the whole scope
down at logout.

**Fix — `systemd-run --user`, plus two prerequisites that are easy to miss:**

```bash
# (a) linger, or the user manager itself exits at logout and takes the unit with it
sudo loginctl enable-linger "$USER"      # persists across reboots

# (b) non-interactive ssh does NOT set these, and systemd-run --user needs both
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus

systemd-run --user --unit=ggml-rpc --collect \
  --setenv=LD_LIBRARY_PATH=<llama.cpp>/build/bin:<OdinLink>/build/lib \
  --setenv=LD_PRELOAD=<OdinLink>/build/verbs/libodl_tb5_verbs.so \
  --setenv=ODL_RDMA_GID_IFACE=bond0 \
  --setenv=GGML_RDMA_DEV=odl_tb5_0 \
  <llama.cpp>/build/bin/ggml-rpc-server -H <peer-ip> -p 50052
```

Without (a) the unit starts and dies at logout. Without (b) `systemd-run` fails and
`systemctl --user status ggml-rpc` reports *"Unit could not be found"* — no unit is
ever created, which reads like the command silently did nothing.

**Verify from the head, not just the peer:**
```bash
ssh peer 'ss -tln | grep :50052'
cat </dev/null >/dev/tcp/<peer-ip>/50052 && echo reachable
```

---

## 4. `ggml-rpc-server` cannot find `libggml.so.0`

**Symptom:** `error while loading shared libraries: libggml.so.0`

**Cause:** the ggml shared objects sit beside the binary in `build/bin` and are not
on the default loader path.

**Fix:** `LD_LIBRARY_PATH=<llama.cpp>/build/bin` (already included above).

---

## 5. `-dev ROCm0,RPC0` rejected with a bare usage dump

**Symptom:** llama-server prints the `--device` usage text and exits.

**Cause:** there is no `RPC0` device, because the RPC server is not reachable. The
error names the flag, not the real problem.

**Fix:** confirm the device list *before* launching the model:
```bash
llama-server --rpc <peer>:50052 --list-devices
#   ROCm0: AMD Radeon Graphics (98304 MiB, ... free)
#   RPC0:  <peer>:50052        (98304 MiB, ... free)
```
Treat a missing `RPC0` as a hard abort. Do not proceed to a 10-minute model load.

---

## 6. `pkill -f ggml-rpc-server` kills the ssh session running it

Already noted in the two-node RPC notes, repeated because it recurs: the pattern
matches the invoking command line. Use a bracketed pattern or kill by PID:
```bash
for p in $(pgrep -f 'ggml-rpc-serve[r]'); do kill -9 "$p"; done
```

---

## Order to bring it up

Each step is cheap and fails loudly; the model load is the only expensive one, so
put it last.

1. `lsmod | grep odl_tb5` and `/dev/odl_tb5_0` present on both nodes
2. `ibv_devices` shows `odl_tb5_0` on both, **with the preload env**
3. `ggml-rpc-server` up under `systemd-run --user`; port reachable from the head
4. `llama-server --list-devices` shows `RPC0`
5. Launch the model
6. **Confirm `RDMA activated` in the log.** No such line means it fell back to TCP.

---

## The `--no-mmap` requirement is unrelated but bites here too

`amdgpu: SVM mapping failed, exceeds resident system memory limit` — SVM
registration is capped by *system RAM* (~31 GB), not the 96 GiB VRAM carve-out.
Always pass `--no-mmap` on the HIP head for large models.
