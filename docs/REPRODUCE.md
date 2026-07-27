# Reproducible RCCL cross-node tensor-parallel for llama.cpp (Strix Halo)

Adds an **in-graph RCCL all-reduce** so `-sm tensor` (tensor parallelism) runs across
separate nodes with the collective rendezvousing inside RCCL, instead of the host-side
"butterfly" of GET/SET_TENSOR RPC round-trips.

The communicator is N-rank — `GGML_NCCL_WORLD` sets the size and rank 0 serves the unique
ID to all `N-1` peers. Everything below is written for and measured on two nodes, which is
the hardware available here; larger worlds should follow from the same steps but have not
been run.

**Measured (Qwen3.5-2B, `-sm tensor`, 2× Strix Halo over the Thunderbolt bond, TCP):**
| path | t/s |
|---|---|
| butterfly (upstream) | 19.35 ± 0.15 |
| RCCL world-allreduce (this) | 19.28 ± 0.17 |

Break-even over TCP — the per-sync RPC-graph dispatch+sync overhead matches the butterfly at
this scale. The collective's own latency is ~286 µs/op (measured). The win needs a real RDMA
floor (~30–100 µs/op); over Thunderbolt-TCP it's a wash. **That floor now exists without a NIC**
— 22.0 µs over the Thunderbolt cable, see [../odinlink/](../odinlink/) — but `-sm tensor` has
**not** been benchmarked over it. The run simply has not been made, so every number on this page
is TCP. The value delivered here is a
**working, correct, reproducible implementation** to measure against — and two upstream bug
fixes that make `-sm tensor` over RPC work at all.

Base commit this was developed on: `635cdd5`. Branch: `nccl-tp`. Single patch:
`../rccl-tp-port.patch` (regenerate with `git diff <base> nccl-tp -- ggml/`).

---


## Measured results — full A/B (2 nodes, Thunderbolt bond, TCP, in-container ROCm 7.14)

| model | -sm layer (pipeline) | -sm tensor + butterfly | -sm tensor + RCCL (this) |
|---|---|---|---|
| Qwen3.5-2B Q8_0 | 70.98 | 19.35 ± 0.15 | 19.28 ± 0.17 (break-even) |
| Huihui-Qwen3.6-27B Q6_K | **8.87** | 3.10 ± 0.04 | **3.65 ± 0.04 (+18%)** |

Reading it honestly:
- The RCCL port WORKS end-to-end on both, and **beats the butterfly by ~18% on the 27B** (larger
  per-allreduce tensors let the collective's efficiency show; the 2B is too small — per-sync
  overhead dominates, break-even).
- **But `-sm layer` (pipeline) is the fastest cross-node path** (8.87 vs 3.65 on the 27B): tensor-
  parallel does an all-reduce EVERY layer, and over TCP that per-sync cost dominates regardless of
  butterfly-vs-RCCL. Pipeline crosses the wire once per token. This matches the upstream guidance
  (cross-node = pipeline; TP = intra-node). The RCCL win is *within* the TP path, not vs pipeline.
- The RCCL TP number is bottlenecked by the ~286us/allreduce TCP floor; a lower RDMA floor is what
  would let TP-RCCL overtake pipeline. 22.0us is measured over Thunderbolt RDMA, but the TP run over
  it has not been made — so this remains untested, not disproven. Note the 2B break-even above says
  per-sync dispatch overhead alone roughly equals the whole butterfly cost, and RDMA does not touch
  that, so a lower collective floor may not be sufficient. On this rig, for a model that FITS on one
  node, single-node (~12.9 t/s for the 27B) still beats all cross-node options — TP's value is
  capacity (models too big for one 96GB carve), where the +18% is the payoff.
- STABILITY: the 27B world path intermittently HANGS at startup (head zombies, peer spins 100% GPU
  in the collective) — a rendezvous race to fix before this is production-usable. Re-run succeeds.

## The changes (7 files) — so they can be re-applied after a llama.cpp update

Re-apply by understanding each change and finding its new anchor (line numbers drift; the
anchors are code, not line numbers). Order matters only for the op-enum (do it first).

### 1. New op `GGML_OP_ALLREDUCE` — `ggml/include/ggml.h`, `ggml/src/ggml.c`
- `ggml.h`: add `GGML_OP_ALLREDUCE` to `enum ggml_op` **immediately before `GGML_OP_COUNT`**.
  Declare `GGML_API struct ggml_tensor * ggml_allreduce(struct ggml_context *, struct ggml_tensor *);`
- `ggml.c`: append `"ALLREDUCE"` to `GGML_OP_NAME[]` and `"allreduce(x)"` to `GGML_OP_SYMBOL[]`
  (both arrays are sized `[GGML_OP_COUNT]`, so both need the new entry), and **bump the two
  `static_assert(GGML_OP_COUNT == N)` to N+1**. Add the builder `ggml_allreduce()` (dup_tensor,
  `op = GGML_OP_ALLREDUCE`, `src[0] = a`) near `ggml_cont`.
- WHY: a graph op the CUDA/HIP backend executes as a collective. GGML_OP_COUNT changing is why
  the RPC proto guard (#4) trips — that's expected and safe (existing op numbers don't move).

### 2. Cross-process "world" communicator — `ggml/src/ggml-cuda/ggml-cuda.cu` (the bulk, ~180 lines)
- Add `ggml_cuda_world_init_once()`: builds a process-global `ncclComm_t g_world_comm` with
  **`ncclCommInitRank`** (NOT `ncclCommInitAll` — that's local-GPUs-only). Rank 0 opens a TCP
  listener, generates the `ncclUniqueId`, sends it to each connecting peer; rank >0 retries-
  connects (~60 s) and receives it. Config via env `GGML_NCCL_RANK` / `GGML_NCCL_WORLD` /
  `GGML_NCCL_MASTER=<ip:port>`. Guarded by `#ifdef GGML_USE_NCCL`; includes `<arpa/inet.h>` etc.
- In `ggml_cuda_compute_forward`'s op switch, add `case GGML_OP_ALLREDUCE:` → `world_init_once()`
  then `ncclAllReduce(src0->data, dst->data, ne, dt, ncclSum, g_world_comm, ctx.stream())`
  (f32/f16/bf16). In-place is fine (src0 aliases dst).
- In `ggml_backend_cuda_device_supports_op`, add `case GGML_OP_ALLREDUCE:` returning
  `ggml_cuda_world_available() && type∈{f32,f16,bf16}`.
- In the CUDA-graph node-scan (the `for` loop that sets `use_cuda_graph = false` for unsupported
  nodes), add `if (node->op == GGML_OP_ALLREDUCE) use_cuda_graph = false;` — a blocking collective
  can't be graph-captured (same reason vLLM forces eager).
- Export `ggml_backend_cuda_world_init()` in `ggml/include/ggml-cuda.h` (#3).

### 3. Export — `ggml/include/ggml-cuda.h`
- Declare `GGML_BACKEND_API bool ggml_backend_cuda_world_init(void);` (optional early init).

### 4. RPC proto guard — `ggml/include/ggml-rpc.h`
- Bump `RPC_PROTO_PATCH_VERSION` +1 and update the `static_assert(GGML_OP_COUNT == N)` to N+1.
  WHY: the op enum grew; this keeps head/peer in sync and documents the wire change.

### 5. **BUGFIX** empty-slice bounds — `ggml/src/ggml-rpc/ggml-rpc.cpp` (`deserialize_tensor`)
- Change the buffer-bounds guard from `if (result->buffer) { …GGML_ASSERT(data within buffer)… }`
  to `if (result->buffer && !ggml_is_empty(result)) { … }`.
- WHY: `-sm tensor` splitting hands a rank a **zero-sized slice** (`ne=[N,0,1,1]`, nbytes=0) whose
  data pointer legitimately lands past the buffer end (nothing is accessed). The old assert aborted
  the ggml-rpc-server mid-graph → "Remote RPC server crashed or returned malformed response".
  **This is what unbreaks `-sm tensor` over RPC.** (Distinct from #21006/PR#21030, already in base.)

### 6. Route the TP sync point through the collective — `ggml/src/ggml-backend-meta.cpp` (~48 lines)
- In `ggml_backend_meta_graph_compute`, add lambda `allreduce_world(i)`: for each backend build a
  one-node in-place `GGML_OP_ALLREDUCE` aux-graph over that backend's shard tensor (mirror the
  existing `allreduce_fallback` aux-node pattern: `get_node_aux`, view of the node, `get_cgraph_aux`).
- **Dispatch order is critical**: iterate backends **high→low (n-1 … 0)**. Backend 0 is the local
  CUDA backend whose `graph_compute_async` runs the allreduce INLINE and blocks in `world_init`
  (rank-0 accept); the RPC backends are fire-and-forget. Dispatch RPC first so the peer's allreduce
  is en route and its `world_init` retry-connects before the local rank-0 blocks. Then
  `ggml_backend_synchronize` each. **Getting this backwards = deadlock.**
- Gate on `getenv("GGML_META_WORLD_ALLREDUCE")`; when set, `continue` past the butterfly path.

### 7. (pre-existing, unrelated) `ggml/src/ggml-rpc/transport.cpp`
- The thunderbolt_ibverbs RDMA sliding-window sender — a separate earlier local patch, not part of
  this port. Leave as-is.

---

## Build (in-container — the ONLY combination that runs here)

The peer is Vulkan-only natively and native ROCm 7.2 crashes in `hipStreamCreate`; the container's
ROCm **7.14 `_rocm_sdk` stack** (the ABI vLLM uses) is the proven-working runtime. Build+run BOTH
ranks in-container. See `../VLLM-TP2-STRIX-HALO.md` for the container itself.

```bash
# container: docker.io/kyuz0/vllm-therock-gfx1151:latest, run with --network host --ipc host
#   --device /dev/dri --device /dev/kfd --group-add video --group-add render -v <models>:/models
# copy the patched llama.cpp source into the container at /full, then:
cmake -S /full -B /full/build -DGGML_HIP=ON -DGGML_HIP_RCCL=ON -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build /full/build --target llama-bench ggml-rpc-server -j$(nproc)
# find_package(rccl) resolves the container's native gfx1151 RCCL at /opt/rocm/lib.
```
Deploy the built `bin/ggml-rpc-server` + `bin/libggml*.so.*` to the PEER container **at the same
paths** (the loaded copy is `/full/build/bin/libggml-*.so.0.16.0`; `podman cp` it directly — a
`cp` inside the container can't see the host's /tmp).

---

## Runtime (launch)

Common env (both nodes):
```bash
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:\
/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151/lib:/full/build/bin
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0
export GGML_NCCL_WORLD=2 GGML_NCCL_MASTER=10.4.0.1:29500     # head bond IP
```
PEER (rank 1) — rpc-server, bind 0.0.0.0 (binding the specific IP fails in-container),
launch with `setsid … </dev/null` so it survives the exec:
```bash
export GGML_NCCL_RANK=1
setsid /full/build/bin/ggml-rpc-server -H 0.0.0.0 -p 50070 </dev/null >/tmp/rpc.log 2>&1 &
```
HEAD (rank 0) — llama-bench, add `GGML_META_WORLD_ALLREDUCE=1` to use the RCCL path (omit for the
butterfly baseline):
```bash
export GGML_NCCL_RANK=0 GGML_META_WORLD_ALLREDUCE=1
/full/build/bin/llama-bench -m /models/<model>.gguf --rpc 10.4.0.2:50070 \
  -sm tensor -ts 50/50 -ngl 99 -p 0 -n 64 -r 3
```
Model arch must NOT be in `llm_arch_supports_sm_tensor()`'s false-list (qwen35, llama, qwen2/3
etc. are fine; deepseek2, glm-dsa, mamba, minimax, grok are not). Only the head needs the GGUF.

---

## Verify it actually used RCCL (not the fallback)
Peer `/tmp/rpc.log` should show `world communicator up: rank 1/2`. Backend column reads `ROCm,RPC`.
`llama-bench` shows `sm=tensor`. If the peer never joins, the head hangs in `accept()` — check the
dispatch order (#6) and that both processes share `GGML_NCCL_MASTER`/`WORLD` and `NCCL_SOCKET_IFNAME=bond0`.

## Gotchas (each cost real debugging time)
- `.so` deploy: the LOADED lib is `/full/build/bin/libggml-*.so.0.16.0`; incremental builds don't
  always refresh the `bin/` copy — `podman cp` the exact file. `ggml-cuda.cu` is in `libggml-hip.so`,
  `ggml-backend-meta.cpp`/`ggml-rpc.cpp` in `libggml-base.so`/`libggml-rpc.so`.
- rpc-server must bind `0.0.0.0`, not the node IP; launch with `setsid` or it dies with the exec.
- `NCCL_CUMEM_ENABLE=0` needed (gfx1151 HIP lacks the cuMem oversubscribe path the lib probes).
- The container's `/opt/rocm` librccl segfaults standalone; the `_rocm_sdk` one (on LD_LIBRARY_PATH
  above) works. `HSA_OVERRIDE_GFX_VERSION=11.5.1` optional.

Related: `NCCL-TP-PORT.md` (design history), `VLLM-TP2-STRIX-HALO.md` (container + vLLM baseline),
memory `strix-halo-vllm-tensor-parallel-rccl-mod.md`.
