# Speculative decoding on hy_v3 across a 2-node RPC split

Everything here was measured on 2026-07-30/31 against **Hy3-heretic 295B-A21B**
(Q4_K_S, 167.68 GB) layer-split across two Strix Halo nodes over OdinLink RDMA.

The short version: **speculation gains ~4% and no more**, and the reason is the MoE's
expert-weight traffic, not the RPC link. Four llama.cpp bugs had to be fixed before any
of it worked at all — each of which fails *silently*, with a model that loads, generates
correct text, and reports `draft acceptance = 0.00000`.

Patch: `../hy3-speculative-fixes.patch` (applies to llama.cpp `aadd131`).

---

## Results

Target: Hy3-heretic Q4_K_S, `-c 131072 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 --no-mmap`,
`--tensor-split 48,52 -dev ROCm0,RPC0`.

| config | decode t/s | acceptance | mean accepted len |
|---|---|---|---|
| plain (no draft) | 13.68 | - | - |
| **MTP n=1** | **14.2 - 14.6** | 0.60-0.63 | 1.60 |
| MTP n=2 | 13.17 | 0.484 | 1.96 |
| MTP n=3 | 11.90 | 0.373 | 2.11 |
| DFlash-B8 n=2 | 12.54 | 0.490 | 1.97 |
| DFlash-B8 n=7 | 7.92 | 0.204 | 2.40 |

Only width 2 (`n_max=1`) beats plain, by ~3.5-6%.

### Why: verification cost tracks unique-expert traffic

From `llama-batched-bench -npp 128 -ntg 64 -npl 1,2,4,8`:

```
npl:   1      2      4      8
TG:  13.46  18.06  21.16  28.92   t/s
C_k = k*TG_1/TG_k:  1.00  1.49  2.54  3.72
```

`C_k` is the cost of a width-k decode step relative to width 1. Predicted from the GGUF's
own tensor sizes and the expected number of distinct experts touched,
`192 * (1 - (184/192)^k)`:

| width | predicted | observed |
|---|---|---|
| 2 | 1.55 | 1.49 |
| 4 | 2.59 | 2.54 |
| 8 | 4.41 | 3.72 |

Weight traffic per token: 5.001 GB always-read trunk + 6.766 GB for 8 selected experts
= 11.768 GB. At the measured 73.1 ms/token that is **161 GB/s effective, 62.9% of the
256 GB/s peak** - normal for irregular Q4_K MoE GEMV on gfx1151.

**Speculation wins only when mean accepted length > C_width.** Upper bounds:

```
width 2:  1.60/1.49 = 1.074x     <- the ~4% we actually see
width 3:  1.96/1.91 = 1.026x
width 4:  2.11/2.54 = 0.83x
width 8:  2.40/3.72 = 0.65x
```

### The RPC link is NOT the bottleneck

Tempting mis-inference: theoretical 46 ms/token vs measured 73 ms leaves "27 ms
unaccounted", which looks like RPC overhead. It is not - it is the gap between peak and
achievable DRAM bandwidth (above). Two further points:

- The constant in the affine fit `C_k = 0.39k + 0.74` is the **5.0 GB always-read trunk**,
  not a fixed RPC latency.
- A genuinely fixed per-step RPC cost would *help* speculation, since it is paid once per
  verification batch. Removing it pushes `C_k` toward `k`, making speculation worse.

The two nodes are there for **capacity** (167.68 GB vs 103.1 GB per node), and the
layer-split costs far less than the expert traffic. `-sm tensor` measured 6-7x slower.

---

## The four fixes

### 1. `-md` silently loaded the target instead of the draft

`common/speculative.cpp`, `has_draft` branch. It computed `model_path` from the draft
params, logged it, then called `llama_model_load_from_file(params.model.path, ...)` -
the **target**. So `-md <draft>` loaded the 167 GB target a second time while logging as
though it had loaded the draft. Affects every model, not just hy_v3.

### 2. hy_v3 never exported layer inputs

`src/models/hy-v3.cpp`. DFlash / EAGLE3 / DSpark extract hidden states from selected
target layers via `llama_set_embeddings_layer_inp()`, which reads `res->t_layer_inp[il]`.
Eight archs assign it (qwen3, qwen3moe, qwen35, qwen3next, llama, gemma4, openai-moe,
minimax-m2); hy_v3 did not, so extraction captured nothing.

One line, first in the layer loop:

```cpp
res->t_layer_inp[il] = inpL;
```

Layer 0's `inpL` is the post-embedding residual; later ones are the previous block's
output. Norms/RoPE/routing happen after, so this is the layer INPUT, matching qwen3's
capture point. Only trunk layers - `n_layer()` excludes the appended MTP block.

**Free when unused**: the tensor is only marked a graph output when the corresponding
enable flag is set (`llama-graph.cpp`), and extraction skips disabled layers before
touching it (`llama-context.cpp`). Safe to keep permanently.

**RPC is safe here**: extraction falls back to synchronous `ggml_backend_tensor_get` ->
`RPC_CMD_GET_TENSOR` and blocks. The failure mode for remote layers is latency, not
garbage or zeros.

### 3. `--spec-draft-device` was a no-op on the server path

`arg.cpp:4016` parses it into `params.speculative.draft.devices`, but only
`examples/speculative/speculative.cpp:83` ever applied it. `llama-server` built the
draft's model params with `common_model_params_to_llama(params)`, which copies the
**target's** `devices` and `tensor_split`.

Fixed by overriding both when the flag is given. Note `parse_device_list` nullptr-
terminates, so `.data()` is safe.

**Measured effect on the MTP draft: none** - that draft has exactly one block (`blk.80`),
and a one-layer model cannot be split by `--tensor-split` anyway; it was already local.
The fix matters for multi-layer drafts (DFlash's 5 layers *would* have been split).

### 4. DFlash conversion died on `embed_tokens`

`conversion/qwen.py`, `DFlashModel.filter_tensors`. It prepends `model.` to every tensor
name, but `MODEL_ARCH.DFLASH` has no `TOKEN_EMBD` entry - DFlash shares the target's
embedding and output head. AngelSlim's checkpoint ships `embed_tokens.weight`, so
conversion aborted with `Can not map tensor 'model.embed_tokens.weight'`. Every other
tensor maps cleanly (`norm`->`output_norm`, `fc`->`fc`, `hidden_norm`->`enc.output_norm`).

---

## Building an MTP draft GGUF for Hy3

mradermacher's Q4_K_S quant declares `hy_v3.nextn_predict_layers = 1` in metadata but
ships **zero** nextn tensors (blocks 0..79, no blk.80). `trohrbaugh/Hy3-heretic` - the
abliterated source - has no MTP block either; the abliteration dropped it. Only base
`tencent/Hy3` (layers 0..80) and derived quants carry it.

**Check the tensor list, not the metadata key.**

`prometheusAIR/Hy3-heretic-MTP-GGUF` grafts a base-Hy3 blk.80 onto a heretic quant. Its
blk.80 is contiguous at EOF, so only 2.11 GB of the 81.36 GB file needs fetching via an
HTTP range request; tensors can then be copied verbatim (no requantisation).

Two things the draft MUST contain beyond blk.80, both learned the hard way:

- **`output.weight`** - Hy3 has an UNTIED LM head (`tie_word_embeddings=false`, separate
  Q6_K output). `hy-v3.cpp` marks `output` TENSOR_NOT_REQUIRED and **silently aliases
  `token_embd`** when absent, so the draft computed logits with the embedding matrix:
  `draft acceptance = 0.00000 (0 accepted / 441 generated)`. Not "poor" - exactly zero.
- **`token_embd.weight` + `output_norm.weight`** - created with a literal `0` (REQUIRED)
  even on the `mtp_only` path.

The draft's KV/tokenizer must match the target exactly; copy the target's KV pairs.
Embedding precision does **not** matter measurably: Q4_K vs Q8_0 `token_embd` gave
14.03 vs 14.16 t/s and acceptance 0.620 vs 0.602 - inside noise.

---

## Serving flags that are not optional

```
--no-spec-draft-backend-sampling
```
Backend top-k runs over the 120832-token vocab -> `next_power_of_2` = 131072 * 4 B =
512 KB shared memory, 8x gfx1151's 64 KB limit -> `GGML_ASSERT` abort in
`argsort_f32_i32_cuda_bitonic`. Any MTP model with vocab > 16384 hits this.

```
--spec-draft-n-max 1
```
Anything wider loses (see the table). For DFlash, B8 means anchor + 7 drafts, so
`n_max=7` is width 8 - and `n_max` also shrinks the drafter's noise block, not just the
verification width.

`draft-mtp` and `draft-dflash` cannot be combined - both share `ctx_dft`, and selecting
MTP forces `LLAMA_CONTEXT_TYPE_MTP`.

---

## Tensor split: 48/52, not 50/50

`--tensor-split` balances **layer weights only**. The head additionally holds:

```
token_embd.weight          0.28 GB
output.weight + norm       0.41 GB
MTP draft model            2.80 GB
logits buffer (-ub 2048)   0.99 GB   (2048 x 120832 x f32)
                          ------
                           4.48 GB accounted of a 7.5 GB measured delta
                                    (rest: activations, compute buffers, desktop)
```

Measured, KV preallocated at `-c 131072`:

```
50/50   head 102.6 GB (99.5%)   peer 95.2 GB (92.3%)   14.16 t/s
48/52   head  99.8 GB (96.8%)   peer 97.5 GB (94.6%)   14.11 t/s (median)
46/54   head  95.9 GB (93.0%)   peer 101.9 GB (98.8%)  <- peer becomes the constraint
```

48/52 frees 2.8 GB on the head for no measurable throughput change. Speed difference is
inside run-to-run variance; take it for the headroom.

---

## Things that did NOT work, so you do not retry them

- **`--spec-draft-p-min`** - raising it *reduces* throughput. It hits the 80-90% acceptance
  figure quoted for MTP elsewhere, and is slower doing it: p_min 0.00 -> 14.16 t/s
  (acc 0.602, 56 tokens gained); 0.50 -> 12.96 (acc 0.797, 51 gained); 0.75 -> 12.55
  (acc 0.875, 49 gained). **Acceptance rate is the wrong metric; total accepted tokens is
  the right one.** Verifying 2 tokens costs far less than 2x verifying 1, so declining
  drafts saves little and forfeits real tokens.
- **`AngelSlim/Hy3-MTP-TTT3`** - a BF16 MTP head trained as a drafter. Untested fairly:
  the run that scored 5.51 t/s predated the `output.weight` fix. Rebuild before judging.
- **`AngelSlim/Hy3-DFly-Block8`** - `Qwen3DFlyModel` is not registered in llama.cpp's
  converter, and it ships a pytorch pickle. Not convertible.
- **Higher-precision embeddings** in the draft - no measurable effect (above).
