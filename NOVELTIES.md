# Novelties and plans

Ideas, plans and findings that are **not implemented or not measured**. What
is implemented and measured is in [README.md](README.md). Nothing here is a
result.

## Milestone 2: a real model through tiermem

The workload is [ARC 1.0](https://huggingface.co/rayline-ai/ARC-1.0)
(`rayline-ai/ARC-1.0`, Apache-2.0, release `arc-1.0-20261111`), a decision
model on a Qwen3.5-0.8B backbone.

### 2a. Feasibility report (2026-11-12)

No code was written for this step. The safetensors headers were read with
HTTP range requests (the 2.1 GiB of weights were not downloaded), along with
`config.json`, `arc_release.json`, `provenance.json`, the question head's
`manifest.json`, both golden files and the model card.

**Verdict: the backbone cannot be run from Rust on tiermem-managed memory
without large custom work. Milestone 2 stops here for review.** The reasons
are in "Can Rust run the backbone" below, and the options are at the end.

#### The files

| File | Size | Tensors | Element types |
|---|---|---|---|
| `model.safetensors` | 1232.7 MiB | 421 | BF16, F32, F64, I64, U8 |
| `question_head/qh4in-full-e3/upper.safetensors` | 950.1 MiB | 159 | F32 |
| `question_head/qh4in-full-e3/pointer.safetensors` | 2.0 MiB | 3 | F32 |
| Total | 2184.8 MiB | 583 | |

`model.safetensors` by element type:

| Type | Tensors | Elements | Size |
|---|---|---|---|
| BF16 | 296 | 603,928,464 | 1151.90 MiB |
| F64 | 37 | 9,991,386 | 76.23 MiB |
| F32 | 73 | 1,069,432 | 4.08 MiB |
| I64 | 10 | 63,509 | 0.48 MiB |
| U8 | 5 | 1,960 | 1,960 bytes |

How the tensors group:

| Group | Tensors | Size | Type | Used by |
|---|---|---|---|---|
| `model.language_model.embed_tokens.weight` `[248320, 1024]` | 1 | 485.00 MiB | BF16 | every request |
| `model.language_model.layers.{0,1,2,4,5,6,8,9,10}` (linear attention) | 14 each | 41.11 MiB each | BF16, two small F32 | every request |
| `model.language_model.layers.{3,7,11}` (full attention) | 11 each | 35.00 MiB each | BF16 | every request |
| `model.language_model.norm.weight` `[1024]` | 1 | 2 KiB | BF16 | every request |
| `model.visual.*` (12 ViT blocks, merger, patch and position embeddings) | 153 | 191.87 MiB | BF16 | requests with images |
| `signals.pii.payload`, `signals.topic.domain`, `signals.topic.operation` | 7 each | about 1 MiB each | F32, F64 | specialized heads |
| `stage_one.*` (model success and token usage) | 44 | about 76 MiB | F64, I64, U8 | routing |
| `stage_two.*` (arm selection) | 42 | about 1.5 MiB | F32, I64 | routing |
| `upper.safetensors` `layers.{12..23}`, same shapes as 0..11 | 14 or 11 each | 82.23 or 70.01 MiB each | F32 | question head |
| `pointer.safetensors` `q.weight`, `k.weight` `[256, 1024]`, `log_scale` `[]` | 3 | 2.0 MiB | F32 | question head |

The lower twelve layers total 475.0 MiB and the upper twelve 950.1 MiB: the
upper layers have the same shapes but are stored as F32, so they are twice
the bytes. A text request that only needs the specialized heads touches
about 1040 MiB (embedding, lower layers, heads). The question head adds the
upper layers, about 1990 MiB in all. The vision tower is not touched by text.

Things in these files that the manager does not handle today:

- **Five element types under one budget.** `Manager<T>` holds one type.
- **Two tensors with no elements** (`stage_one.ens_draws`,
  `stage_one.ens_owner`) and **one of rank 0** (`log_scale`). Reloading an
  empty tensor is an error in `copy.rs`.
- **One tensor is 39% of the main file.** The embedding table is 485 MiB and
  a request reads a few hundred of its 248,320 rows. tiermem moves whole
  tensors.
- **583 tensors, most of them tiny.** A layer is 11 or 14 tensors that are
  always used together, from a `[16]` vector to a `[6144, 1024]` matrix.
- **The weights start in a file, not on the device.** `register` takes a
  device tensor, so loading under a budget smaller than the model would push
  every tensor through the device once just to spill it.

#### The architecture

From `config.json` (`model_type: qwen3_5`, architecture
`ArcImagePolicyForPooling`):

- Hidden size 1024, SwiGLU MLP 1024 → 3584 → 1024, RMSNorm before each
  block, no biases in the language model.
- `layer_types` repeats three `linear_attention` layers and one
  `full_attention` layer. ARC ships layers 0 to 11 of the base unchanged and
  layers 12 to 23 fine-tuned.
- **Linear attention** is a gated-delta-rule recurrent layer: one projection
  to q, k and v (`in_proj_qkv` `[6144, 1024]`, 16 heads of 128), a causal
  depthwise convolution of width 4 over them, per-head decay and write gates
  (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`), an output gate
  (`in_proj_z`), a per-head norm and an output projection. Its state is
  carried from token to token, so it is a sequential scan, not a matrix
  product.
- **Full attention** is grouped-query attention with 8 heads and 2 key/value
  heads of dimension 256, RMSNorm on q and k, and an output gate (the
  `q_proj` is `[4096, 1024]`, twice the 2048 the heads need).
- Positions are multimodal RoPE on a quarter of each head (sections 11, 11,
  10, theta 1e7) with YaRN scaling by 4. ARC names this `yarn-f4` and refuses
  to load against a different RoPE.
- The specialized heads read `norm(h_12)` at the last token of each role
  block. Each is a standardization (`mean`, `scale`), two linear layers
  (1024 → 256 → classes) and a Platt map. The activation between the two
  layers is not named anywhere in the release.
- The question head runs all 24 layers and scores options with a bilinear
  pointer head.

#### Can Rust run the backbone

- **Candle: no.** `candle-transformers` on `main` (commit `cfcbd718`,
  2026-11-01) has `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_vl`, `mamba` and
  `mamba2`. A code search of the repository finds nothing for `qwen3_5`,
  `linear_attn`, `gated_delta` or `in_proj_qkv`. Issue #3393 "Qwen3.5
  Support" is open, and so is PR #3461, which adds the architecture for
  quantized GGUF checkpoints only.
- **mistral.rs: probably, on its own tensors.** A code search finds
  `qwen3_5` (30 hits), `linear_attn` and `gated_delta` in
  `EricLBuehler/mistral.rs`. I did not build or run it.
- **Neither computes on tiermem's memory, and that is the real obstacle.**
  tiermem manages `cutile::Tensor<T>`. Candle and mistral.rs own their device
  memory through their own CUDA layer. Running their forward pass "through
  `ensure_resident`" would mean either copying each layer from the tiermem
  tensor into a Candle tensor on every access, which puts a second copy of
  the layer on the device outside the budget, or building Candle tensors
  over device pointers that tiermem owns. I found no supported way to do the
  second, and did not look for an unsupported one (ground rule 6).
- **`cutile` 0.4 has the element types but not the operators.** `bf16`,
  `f32`, `f64`, `i64` and `u8` are all kernel element types. The crate ships
  kernels only for creating and converting tensors (`full`, `arange`,
  `linspace`, `eye`, `convert`). There is no linear layer, norm, softmax,
  convolution or gather. The DSL has the primitives to write them (`mma`,
  `reduce_sum`, `exp`, `rsqrt`, `select`, `permute`, `scan_sum`).

So a forward pass on managed tensors means writing every operator as a tile
kernel: linear layers over BF16 weights, RMSNorm, SiLU and sigmoid, softmax
attention with grouped heads, q/k norm and the output gate, RoPE with the
YaRN scaling, the causal convolution, the gated-delta recurrence, the
embedding gather, the last-token readout, and the heads. That is the large
custom work the plan says to stop for.

#### What there is to check against

- **`goldens.jsonl`** (54 synthetic sessions) expects routing decisions:
  scores for 49 actions and the selected arm. Reproducing them needs stage
  one, stage two, the policy and the price card, which is the routing policy
  this milestone leaves out.
- **`goldens.signals.jsonl`** (54 rows) expects the three signal heads'
  probabilities to six decimals, read from layer 12. In principle that checks
  layers 0 to 11 and a head end to end. In practice the input rendering is
  not published: the encoding profile names it
  (`pathfinder.input-projection.v1`, serializer `arc-role-blocks-v2`,
  `canonical_v2`, `harness_shell: include`, `scrub_v2`) without defining it.
  "What is two plus two?" is 23 tokens in the golden, which looks like more
  than a bare `<|im_start|>user ... <|im_end|>` block needs (I did not
  tokenize it to confirm), so something is probably added. The
  public `vllm-arc` plugin has the model classes and the stage-one head only;
  its README says signals, the question head and serving are not included.
- **A Python reference is the workable check.** `transformers` has a
  `qwen3_5` model. Feeding it the same token ids, with ARC's RoPE setting,
  and saving the hidden state after every layer gives a layer-by-layer
  reference that does not depend on the unpublished rendering. Nothing is
  installed here yet (no `torch`, `transformers` or `safetensors`; `uv` is
  available, with 834 GiB of disk and 31 GiB of RAM free).
- **Bit parity is not a realistic target.** ARC's own parity is against one
  pinned vLLM build with fused kernels turned off. An independent BF16
  implementation will need a tolerance, to be chosen from the reference
  run's own BF16-against-F32 difference.

#### How this workload differs from the synthetic benchmark

Read off the files, not measured:

- Sizes are uneven by a factor of 14 between the embedding table and a layer
  (485 MiB against 35 to 82 MiB), and by far more against the heads.
- At a quarter of the question-head working set (about 500 MiB) the
  embedding table alone nearly fills the budget, so it would be evicted
  before layer 0 can load, on every request. At about 27 GB/s that is roughly
  19 ms each way for 485 MiB, to read a few hundred rows.
- The lower twelve layers are shared by every request and the upper twelve
  only by question requests, so the access order branches (see "Branching
  access plans" below).
- Per-layer compute is not constant: the linear-attention layers scan the
  sequence and the full-attention layers are quadratic in it.

#### Options for review

1. **Write the Qwen3.5 text forward as tile kernels**, checked layer by layer
   against a `transformers` reference. Faithful to the milestone, and the
   largest piece of work: about a dozen kernels, the recurrence being the
   hard one.
2. **Do 2b now and defer 2c.** Multi-type support, registering from host
   memory and residency hints are needed by every option and do not depend on
   the backbone. The real size distribution could then be benchmarked with
   the synthetic kernel standing in for compute, which answers how uneven
   sizes change the numbers but not how real compute does.
3. **Compute with mistral.rs and copy each layer device to device** from the
   tiermem tensor. Quick, but the compute copy sits outside the budget, so it
   measures tiermem's traffic and not a model running in less memory than it
   needs.
4. **A smaller backbone that Candle already runs** (it has Qwen3), keeping
   ARC for later. Loses the shared-trunk structure that makes ARC
   interesting.

My recommendation is 2 and then 1, starting with the MLP and full-attention
layers, which are ordinary, and leaving the gated-delta layer for last.

### 2b to 2d

Not started. Waiting on the review above.

## Ideas

Each of these is unmeasured. Some are only a sentence.

### Branching access plans for ARC's shared trunk

After Milestone 2. `expect_accesses` declares one linear order. An ARC
request runs the shared lower twelve layers and then either stops at the
specialized heads or continues through the upper twelve for the question
head. A plan that can say "the trunk, then one of these branches, with these
odds" would let eviction and prefetch weigh a tensor's next use across
branches, instead of treating the upper layers as either certainly next or
not declared at all. (This is my reading of the idea from the task list;
correct it if it meant something else.)

### Memory-aware model routing

Milestone 3, after Milestone 2. ARC chooses which model should handle a
request from predicted success and cost. What is resident, and what it would
cost to make a model resident, is a cost the router does not see. Feeding
tiermem's state into the routing decision, for example preferring a local
model whose weights are already on the device, would make the two layers
cooperate. (Also my reading of a one-line item.)

### Smaller ideas that came out of this round of work

- **Row-granular residency for embedding tables.** Keep only the rows a
  request uses on the device. ARC's 485 MiB table is the case for it.
- **Registering a tensor from host memory**, already spilled, so loading a
  model never pushes weights through the device just to evict them.
- **Cost-aware eviction for accesses.** Automatic prefetch already prefers a
  victim that is free to evict. An access still evicts the tensor needed
  farthest away even when that means waiting for a copy and a slightly nearer
  one is free.
- **Chaining a reload behind a write-back** with a stream event, so the pair
  runs without the manager being called in between.
- **A call to give unused page-locked blocks back** to the system.
- **Trying the two copy streams on a device with more than one async
  engine**, where a reload and a write-back might really overlap.
