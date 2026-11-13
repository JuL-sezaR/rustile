# Design: four tile kernels and the manager changes they need

Written 2026-11-13 (design phase). This is a spec to implement from, one step
at a time. It is not a result: nothing here is measured, and nothing from it
goes into the README until the step that implements it has numbers.

Rules for whoever implements a step:

- Implement exactly one step from "Steps" below, in order. Do not redesign.
- If something here is wrong, unclear or does not compile as written, stop,
  write the question under "Open questions" at the end, and hand back.
- The project ground rules apply: medians of 5 runs after a warmup, no claim
  inside ±10%, the README only gets what is implemented and measured, small
  commits, a test for every behavior change, and if `cutile` 0.4 cannot
  express something, write down what is missing instead of working around it.
- Do not touch `NOVELTIES.md`. Do not pick a model for Milestone 2. The
  shapes used in tests and benchmarks below are arbitrary sizes, not a
  model's.
- Run only the tests of the step while working, and the full suite
  (`cargo test --workspace`) once before committing.

## State on 2026-11-13

- `tiermem-arena-streams` is merged into `main` (PR #9, merge commit
  `6ce0ea2`); the working tree was clean and no branch is unmerged.
- Of the three model-independent parts of Milestone 2 step b (issue #4):
  - **Element types beyond `f32`: not done.** `Manager<T: DType>` holds one
    element type, so BF16, F32, F64, I64 and U8 tensors cannot share one
    budget. The host pool already hands out bytes, so only the manager needs
    to change. Designed in step 6.
  - **Registering from host memory: not done.** `register` takes a device
    tensor. Designed in step 7.
  - **Residency hints: done.** `Hint::{Pinned, Hot, Evictable}` with eviction
    ranks, tested by `evictable_goes_before_hot_and_pinned_never_goes` and
    `fails_when_only_pinned_tensors_remain`. One gap, left alone because
    nothing below needs it: a hint is fixed at registration.
- No kernel other than the benchmark's vector add exists in the repo.

## What cutile 0.4 can and cannot express

Checked on the development machine (RTX 5080 Laptop, CUDA 13.3, `tileiras`
bytecode versions 13.1 to 13.3) with a throwaway probe that compiled and ran
every kernel body in this document and compared it with a CPU loop. The probe
is not in the repository. Its results are quoted where they decide something;
they are single runs and are not benchmark numbers.

Facts the design rests on:

1. **Every tile dimension must be a positive power of two**, and a tile holds
   at most 16,777,216 elements (`cutile-ir` `verify_target.rs`).
2. **Reads past the end of a tensor are zero.** `x.partition(shape)`,
   `x.load_like(y)` and `x.load_tile(..)` build their view with
   `padding::Zero`.
3. **Stores past the end of the output are dropped.** A host-side
   `(&mut y).partition([bm, bn])` accepts a shape that is not a multiple of
   the tile, launches `ceil(M/bm) × ceil(N/bn)` blocks, and the last blocks
   write only the part that exists (cutile's own `slice_non_divisible` tests,
   and the probe).
4. **`mma` takes BF16, F32 or F64 operands.** BF16 and F32 operands with an
   F32 accumulator, and F64 with F64, all ran and matched the reference.
5. **`GatherScatterView` works on this toolchain** (it needs Tile IR 13.3).
   Its dense index is an element offset, not a tile index. With
   `padding::Zero`, an index outside the table (negative, or past the last
   row) reads zeros.
6. **Float conversion inside a kernel works** (`convert_tile` from BF16 to
   F32). A kernel generic over `E: ElementType` is instantiated by passing
   the type's name first in `.generics(..)`.
7. **`.sync()` returns after the kernel has finished** (it synchronizes the
   stream), so timing a `.sync()` call times the kernel.
8. First-call (JIT) time grows with the tile's element count: about 0.3 s for
   tiles up to 4096 elements, 0.9 s at 32,768 and 20 s at 262,144 (softmax
   probe, one run each).

What is missing. Per the ground rules these are written down, not worked
around:

- **I64 token ids cannot be narrowed inside a kernel.** The gather view takes
  a `Tile<i32>` index. `convert_tile::<i32, i64>` fails with
  `Conversion "i64" -> "i32" not yet implemented`, and
  `trunci(x, overflow::None)` fails with `missing attribute 'overflow' on op
  TruncI`. `trunci` with `overflow::NoSignedWrap` was not tried: it promises
  the value fits, which a kernel cannot promise for caller data.
  **Decision: `gather` takes `Tensor<i32>` ids only.**
- **A scalar broadcast to a shape written with a const generic does not
  type-check** in a generic kernel (`df.broadcast(shape![BM])` gives
  `Tile<f32, {[2]}>` against `Tile<f32, {[BM]}>`). Broadcasting to
  `y.shape()` works, which is what the kernels below do.
- **No reduction across tiles in one launch.** A block can only store its own
  tile, so a row that is reduced (softmax, RMSNorm) has to fit in one tile.
  See "Tile-shape rules" for the consequence.
- BF16 output is not designed: kernels that compute write F32 (or F64 for F64
  input). Converting F32 to BF16 on store was not probed.

`make_gather_scatter_view` and `load_gather_scatter_view_tko` are `unsafe`
functions of cutile's DSL. Using them is the documented way to gather in
cutile 0.4 (its own tests do), not a workaround. The safe host wrapper is
sound because both tensors are borrowed for the whole call, the call waits
for the kernel, and out-of-range lanes are padded instead of read.

## Layout

A new workspace crate, `tilekernels`, which depends only on `cutile`. The
kernels take `&Tensor<T>` and know nothing about tiermem; step 8 shows them
running on managed tensors.

```
tilekernels/
  Cargo.toml            name = "tilekernels", cutile.workspace = true,
                        edition/license from the workspace
  src/lib.rs            public API, KernelError, the Float trait
  src/tiles.rs          tile-shape rules (pure functions)
  src/reference.rs      CPU references (pure functions)
  src/kernels.rs        the #[cutile::module] with the DSL entries
  tests/common/mod.rs   shared test helpers
  tests/softmax.rs, rmsnorm.rs, gather.rs, linear.rs
  examples/kernelbench.rs
```

Add `"tilekernels"` to `members` in the workspace `Cargo.toml`. BF16 is
`cutile::half::bf16`; do not add a `half` dependency.

### Common API

```rust
#[derive(Debug)]
pub enum KernelError {
    /// The arguments do not fit together. Nothing was launched.
    Shape(String),
    /// A cutile or CUDA failure, including a kernel that does not compile.
    Device(cutile::error::Error),
}
// Display, std::error::Error, From<cutile::error::Error>,
// From<DeviceError> (through cutile::error::Error, as tiermem's `dev` does).

/// Element types the computing kernels accept, with the type they write.
/// Sealed: implemented for bf16 (Out = f32), f32 (Out = f32), f64 (Out = f64).
pub trait Float: DType + sealed::Sealed {
    type Out: DType;
}
```

Every kernel has two entry points: `name(..) -> Result<Tensor<Out>, _>`, which
allocates the output with `api::zeros` and calls the second, and
`name_into(y, ..) -> Result<(), _>`, which writes into a tensor the caller
owns. `_into` overwrites every element of `y` and does not read it. The
benchmark uses `_into` so that allocation is not timed.

All tensors are 2-D (`[rows, columns]`) except vectors named below. A caller
with `[batch, seq, d]` reshapes to `[batch * seq, d]` first. Every wrapper
checks ranks and sizes before launching and returns `KernelError::Shape` with
a message naming the argument. A dimension of 0 anywhere is a `Shape` error.

Launch pattern (verified):

```rust
kernels::softmax_f32((&mut *y).partition([bm, bn]), x)
    .generics(vec![E::DTYPE.as_str().to_string(), bm.to_string(), bn.to_string()])
    .sync()
    .map(|_| ())?;
```

The kernels for F64 are separate, non-generic entries (`*_f64`), because the
accumulator type and the constants differ. How a wrapper picks the entry for
`E` is up to the implementer (a private method on the sealed trait is the
obvious way).

### Tile-shape rules (`tiles.rs`)

`pow2(n)` is `n.next_power_of_two()`.

| Function | Result | Used by |
|---|---|---|
| `row_tiles(rows, cols) -> (bm, bn)` | `bn = pow2(cols)`; `bm = min(pow2(rows), clamp(4096 / bn, 1, 16))` | softmax, RMSNorm |
| `gather_tiles(n_ids, d) -> (bm, bd)` | `bd = min(pow2(d), 256)`; `bm = min(pow2(n_ids), 16)` | gather |
| `LinearTiles::for_shape(m, k, n)` | `bm = clamp(pow2(m), 16, 64)`, `bn = clamp(pow2(n), 16, 64)`, `bk = clamp(pow2(k), 16, 32)` | linear |

Each returns `Err(KernelError::Shape)` if a tile would exceed 16,777,216
elements or a dimension does not fit in `i32`.

Why these shapes:

- Powers of two are required (fact 1). Sizes that are not tile multiples need
  no special code path: reads are zero-padded and stores are clipped (facts 2
  and 3). Each kernel section says why zero padding is harmless there, or how
  the padding is masked.
- **Row kernels** (softmax, RMSNorm) reduce over a whole row, and a block can
  only write its own tile. Splitting a row across blocks would make every
  block recompute the row's sum, so the tile spans the row: `bn = pow2(cols)`.
  `bm` rows share a block as long as the tile stays at 4096 elements or
  fewer, which is where the probe's first-call time was still about 0.3 s.
- **Gather** only copies, so its columns can be split: 256 columns by up to
  16 ids, again 4096 elements.
- **Linear** tiles are at least 16 on every axis, the smallest size the probe
  ran `mma` with. The output tile is at most 64 × 64 so a short input (a few
  rows) is not padded to a large tile of zero rows, and `bk` is at most 32 so
  the loop body holds two 64 × 32 operands. The probe ran 16 × 16 × 16 and
  32 × 64 × 32. **These defaults are a starting point. Step 5's benchmark
  sweeps the tile shape, and the default changes only if another shape is
  faster by more than the ±10% variance.**

### CPU references (`reference.rs`)

Plain single-threaded loops over slices, accumulating in `f64`, row-major.
They are the oracle for the tests and the naive baseline for the benchmark.
The caller converts stored values to `f64` first (`bf16::to_f64`,
`f32 as f64`), so the reference sees exactly the values the device sees.

```rust
/// Row-wise softmax of a `rows × cols` matrix.
pub fn softmax(x: &[f64], rows: usize, cols: usize) -> Vec<f64>;
/// y[i][j] = x[i][j] / sqrt(mean_j(x[i][j]^2) + eps) * gain[j]
pub fn rmsnorm(x: &[f64], gain: &[f64], rows: usize, cols: usize, eps: f64) -> Vec<f64>;
/// out[i] = table[ids[i]], or a row of `zero` when ids[i] is negative or
/// >= the number of table rows (`table.len() / d`).
pub fn gather<T: Copy>(table: &[T], ids: &[i32], d: usize, zero: T) -> Vec<T>;
/// y[i][j] = sum_p x[i][p] * w[j][p] + bias[j]; `w` is `n × k`.
/// Also returns scale[i][j] = sum_p |x[i][p] * w[j][p]| + |bias[j]|.
pub fn linear(x: &[f64], w: &[f64], bias: Option<&[f64]>, m: usize, k: usize, n: usize)
    -> (Vec<f64>, Vec<f64>);
```

Softmax subtracts the row maximum before `exp`, like the kernel.

### Test helpers (`tests/common/mod.rs`)

```rust
/// Deterministic values in [-1, 1): xorshift64 seeded with
/// seed.wrapping_mul(0x9E3779B97F4A7C15) | 1, steps `s ^= s << 13;
/// s ^= s >> 7; s ^= s << 17`, value ((s >> 40) as f32 / 16777216.0) * 2 - 1.
pub fn rnd(n: usize, seed: u64) -> Vec<f32>;
/// Uploads a host vector as a tensor of `shape`.
pub fn dev<T: DType>(v: Vec<T>, shape: &[usize]) -> Tensor<T>;   // copy_host_vec_to_device + reshape
/// Reads a tensor back.
pub fn host<T: DType>(t: &Tensor<T>) -> Vec<T>;                  // t.dup().to_host_vec().sync()
```

Tests run on the GPU, like `tiermem/tests/manager.rs`. Every numeric test
prints the largest error it saw, so a tolerance that is far too loose or
nearly violated is visible in the test output.

### Tolerances

`got` is the device result converted to `f64`; `want` is the reference.

| Kernel | Output F32 (input BF16 or F32) | Output F64 |
|---|---|---|
| softmax | `abs(got - want) <= 1e-6`, and each row sums to 1 within `1e-5` | `1e-14`, row sum within `1e-13` |
| RMSNorm | `abs(got - want) <= 1e-6 + 1e-5 * abs(want)` | `1e-12 + 1e-12 * abs(want)` |
| gather | bit-exact (`==`), all five element types | bit-exact |
| linear | `abs(got - want) <= 16 * 2^-24 * scale + 1e-30` | `16 * 2^-53 * scale + 1e-300` |

BF16 has no tolerance of its own. BF16 inputs are converted exactly to `f64`
for the reference, the kernel computes in F32, and a product of two BF16
values is exact in F32, so the F32 column applies. What BF16 loses is in the
inputs, before either side computes.

The probe's largest errors, for orientation: softmax `1.5e-8` (F32),
`1.6e-9` (BF16 in), `1.4e-17` (F64); RMSNorm `1.2e-7` (F32 and BF16 in),
`1.1e-14` (F64); linear `1.9e-6` with a scale of about 25 (F32), `9.5e-7`
(BF16 in), 0 (F64). If a tolerance in the table fails on correct-looking
output, that is an open question, not a number to loosen.

## The kernels

The bodies below compiled and ran in the probe. Copy them; rename freely.
All sit in one `#[cutile::module] mod kernels { use cutile::core::*; use
cutile::tileir::*; .. }`.

### Softmax

`softmax<E: Float>(x: &Tensor<E>) -> Tensor<E::Out>`, along the columns of a
`[rows, cols]` tensor. Tiles from `row_tiles(rows, cols)`.

The padded columns of the tile read as 0, and `exp(0 - max)` would be added
to the sum, so they are replaced by `-inf` first: `iota` gives the column
index, `lt_tile` against the real column count gives a mask, `select` applies
it. Padded rows compute garbage that the clipped store drops.

```rust
#[cutile::entry()]
fn softmax_f32<E: ElementType, const BM: i32, const BN: i32>(
    y: &mut Tensor<f32, { [BM, BN] }>,
    x: &Tensor<E, { [-1, -1] }>,
) {
    let tx: Tile<E, { [BM, BN] }> = x.load_like(y);
    let tx: Tile<f32, { [BM, BN] }> = convert_tile(tx);
    let n: i32 = x.shape()[1];
    let idx: Tile<i32, { [BN] }> = iota(shape![BN]);
    let nn: Tile<i32, { [BN] }> = n.broadcast(shape![BN]);
    let mask: Tile<bool, { [BN] }> = lt_tile(idx, nn);
    let mask: Tile<bool, { [BM, BN] }> = mask.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
    let ninf: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);
    let tx: Tile<f32, { [BM, BN] }> = select(mask, tx, ninf);
    let mx: Tile<f32, { [BM] }> = reduce_max(tx, 1i32);
    let mx: Tile<f32, { [BM, BN] }> = mx.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
    let num: Tile<f32, { [BM, BN] }> = exp(tx - mx);
    let den: Tile<f32, { [BM] }> = reduce_sum(num, 1i32);
    let den: Tile<f32, { [BM, BN] }> = den.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
    y.store(num / den);
}
```

`softmax_f64` is the same with `f64` for `E` and `f32`, no `convert_tile`,
and `f64::NEG_INFINITY`. Generics: `[E, BM, BN]` and `[BM, BN]`. The probe
ran the generic entry with BF16 and the F64 entry; the F32 case ran as a
non-generic entry without the `convert_tile` line. `convert_tile` from F32 to
F32 does compile: the generic RMSNorm entry below ran with `E = f32`.

Element types: BF16 and F32 (output F32), F64 (output F64). I64 and U8 are
not accepted: `Float` is not implemented for them.

Tests (`tests/softmax.rs`), each against `reference::softmax` with the table's
tolerance:

- F32, sizes `(5, 1000)`, `(4, 64)` (exact tile), `(3, 65)` (one over),
  `(3, 63)` (one under), `(1, 1)` (result exactly 1.0), `(17, 7)` (more rows
  than `bm`), `(3, 4096)`. Inputs `rnd * 8`.
- BF16 input and F64, size `(5, 1000)`.
- Large values: a row holding `1e4`, `1e4 - 1`, `-1e4` gives finite results
  matching the reference (no overflow).
- `-inf` entries: a row with some `-inf` gives exactly 0 there and still sums
  to 1.
- Shape errors: a 1-D input, a 3-D input, and an `_into` output of the wrong
  shape each return `KernelError::Shape`.
- One `#[ignore]`d test, `wide_row`, size `(3, 248_320)`: run it once
  (`cargo test -p tilekernels --test softmax -- --ignored --nocapture`) and
  report its error and time in the step's hand-back. It is ignored because
  its first call took about 20 s in the probe.

Unspecified and untested: a row that is all `-inf` or contains NaN or `+inf`.

Benchmark baseline: `reference::softmax` on the same data.

### RMSNorm

`rmsnorm<E: Float>(x: &Tensor<E>, gain: &Tensor<E>, eps: f64) ->
Tensor<E::Out>`: `y[i][j] = x[i][j] * rsqrt(mean_j(x[i][j]^2) + eps) *
gain[j]`, with `x` `[rows, d]` and `gain` `[d]`. Tiles from
`row_tiles(rows, d)`.

Zero padding is harmless: padded columns add 0 to the sum of squares, and
the mean divides by the real `d`, read from `x.shape()[1]`. `gain` is loaded
as one zero-padded tile.

```rust
#[cutile::entry()]
fn rmsnorm_f32<E: ElementType, const BM: i32, const BD: i32>(
    y: &mut Tensor<f32, { [BM, BD] }>,
    x: &Tensor<E, { [-1, -1] }>,
    g: &Tensor<E, { [-1] }>,
    eps: f32,
) {
    let tx: Tile<E, { [BM, BD] }> = x.load_like(y);
    let tx: Tile<f32, { [BM, BD] }> = convert_tile(tx);
    let d: i32 = x.shape()[1];
    let df: f32 = convert_scalar(d);
    let ss: Tile<f32, { [BM] }> = reduce_sum(tx * tx, 1i32);
    let ss: Tile<f32, { [BM, BD] }> = ss.reshape(shape![BM, 1]).broadcast(y.shape());
    let dt: Tile<f32, { [BM, BD] }> = df.broadcast(y.shape());
    let et: Tile<f32, { [BM, BD] }> = eps.broadcast(y.shape());
    let inv: Tile<f32, { [BM, BD] }> = rsqrt(ss / dt + et, ftz::Disabled);
    let tg: Tile<E, { [BD] }> = g.partition(shape![BD]).load([0i32]);
    let tg: Tile<f32, { [BD] }> = convert_tile(tg);
    let tg: Tile<f32, { [BM, BD] }> = tg.reshape(shape![1, BD]).broadcast(y.shape());
    y.store(tx * inv * tg);
}
```

`rmsnorm_f64` is the same in `f64` without the two `convert_tile` calls. The
wrapper passes `eps as f32` to the F32 entry. This generic entry ran in the
probe with both F32 and BF16.

Element types: as softmax.

Tests (`tests/rmsnorm.rs`), against `reference::rmsnorm`, `eps = 1e-6`:

- F32, sizes `(7, 1000)`, `(4, 1024)` (exact), `(3, 1025)`, `(3, 1023)`,
  `(1, 1)`, `(33, 16)`. `x` and `gain` from `rnd` with different seeds.
- BF16 input and F64, size `(7, 1000)`.
- A row of zeros gives a row of exact zeros (`0 * rsqrt(eps) * g`).
- A gain of all ones on a row of constant `c` gives `c / sqrt(c^2 + eps)`.
- Shape errors: `gain` of the wrong length, `gain` 2-D, `x` 1-D, wrong
  `_into` shape.

Benchmark baseline: `reference::rmsnorm`.

### Embedding gather

`gather<E: DType>(table: &Tensor<E>, ids: &Tensor<i32>) -> Tensor<E>`:
`out[i] = table[ids[i]]`, with `table` `[v, d]`, `ids` `[n]`, output
`[n, d]`. Tiles from `gather_tiles(n, d)`.

Block `(p0, p1)` loads `bm` ids and copies columns `p1 * bd ..` of those
rows. Ids past the end of `ids` read as 0 (row 0 exists), and the rows they
produce are clipped. Columns past `d` read as zero and are clipped.

```rust
#[cutile::entry()]
unsafe fn gather<E: ElementType, const BM: i32, const BD: i32>(
    y: &mut Tensor<E, { [BM, BD] }>,
    table: &Tensor<E, { [-1, -1] }>,
    ids: &Tensor<i32, { [-1] }>,
) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let tid: Tile<i32, { [BM] }> = ids.partition(shape![BM]).load([pid.0]);
    let view: GatherScatterView<E, { [BM, BD] }, 0> =
        unsafe { make_gather_scatter_view(table, shape![BM, BD], padding::Zero) };
    let (t, _tok): (Tile<E, { [BM, BD] }>, Token) = unsafe {
        load_gather_scatter_view_tko(
            &view, tid, pid.1 * BD, new_token_unordered(),
            ordering::Relaxed, scope::TileBlock,
        )
    };
    y.store(t);
}
```

The host wrapper is a safe function; its `unsafe` block around the launch
carries a `// SAFETY:` comment with the argument from "What cutile 0.4 can
and cannot express".

Element types: any `DType`; tested for BF16, F32, F64, I64 and U8 (all five
ran bit-exact in the probe). Ids: I32 only (see the missing-API list).

**An id outside `0..v` gives a row of zeros**, not an error: the ids are on
the device and checking them would cost a copy back. This is documented on
the function and tested.

Tests (`tests/gather.rs`), against `reference::gather`, bit-exact:

- F32, `v = 50`, `d = 100`, 13 ids `(i * 17 + 3) % v`, table value
  `row * d + col`; again with explicit tiles `(4, 32)`, `(1, 128)` and
  `(16, 16)` through a `gather_into_with_tiles` that the public functions
  call.
- BF16, F64, I64, U8 with the same ids.
- `d` of 256 (exact), 257, 255, 1; `n` of 1, 16, 17.
- Repeated ids, the first row and the last row.
- Ids `-1` and `v` give zero rows; the rows around them are right.
- Shape errors: `ids` 2-D, `table` 1-D, wrong `_into` shape.

Benchmark baseline: `reference::gather`.

### Linear

`linear<E: Float>(x: &Tensor<E>, w: &Tensor<E>, bias: Option<&Tensor<E::Out>>)
-> Tensor<E::Out>`: `y[i][j] = sum_p x[i][p] * w[j][p] + bias[j]`, with `x`
`[m, k]`, `w` `[n, k]`, `bias` `[n]`, output `[m, n]`. `linear_into` takes
the same plus `y` and a `LinearTiles { bm, bn, bk }`; `linear` uses
`LinearTiles::for_shape`.

`w` is `[n, k]`, one row per output, which is how weight files store a linear
layer. The kernel transposes each weight tile as it loads it, so no
transposed copy of the weight is ever made.

The loop over `k` runs `ceil(k / BK)` times. The last `x` tile and the last
`w` tile are both zero-padded along `k`, so the padding adds 0. Padded rows
and columns of the output are clipped. With `bias: None` the wrapper passes a
zero tensor of `[n]` that it allocates for the call (documented; a caller who
minds allocates one zero bias and reuses it).

```rust
#[cutile::entry()]
fn linear_f32<E: ElementType, const BM: i32, const BN: i32, const BK: i32>(
    y: &mut Tensor<f32, { [BM, BN] }>,
    x: &Tensor<E, { [-1, -1] }>,
    w: &Tensor<E, { [-1, -1] }>,
    b: &Tensor<f32, { [-1] }>,
) {
    let part_x = x.partition(shape![BM, BK]);
    let part_w = w.partition(shape![BN, BK]);
    let pid: (i32, i32, i32) = get_tile_block_id();
    let k: i32 = x.shape()[1];
    let nk: i32 = ceil_div(k, BK);
    let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
    for i in 0i32..nk {
        let tx: Tile<E, { [BM, BK] }> = part_x.load([pid.0, i]);
        let tw: Tile<E, { [BN, BK] }> = part_w.load([pid.1, i]);
        let twt: Tile<E, { [BK, BN] }> = tw.transpose();
        acc = mma(tx, twt, acc);
    }
    let tb: Tile<f32, { [BN] }> = b.partition(shape![BN]).load([pid.1]);
    let tb: Tile<f32, { [BM, BN] }> = tb.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
    y.store(acc + tb);
}
```

`linear_f64` is the same with `f64` throughout (the probe ran it without the
bias lines; add them as above). Generics: `[E, BM, BN, BK]` and
`[BM, BN, BK]`. The generic entry ran in the probe with F32 and BF16.

Element types: BF16 and F32 (output and bias F32), F64 (output and bias F64).

Tests (`tests/linear.rs`), against `reference::linear` with the scaled
tolerance:

- F32 with default tiles, sizes `(m, k, n)`: `(37, 100, 70)`,
  `(64, 32, 64)` (exact), `(65, 33, 65)`, `(63, 31, 63)`, `(1, 1, 1)`,
  `(1, 1024, 16)`, `(200, 8, 3)` (`k` below `bk`).
- F32 `(37, 100, 70)` with explicit tiles `(16, 16, 16)` and `(32, 64, 32)`.
- BF16 input and F64, `(37, 100, 70)`.
- `bias: None` equals a zero bias; a non-zero bias is added.
- An identity weight (`n = k`, `w = I`) with no bias returns `x` exactly
  (F32).
- Shape errors: `w` columns differ from `x` columns, bias of the wrong
  length, bias 2-D, wrong `_into` shape, a tile that is not a power of two.

Benchmark baselines: `reference::linear`, and the kernel itself at tiles
`(16, 16, 16)` as the naive tile shape.

## The benchmark (`examples/kernelbench.rs`)

`cargo run --release -p tilekernels --example kernelbench -- <kernel>
[--runs N] [--reps R]`, where `<kernel>` is `softmax`, `rmsnorm`, `gather` or
`linear`. Defaults: `--runs 5`, `--reps 20`. Each step adds its subcommand.

For each case of the kernel:

1. Build the inputs on the host with the same xorshift generator as the
   tests, upload them, and allocate the output. Not timed.
2. **First call**: time one `_into` call. It includes the JIT. It is the
   warmup and is reported on its own, never in a median.
3. **GPU**: `runs` measurements. One measurement is the wall time of `reps`
   consecutive `_into` calls divided by `reps`. Every call waits for its
   kernel (fact 7).
4. **CPU**: one unreported warmup call of the reference, then `runs` timed
   calls on the same data, already converted to `f64` (the conversion is not
   timed).
5. Print one line per case:

```
softmax f32 4096x1024 tiles 4x1024: first 0.31 s | gpu 0.123 ms (min 0.120, max 0.140) | cpu 45.6 ms (min 45.1, max 47.0)
```

(The numbers in that line are placeholders that show the format.) `gpu` and
`cpu` are medians. Nothing is excluded except the warmups named above. The
output of the last GPU call is checked against the reference with the test
tolerance, and a mismatch makes the example exit non-zero.

Cases:

| Kernel | Cases |
|---|---|
| softmax | F32 `4096 × 1024`; F32 `16 × 32768`; BF16 `4096 × 1024` |
| rmsnorm | F32 `4096 × 1024`; BF16 `4096 × 1024`; F32 `4096 × 1000` |
| gather | F32 and BF16, table `65536 × 1024`, 4096 ids `(i * 7919) % 65536` |
| linear | F32 and BF16 `(m, k, n) = (256, 1024, 4096)`; F32 `(100, 1000, 3000)`; F32 `(1, 1024, 4096)`; and F32 `(256, 1024, 4096)` at tiles `(16, 16, 16)`, `(32, 32, 32)`, `(64, 64, 32)`, `(64, 64, 64)` |

The example's argument parsing and its median function have unit tests, like
`overbudget` (`[[example]] name = "kernelbench" test = true` in
`tilekernels/Cargo.toml`).

What a README table for a kernel step contains: the cases as rows; first
call, GPU median and CPU median as columns; the date, GPU and "median of 5
after a warmup call, 20 calls per measurement" in the caption. The "what
this shows" paragraph may say how the kernel compares with the single-threaded
CPU loop, and must say that the CPU loop is a naive baseline, not an
optimized library. Tile-shape or element-type differences inside ±10% are
reported as no difference.

## Manager changes (tiermem)

### One manager for every element type (step 6)

`Manager<T>` becomes `Manager`. The methods that touch a tensor take the
element type instead:

```rust
impl Manager {
    pub fn register<T: DType>(&mut self, tensor: Tensor<T>, hint: Hint) -> Result<TensorId, TierError>;
    pub fn register_reconstructible<T: DType>(&mut self, tensor: Tensor<T>, hint: Hint,
        rebuild: impl Fn() -> Result<Tensor<T>, Error> + 'static) -> Result<TensorId, TierError>;
    pub fn register_with<T: DType>(&mut self, tensor: Tensor<T>, hint: Hint, backing: Backing<T>)
        -> Result<TensorId, TierError>;
    pub fn unregister<T: DType>(&mut self, id: TensorId) -> Result<Option<Tensor<T>>, TierError>;
    pub fn tensor<T: DType>(&self, id: TensorId) -> Result<&Tensor<T>, TierError>;
    pub fn tensor_mut<T: DType>(&mut self, id: TensorId) -> Result<&mut Tensor<T>, TierError>;
    pub fn dtype(&self, id: TensorId) -> Result<DTypeId, TierError>;
    pub fn shape(&self, id: TensorId) -> Result<&[usize], TierError>;
    // everything else is unchanged
}
```

- New error `TierError::WrongDType { id, stored: DTypeId, requested: DTypeId }`,
  not recoverable. `tensor`, `tensor_mut` and `unregister` return it when `T`
  is not the type the tensor was registered with, before changing anything
  (`unregister` leaves the tensor registered; `tensor_mut` does not drop the
  clean copy).
- `EntrySnapshot` gains `dtype: DTypeId`, and `Snapshot`'s `Display` prints
  it.
- **Element size and the budget.** A tensor's size is `tensor.num_bytes()`,
  elements times `size_of::<T>()`, which the manager already uses. The
  budget, the soft limit, the host limit and every `Stats` field stay in
  bytes, so a `[1024]` BF16 tensor counts 2048 bytes and a `[1024]` F64
  tensor 8192, and they evict each other under the one budget. Nothing in
  the policy looks at the element type.
- **The pool.** `HostPool` and `HostBlock` are already byte-based and
  unaligned; they do not change.
- **Copies.** A copy needs no element type, only bytes:
  - `SaveToHost { keep, src, bytes, host }`: `keep` is the erased tensor,
    retained until the copy completes; `src` is its `CUdeviceptr`; the copy is
    `memcpy_dtoh_async::<u8>(host.as_ptr(), src, bytes, ..)`.
  - `LoadFromHost { host, ops }` allocates through `ops.alloc` and copies
    `host.bytes()` bytes with `memcpy_htod_async::<u8>`. Its output is the
    erased 1-D tensor, which the manager reshapes through `ops.reshape`.
- **Suggested internals** (anything with the same public behavior is fine):
  the entry stores the device tensor as `Arc<dyn Any + Send + Sync>` that is
  always an `Arc<Tensor<T>>`, plus a small `Copy` table of function pointers
  made by a generic `fn ops<T: DType>() -> DTypeOps` at registration:
  `id: DTypeId`, `device_ptr(&dyn Any) -> CUdeviceptr`, `alloc(&ExecutionContext,
  elements) -> Result<Box<dyn Any + Send + Sync>, DeviceError>` (wrapping
  `Tensor::<T>::uninitialized`, with the existing SAFETY argument),
  `reshape(Box<dyn Any + Send + Sync>, &[usize]) -> Result<Arc<dyn Any + Send
  + Sync>, Error>`. `tensor::<T>` is `downcast_ref`, `tensor_mut::<T>` is
  `Arc::get_mut` then `downcast_mut`, `unregister::<T>` is `Arc::downcast`
  then `Arc::into_inner`. A `Backing::Reconstruct` closure is wrapped at
  registration into one that returns the erased tensor. This was designed
  from the signatures in cutile 0.4 and was not compiled.
- Not designed, because only the on-hold ARC files need them: tensors with
  no elements and tensors of rank 0. Their behavior stays what it is today.

Existing code to update: `tiermem/tests/manager.rs` and
`tiermem/examples/overbudget.rs` (`Manager<f32>` becomes `Manager`; add a
turbofish where inference no longer finds `T`).

Tests added to `tiermem/tests/manager.rs`:

- `mixed_dtypes_share_one_budget`: budget of 8192 bytes; register a `[1024]`
  F32 (4096) and a `[2048]` BF16 (4096); registering a `[1024]` U8 (1024)
  evicts the least recently used one; `resident_bytes` is 5120.
- `spill_round_trip_per_dtype`: for BF16, F32, F64, I64 and U8, register a
  2-D tensor with distinct values, force it out, bring it back, and compare
  contents and shape bit-exactly.
- `wrong_dtype_is_reported`: `tensor::<f64>`, `tensor_mut::<f64>` and
  `unregister::<f64>` on an F32 tensor return `WrongDType`; afterwards the
  tensor is still registered, resident, and (for one that had a clean copy)
  `clean_evictions` still counts its next eviction.
- `dtype_and_shape_are_reported` for a resident and a spilled tensor.
- `reconstructible_bf16_is_rebuilt`: the rebuild path with a non-F32 type.
- `element_size_drives_eviction_count`: under one budget, admitting one F64
  tensor of `n` elements evicts four BF16 tensors of `n` elements.

Benchmark (no behavior should change; this is a regression check): the three
commands below, before the change and after, same day.

```sh
cargo run --release -p tiermem --example overbudget -- spill --lookahead
cargo run --release -p tiermem --example overbudget -- spill --lookahead --epochs 30
cargo run --release -p tiermem --example overbudget -- spill --lookahead --epochs 30 --prefetch 1 --work 8
```

README: a short section with the before/after table (total and run medians)
and a "what this shows" paragraph; if every difference is inside ±10%, say
that nothing changed. Update Limitations: replace "`f32` tensors only in the
benchmark" with what is now true (the manager holds any element type; the
benchmark still uses F32), and update the crate table's description.

### Registering from host memory (step 7)

```rust
/// Registers a tensor whose contents are in host memory, without touching
/// the device. The tensor starts spilled; the first `ensure_resident` or
/// prefetch copies it to the device.
pub fn register_host<T: DType>(&mut self, data: &[T], shape: &[usize], hint: Hint)
    -> Result<TensorId, TierError>;
```

- `data` is copied once into a page-locked block from the pool and can be
  dropped when the call returns. The entry is `Backing::Spill`, location
  `Host`, no clean copy, no copy in flight, `last_use` from the clock like
  any registration.
- Accounting: `spilled_bytes` and `host_bytes` grow by the tensor's bytes.
  `resident_bytes`, `evictions`, `bytes_to_host` and `bytes_to_device` do
  not change; nothing is evicted and no pressure fires. The block comes from
  `alloc_host(bytes, .., reclaim = true)` after `reserve_host()`, so it obeys
  `host_limit_bytes` and works with both `HostAlloc` modes. If the host limit
  cannot hold it, the call returns the recoverable `HostBudgetExceeded` and
  registers nothing.
- The first access is an ordinary reload: it makes room, copies, counts in
  `reloads` and `bytes_to_device`, and leaves the block as the clean copy.
  A tensor larger than the budget registers fine and fails at that access
  with `BudgetExceeded`, as a spilled tensor would.
- Any hint is allowed. `Pinned` takes effect once the tensor is on the
  device.
- New error `TierError::InvalidArgument(String)`, not recoverable, when the
  product of `shape` is not `data.len()` or `data` is empty.
- The byte copy lives in one new method, `HostBlock::fill_from(&mut self,
  bytes: &[u8])` in `host.rs` (it asserts equal lengths), and `register_host`
  views `data` as bytes with `slice::from_raw_parts(data.as_ptr().cast::<u8>(),
  size_of_val(data))`. Both need a `// SAFETY:` comment: `DType` types are
  plain data without padding, and the block is exclusively owned and not yet
  reachable. Update `HostBlock`'s doc comment, which says a block is only
  filled by the device.
- Trace: a new `EventKind::AdmitHost`, named `admit_host`, as a span over the
  call with the tensor's bytes.

Tests added to `tiermem/tests/manager.rs`:

- `register_host_does_not_touch_the_device`: `Policy::Strict`, a budget of
  three tensors, six tensors registered from host: all succeed,
  `resident_bytes == 0`, `spilled_bytes == host_bytes == 6 * BYTES`,
  `evictions == 0`, `bytes_to_host == 0`, none resident.
- `register_host_round_trip`: a 2-D F32 tensor and a BF16 tensor come back
  with the right contents and shape after `ensure_resident`; `reloads == 1`
  each.
- `host_registered_tensor_evicts_cleanly`: after the first access, evicting
  it is a clean eviction and `bytes_to_host` stays 0.
- `register_host_obeys_the_host_limit`: with a host limit of two tensors, the
  third returns `HostBudgetExceeded`, is recoverable, and the snapshot still
  has two entries. Run for `HostAlloc::Lazy` and `HostAlloc::Eager`.
- `register_host_rejects_bad_arguments`: wrong shape product and empty data
  give `InvalidArgument`.
- `host_registered_tensors_can_be_prefetched`: `prefetch(&[id])` starts the
  load and a later `ensure_resident` finds it.
- `fill_from` has no unit test of its own, because blocks only come from a
  pool with a device; the round-trip test covers it.

Benchmark: add `--register device|host` to `overbudget` (default `device`,
the current behavior; `spill` mode only, an error with `rebuild`). With
`host`, the layers are built as host vectors before the setup timer starts
(they stand in for weights that were read from a file; say so in the README
caption as the thing excluded) and registered with `register_host`. The
argument parser's tests cover the new flag.

```sh
cargo run --release -p tiermem --example overbudget -- spill --lookahead
cargo run --release -p tiermem --example overbudget -- spill --lookahead --register host
cargo run --release -p tiermem --example overbudget -- spill --lookahead --host-alloc eager --host-limit-mib 768
cargo run --release -p tiermem --example overbudget -- spill --lookahead --host-alloc eager --host-limit-mib 768 --register host
cargo run --release -p tiermem --example overbudget -- spill --lookahead --epochs 30
cargo run --release -p tiermem --example overbudget -- spill --lookahead --epochs 30 --register host
```

README: a section with setup, run and total medians for both ways of
registering, plus memory page-locked, bytes copied to the host and reloads.
What is expected, to be confirmed or contradicted by the numbers: setup no
longer allocates device memory or copies device to host, all 12 layers are
page-locked instead of only the ones that were spilled, and the run reloads
up to four layers more because nothing starts on the device. Add a
limitation: `register_host` copies the data once more, from the caller's
memory into the page-locked block.

### Kernels on managed tensors (step 8)

No new API. A new test file, `tiermem/tests/kernels.rs`, with `tilekernels`
as a dev-dependency of `tiermem` (`tilekernels = { path = "../tilekernels" }`
under `[dev-dependencies]`):

- `linear_runs_on_weights_registered_from_host`: three BF16 weights of
  `[64, 256]` and three F32 biases of `[64]` registered with `register_host`
  under a budget that holds one weight and one bias, `Policy::EvictLookahead`
  with the access order declared. For each layer: `ensure_resident(&[w, b])`,
  `tilekernels::linear(&x, m.tensor::<bf16>(w)?, Some(m.tensor::<f32>(b)?))`
  with an unmanaged BF16 `x` of `[5, 256]`, compare with
  `tilekernels::reference::linear` at the linear tolerance. Then check
  `peak_resident_bytes <= budget` and `reloads == 6`.
- `gather_runs_on_a_managed_table`: an I64 table and an F32 table registered
  from host under a budget that holds one of them; gather from each in turn,
  twice; results bit-exact; the second round's evictions are clean.

No benchmark. README: one sentence in the `tilekernels` section saying the
kernels run on tensors the manager holds, pointing at the test. No table.

## Steps

Smallest first. Each is one commit series and is testable alone. Steps 2 to
5 depend on step 1; step 7 on step 6; step 8 on steps 4, 5 and 7. Steps 6
and 7 do not depend on the kernels.

| Step | What | Tests | Benchmark and README |
|---|---|---|---|
| 1 | The `tilekernels` crate: `Cargo.toml`, workspace member, `KernelError`, `Float`, `tiles.rs`, `reference.rs`. No kernel yet. | Unit tests, no GPU: `tiles.rs` (each rule at 1, a power of two, one over, one under; the 16,777,216 limit) and `reference.rs` (softmax of `[0, ln 2]` is `[1/3, 2/3]`; rmsnorm of `[3, 4]` with gain 1 and eps 0 is `[3, 4] / sqrt(12.5)`; gather with an out-of-range id; linear of a 2 × 3 by 2 × 3 case worked by hand, with its scale). | None. README: add the crate to the Crates table as "CPU references and tile-shape rules for the kernels; kernels follow", and link this document next to the feasibility report. |
| 2 | Softmax: kernel entries, `softmax`, `softmax_into`, `tests/common/mod.rs`, `tests/softmax.rs`, the `kernelbench` example with the `softmax` subcommand. | "Softmax" above, plus the example's unit tests. | `kernelbench -- softmax`. README: a `tilekernels` section with the softmax table, and Limitations for it (one tile per row, first-call time for wide rows as measured by `wide_row`). |
| 3 | RMSNorm. | "RMSNorm" above. | `kernelbench -- rmsnorm`. README table. |
| 4 | Embedding gather. | "Embedding gather" above. | `kernelbench -- gather`. README table; Limitations: I32 ids only, out-of-range ids give zeros, the whole table must be resident. |
| 5 | Linear. | "Linear" above. | `kernelbench -- linear`. README table including the tile sweep; change `for_shape`'s default only as "Tile-shape rules" allows, and say which shape was kept. Limitations: F32 output for BF16 input. |
| 6 | One manager for every element type. | "One manager for every element type" above; all existing manager tests still pass. | The three `overbudget` commands, before and after. |
| 7 | `register_host`, and `overbudget --register`. | "Registering from host memory" above. | The six `overbudget` commands. |
| 8 | Kernels on managed tensors. | `tiermem/tests/kernels.rs`. | None. Last step: hand to review. |

Commands to hand over for a kernel step, run by the user, who pastes back
the printed lines:

```sh
cargo run --release -p tilekernels --example kernelbench -- <kernel>
```

For steps 6 and 7 the lines to paste back are `overbudget`'s `time`,
`setup`, `run` and `page-locking` lines and its counters, for each command;
step 6 needs them once on the commit before the change and once after.

## Open questions

None yet. An implementer who is blocked writes the question here, with the
exact error text and what was tried, and stops.
