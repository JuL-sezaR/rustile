# Review: step 6, one manager for every element type

Reviewed 2026-11-16: branch `manager-any-dtype` against `main` (`3f3998e`),
commits `f2dae3a`..`8561540`, against step 6 of
[docs/design/kernels.md](../design/kernels.md) (as revised in `f2dae3a`) and
the project ground rules. Nothing was fixed.

The public API, the order of the type check, the erased copies, the
snapshot and error text, the seven new tests and the benchmark procedure
match the spec. The benchmark counters were identical in all nine pairs, and
the twelve before/after ranges in the README were recomputed from the pasted
output and are correct. No remaining bare float literal in the tests
registers an `f64` by accident: the literal arrays left all feed typed
helpers (`tensor(v)`, `sized(v, ..)`).

## Problems, most severe first

### 1. `DTypeOps::alloc` is a safe function that returns uninitialized memory

`tiermem/src/dtype.rs:28` and `:44`–`:51`.

`alloc` has the type `fn(&ExecutionContext, usize) -> Result<Box<Erased>,
DeviceError>` and its body calls `assume_init` on a fresh
`Tensor::uninitialized`. Its `SAFETY` comment says "the copy that follows
initializes every element", but that copy is in another file
(`LoadFromHost::execute` in `copy.rs`), and nothing ties the two together:
any code in the crate can call `(ops.alloc)(ctx, n)` from safe code and get a
`Tensor<T>` of whatever the device memory held.

Before step 6 the `assume_init` sat inside `unsafe fn execute`, next to the
copy that makes it sound. The manager accepts any `DType`, including `bool`,
and cutile's `DType` contract makes reading a `bool` tensor with bytes other
than 0 and 1 undefined behaviour. So a safe call, followed by any read-back,
is undefined behaviour.

No caller misuses it today; the only call is the one in `LoadFromHost`,
which fills the tensor at once. Direction, not applied: make the field an
`unsafe fn` pointer with a `# Safety` section, so the call in
`LoadFromHost::execute` is an `unsafe` block that carries the argument.

### 2. "Nothing changes before the type check" is tested only for resident and spilled tensors

`tiermem/src/lib.rs:429`–`:434` (`unregister`), `:615`–`:620`
(`tensor_mut`); test `wrong_dtype_is_reported`, `tiermem/tests/manager.rs:1328`.

The spec names `finish_load` and `finish_save` as things that must not run
before the check passes. The code is right: `typed_entry` runs first in both
methods. But no test calls `unregister::<f64>` or `tensor_mut::<f64>` on a
tensor whose prefetch is in flight (`Location::Loading`) or whose write-back
is in flight (`save`), and then checks that the copy is still in flight and
that `reloads` and `write_backs` did not move. Moving the check below
`finish_load` or `finish_save` would pass every test. Ground rule: a test for
every behaviour change.

### 3. The README claims more than the benchmark measured

`README.md:409`: "This changes no behavior for `f32`, and that was measured."

The three commands measured are all `spill --lookahead` with read-only
layers. Paths that step 6 changed but that no benchmark run took:

- the copy back of a modified tensor (`--dirty`, the asynchronous
  `SaveToHost` path);
- `rebuild` mode (the erased `Reconstruct` closure);
- a host limit, `--host-alloc eager`, and the LRU policy.

The tests cover them for correctness, but the sentence, and "the counters
are identical", should be limited to the three commands that were run.

### 4. `WrongDType`'s `Display` text has no test

`tiermem/src/error.rs:94`.

The spec fixes the text (`tensor {id:?} holds {stored}, not {requested}`),
and nothing asserts it: `snapshot_reports_the_dtype` checks only the
snapshot's text. Low.

### 5. README: "Eight tests did that"

`README.md:430`.

Five literal arrays registered `f64` tensors, one of them in the
`big_manager` helper, and eight tests failed because of them. The sentence
says eight tests did it. Low.

### 6. The spec review is on two branches

`f2dae3a` on this branch is a cherry-pick of `167c2b1` on
`spec-step6-review`. If both branches are merged, the same change to
`docs/design/kernels.md` lands twice in the history. The content is the
same, so there is no conflict; merging only this branch, or closing
`spec-step6-review`, avoids the duplicate. Process, low.
