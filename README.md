# rustile → tiermem

Toward an AI-aware OS layer: coordinated memory, scheduling and observability
for ML workloads across CPU RAM, GPU VRAM and other accelerators.

The plan is **user space first**. Every idea on the list (tiered memory,
recoverable OOM, workload-aware scheduling) needs cooperation from the GPU
driver and runtime, so each one starts as a Rust library on top of
[cuTile Rust](https://github.com/nvlabs/cutile-rs). Only the pieces that need
enforcement across processes (quotas, isolation, accounting) move toward the
kernel, through cgroups, eBPF or HMM, and only once the API has been proven
here.

Ideas that are not implemented or not measured are in
[NOVELTIES.md](NOVELTIES.md). The feasibility report for the next milestone
is in [docs/milestone-2-feasibility.md](docs/milestone-2-feasibility.md), and
the design for the tile kernels and the manager changes they need is in
[docs/design/kernels.md](docs/design/kernels.md).

## Crates

| Crate | What it is |
|---|---|
| [`tiermem`](tiermem/) | VRAM budget manager: residency hints, eviction (spill to pinned host RAM or rebuild), lookahead eviction, prefetch and asynchronous write-back on copy streams of its own, a pool of page-locked host memory (locked as needed or up front) with a limit, soft-limit pressure callbacks with hysteresis, recoverable budget errors with a diagnostic snapshot, Chrome-trace timeline. |
| [`tilekernels`](tilekernels/) | Tile kernels on cuTile Rust, each tested against a CPU reference: softmax so far. |
| [`vecadd`](vecadd/) | The original cuTile hello-world: a single tile kernel adding two vectors. |

## Milestone 1: run a working set larger than the budget

`tiermem/examples/overbudget.rs` streams 12 × 64 MiB "layers" (768 MiB)
through an add kernel, three passes, under a 256 MiB budget.

```sh
cargo run --release -p tiermem --example overbudget -- strict    # baseline: fails at layer 4
cargo run --release -p tiermem --example overbudget -- spill     # LRU, spill to host RAM
cargo run --release -p tiermem --example overbudget -- rebuild   # LRU, regenerate on device
cargo run --release -p tiermem --example overbudget -- spill --lookahead  # declared layer order
cargo run --release -p tiermem --example overbudget -- spill --lookahead --prefetch 1  # reload ahead of use
cargo run --release -p tiermem --example overbudget -- spill --lookahead --host-limit-mib 576  # cap host RAM
cargo run --release -p tiermem --example overbudget -- spill --lookahead --host-alloc eager  # page-lock host RAM up front
cargo run --release -p tiermem --example overbudget -- spill --budget-mib 1024  # everything fits
# add --work N to run the kernel N times per layer (a heavier layer)
# add --dirty to mark each layer as written after use, so evictions copy back
# add --copy-streams 1 to run copies in both directions on one stream
# add --runs N for N measured runs after a warmup run (default 5)
# add --trace out.json and open it in ui.perfetto.dev
```

First results (RTX 5080 Laptop, WSL2, 2026-10-18):

| Mode | Result | Time | Reloads | Bytes moved |
|---|---|---|---|---|
| strict | recoverable `BudgetExceeded` at layer 4 | — | — | 0 |
| spill | correct | 1.88 s | 36 / 36 accesses | 5120 MiB |
| rebuild | correct | 0.068 s | 36 rebuilt | 0 |
| fits in budget | correct | 0.076 s | 0 | 0 |

What these show, and what to fix next:

- **LRU thrashes on layer sweeps.** Every access missed: a cyclic scan over
  more items than fit is LRU's worst case. A model's layer order is
  predictable, so the workload should be able to tell the manager what comes
  next. *Done, see below.*
- **Spilling re-copies unchanged weights.** Weights are read-only, yet each
  eviction copied them back to host RAM (2816 MiB of the traffic). Keeping a
  clean host copy for immutable tensors would make eviction just a free.
  *Done, see below.*
- **Migration is synchronous and goes through pageable memory.** Pinned host
  buffers and a separate copy stream let reloads overlap compute. *Done, see
  below.*
- **Pressure fires on every re-crossing** (45 times here). It needs
  hysteresis or rate limiting before it is useful to an application. *Done,
  see below.*

## Lookahead eviction

`Policy::EvictLookahead` plus `Manager::expect_accesses(&ids)` lets the
workload declare its upcoming access order. Within a hint rank, the manager
evicts tensors with no declared use first, then the one needed farthest in the
future (Belady's rule, optimal for miss count). Each `ensure_resident` consumes
the plan up to that tensor, so skipping accesses doesn't leave stale entries.

Same benchmark (RTX 5080 Laptop, WSL2, 2026-10-29):

| Mode | Misses | Time | Bytes moved |
|---|---|---|---|
| spill, LRU | 36 / 36 | 1.59 s | 5120 MiB |
| spill, lookahead | 27 / 36 | 1.21 s | 3968 MiB |
| rebuild, LRU | 36 rebuilt | 0.056 s | 0 |
| rebuild, lookahead | 27 rebuilt | 0.057 s | 0 |

27 is the floor for 12 layers in 4 slots: 3 layers stay resident and the other
9 rotate through the fourth slot. Further gains have to come from cheaper
misses, not fewer: skipping write-back of clean weights (next item) and
overlapping reloads with compute. Prefetching only pays off once copies are
asynchronous.

## Clean host copies

A spilled tensor that is reloaded keeps its host copy. As long as nothing
writes to it, the next eviction just frees the device memory. The only way to
write is `Manager::tensor_mut`, which drops the clean copy, so the next
eviction copies the tensor back. `Stats::clean_evictions` counts the
evictions that skipped the copy, and `Stats::host_bytes` reports all host RAM
held, which now includes copies of resident tensors.

Same benchmark (RTX 5080 Laptop, WSL2, 2026-11-01):

| Mode | Time | To host | To device | Bytes moved | Host RAM held |
|---|---|---|---|---|---|
| spill, LRU | 0.80 s (was 1.59) | 768 MiB | 2304 MiB | 3072 MiB (was 5120) | 768 MiB |
| spill, lookahead | 0.61 s (was 1.21) | 704 MiB | 1728 MiB | 2432 MiB (was 3968) | 704 MiB |

Each layer now goes to the host once, the first time it is evicted. The rest
of the traffic is reloads, which only asynchronous copies can hide. The cost
is host RAM: clean copies stay until the tensor is written to or
unregistered, or until the host limit (below) needs the room.

## Pinned host memory, a copy stream, and prefetch

Spilled contents now live in page-locked host memory, and every migration
runs on a CUDA stream that belongs to the manager, apart from the streams the
kernels use. That gives two things:

- Copies are faster, because the driver transfers straight to and from the
  buffer instead of staging through its own.
- A reload can be started without waiting for it. `Manager::prefetch(&ids)`
  submits the copies and returns; a later `ensure_resident` waits only for
  what is left. With `Config::prefetch_depth = n`, each `ensure_resident`
  starts reloading the next `n` tensors declared with `expect_accesses`.

A prefetch makes room like any reload, but it never evicts the arguments of
the last `ensure_resident` (they are in use), and an automatic prefetch never
evicts a tensor that is declared to be used sooner than the one it fetches. A
tensor being reloaded counts against the budget from the moment the copy
starts. If it is evicted before it is used, the copy is cancelled and the
host copy stays valid (`Stats::prefetches_cancelled`).

Same benchmark, previous commit against this one, both measured on the same
day (RTX 5080 Laptop, WSL2, 2026-11-12; times vary by about ±10% run to run):

| Mode | 3 epochs, before | 3 epochs, now | 30 epochs, before | 30 epochs, now |
|---|---|---|---|---|
| spill, LRU | 0.67 s | 0.41 s | 3.05 s | 1.45 s |
| spill, lookahead | 0.56 s | 0.45 s | 2.30 s | 1.15 s |
| fits in budget | — | 0.07 s | — | 0.17 s |

A 64 MiB reload now takes about 2.4 ms. What is left in the short run is
mostly page-locking the host buffers. It happens once per layer, the first
time it is spilled, and does not recur, which is why the 30-epoch run gains
more. (This section first put that cost at 29 ms per 64 MiB buffer, about
320 ms per run. Timed directly it is about 18 ms per buffer and 0.2 s per
run; see "Host memory pool" below.)

Prefetch only pays when there is compute to overlap with. One pass of the add
kernel takes about 0.3 ms per layer, far less than the copy, so `--work N`
runs it N times per layer (30 epochs, lookahead):

| Work per layer | Fits in budget | No prefetch | `--prefetch 1` | Overhead hidden |
|---|---|---|---|---|
| 1 kernel (~0.3 ms) | 0.17 s | 1.15 s | 1.10 s | none |
| 8 kernels (~2.7 ms) | 0.97 s | 1.87 s | 1.38 s | about half |
| 32 kernels (~10 ms) | 3.67 s | 4.53 s | 3.99 s | about 60% |

All of these runs still include about 0.2 s of one-time page-locking, which
is part of the 0.32 s that remains with 32 kernels per layer. Prefetching reloads slightly more (10 layers
per epoch instead of 9, because one of the four slots holds the incoming
layer). `--prefetch 2` measured the same as 1: copies on one stream run one
after another.

## Pressure hysteresis

The pressure callback fires when residency rises above the soft limit, then
stays quiet until residency has fallen to `Config::pressure_rearm_bytes`
(60% of the budget by default) and risen again. The benchmark, which
oscillates between three and four resident layers, now reports 1 pressure
event instead of 45.

## Host memory limit

`Config::host_limit_bytes` caps the host RAM held for spilled contents and
clean copies. When a spill would exceed it, the manager first gives up clean
copies of resident tensors (least recently used first) and reuses the freed
page-locked buffer for the spill. If none are left, the eviction fails with
the recoverable `TierError::HostBudgetExceeded` and the tensor stays on the
device.

With `--lookahead --host-limit-mib 576` the benchmark passes in 0.33 s, with
4 clean copies given up and 832 MiB copied to the host instead of 704: a
tensor without a clean copy has to be copied back when it is evicted. (An
earlier version of this section reported 29 clean copies and 2432 MiB. That
was measured before a freed buffer was reused for the next spill, and the
text was not updated when that changed.) 576 MiB
is the minimum here, the 512 MiB that cannot be on the device plus one layer,
because a victim is spilled before the incoming layer's host copy can be
released. With `--prefetch 1` added, the same limit passes in 0.31 s, with
576 MiB copied to the host and no clean copy given up.

## Host memory pool

Page-locked host memory now comes from a pool (`tiermem/src/host.rs`) instead
of one driver allocation per tensor. A block that is given back, because its
tensor was written to, unregistered, or lost its clean copy to the host
limit, serves a later spill without being locked again. The pool hands out
bytes, packed without padding, so tensors of any element type and size can
share it and the host limit counts exactly the bytes held.

`Config::host_alloc` chooses when the memory is locked:

- `HostAlloc::Lazy` (the default) locks a block the first time a spill finds
  no free one of its size, as before. Each block is locked on its own, so
  blocks of different sizes never leave unusable gaps; an unused block of
  another size is unlocked when the host limit needs its room.
- `HostAlloc::Eager` locks one arena of `Config::host_limit_bytes` when the
  first spillable tensor is registered, and never grows it. If its free space
  ends up in pieces too small for a spill, the blocks are moved together.

In both modes a spill fails with `HostBudgetExceeded` only when the bytes
held plus the bytes asked for exceed the limit.

`Stats::pinned_bytes` and `Stats::pin_time` report what was locked and how
long it took. The benchmark now splits its time into setup (creating and
registering the layers), run (the epochs) and the final check, because
registering 12 layers under a 4-layer budget already spills 8 of them.

Same benchmark, previous commit against this one (RTX 5080 Laptop, WSL2,
2026-11-12; median of 5 processes after one warmup process, nothing else
excluded; the previous commit only reports a total):

| Mode | Total, before | Total, lazy | Total, eager | Run only, lazy | Run only, eager |
|---|---|---|---|---|---|
| spill, LRU, 3 epochs | 0.42 s | 0.42 s | 0.42 s | 0.20 s | 0.11 s |
| spill, lookahead, 3 epochs | 0.37 s | 0.37 s | 0.40 s | 0.15 s | 0.086 s |
| spill, LRU, 30 epochs | 1.41 s | 1.31 s | 1.38 s | 1.08 s | 1.00 s |
| spill, lookahead, 30 epochs | 1.08 s | 1.08 s | 1.09 s | 0.85 s | 0.77 s |

| One-time cost | Lazy | Eager |
|---|---|---|
| Memory page-locked | 704 MiB, in 11 blocks | 768 MiB, one arena |
| Time page-locking | 0.19 to 0.21 s | 0.21 to 0.27 s |
| Setup phase | 0.19 s | 0.27 to 0.34 s |

What this shows:

- **The total does not change.** Page-locking costs about 0.3 ms per MiB
  whether it is done in one call or eleven, so one arena is not cheaper than
  separate buffers. Eager allocation moves the cost; it does not remove it.
- **The run phase loses it.** With eager allocation nothing is locked after
  setup, and the 3-epoch run drops from 0.15 s to 0.086 s (lookahead) and
  from 0.20 s to 0.11 s (LRU). At 30 epochs the same 0.07 to 0.08 s is less
  than 10% of the run, which is inside the run-to-run variance, so no
  difference is claimed there.
- **Lazy allocation already paid most of it during setup** in this
  benchmark: only the 3 or 4 layers that are first spilled during the first
  epoch are locked inside the run.
- **The cost is smaller than this README said.** The 29 ms per 64 MiB buffer
  quoted earlier does not hold up when the allocation call is timed
  directly: the median is about 18 ms per buffer (0.2 s for 704 MiB), though
  single runs reached 0.35 s.

Under a host limit the arena behaves like lazy allocation:
`--lookahead --host-limit-mib 576` gives the same counters either way (4
clean copies given up, 832 MiB copied to the host) and the same total (0.35 s
lazy, 0.34 s eager).

## Asynchronous write-back and a copy stream per direction

Until now an eviction that had to copy a modified tensor back to the host
waited for the copy. Three changes:

- `Manager::write_back(&ids)` starts the copy and returns. The tensor stays
  resident, counts against the budget and can still be read. When the copy
  has finished the tensor has a clean host copy, so evicting it is only a
  free. A tensor whose copy is in flight cannot be reused until the copy
  finishes: evicting it, unregistering it or calling `tensor_mut` waits for
  what is left, and writing to it then discards the copy.
- Automatic prefetch (`Config::prefetch_depth`) no longer waits for a copy
  in either direction. It makes room only by evicting tensors that are free
  to evict (a clean host copy, or rebuilt), in the policy's order, and only
  if that makes enough room; otherwise it evicts nothing. It starts
  write-backs for the tensors the policy would evict next, so they are free
  by the time the following prefetch needs the room. An access that still
  finds no room evicts as before, waiting if it must.
- A copy made ahead of need (`write_back`, or the write-backs prefetch
  starts) only takes host memory nobody is using. Under a full host limit it
  does not start, rather than give up another tensor's clean copy. An
  eviction that needs host memory can take back the block of a write-back,
  finished or not.
- Copies to the device and copies to the host run on separate streams
  (`Config::copy_streams`, 2 by default; 1 restores the single stream).

The driver reports `asyncEngineCount = 1` for this GPU
(`tiermem::async_engine_count()`, printed by the benchmark): it runs one copy
at a time. So no claim is made that a reload and a write-back overlap each
other, and the measurements below agree: one stream or two makes no
difference here.

The read-only benchmark never writes back after the first spill, so
`--dirty` marks each layer as written after it is used. Run phase only,
30 epochs, lookahead, previous commit against this one (RTX 5080 Laptop,
WSL2, 2026-11-12; median of 5 processes after one warmup process, nothing
else excluded):

| Work per layer | Fits in budget | No prefetch, before | No prefetch, now | `--prefetch 1`, before | `--prefetch 1`, now | now, `--copy-streams 1` |
|---|---|---|---|---|---|---|
| 1 kernel (~0.3 ms) | 0.12 s | 1.41 s | 1.42 s | 1.50 s | 1.63 s | 1.66 s |
| 8 kernels (~2.7 ms) | 0.92 s | 2.26 s | 2.25 s | 1.86 s | 1.79 s | 1.74 s |
| 32 kernels (~10 ms) | 3.57 s | 4.95 s | 5.04 s | 4.53 s | 3.85 s | 4.01 s |

What this shows:

- **Write-back is hidden once a layer's compute is longer than its two
  copies.** A step moves one layer out and one in, about 5 ms together
  because the device runs them one after the other. With 32 kernels per
  layer (10 ms) both fit behind the compute: the run drops from 4.53 s to
  3.85 s, and what is left above the run that fits in the budget falls from
  0.96 s to 0.28 s.
- **With 8 kernels per layer nothing is gained** (1.86 s before, 1.79 s
  now, inside the variance). Before, a step cost its compute plus the copy
  out; now it costs the two copies, which no longer wait for the compute
  but take as long as compute plus one copy did.
- **With almost no compute, prefetching a written workload costs time**
  (1.63 s against 1.42 s without it, and it was 1.50 s before), because it
  reloads more. The layer being written back keeps its slot until its copy
  has finished, next to the slot for the incoming layer, which leaves one
  layer resident for long instead of three: 10.8 reloads per epoch instead
  of 8.8, and 41984 MiB moved instead of 34304. At 8 kernels per layer the
  write-backs fall behind the prefetches, a second outgoing layer is still
  being copied when room is needed, and every access misses: 12 reloads per
  epoch and 46464 MiB. At 32 kernels it is 10.8 again. (This paragraph first
  gave the 8-kernel counters as the 1-kernel ones. The traces behind the
  correction are in issue #7.)
- **A second stream does not help on this device.** The differences between
  one and two streams (1.74 s and 1.79 s, 4.01 s and 3.85 s) are inside the
  variance, as one async engine predicts.
- Without prefetch nothing changes: an eviction forced by an access still
  waits for its copy.

The read-only benchmark, re-run the same way (run phase, 30 epochs,
lookahead; each cell is before / now):

| Work per layer | Fits in budget | No prefetch | `--prefetch 1` | `--prefetch 2` |
|---|---|---|---|---|
| 1 kernel | 0.12 s | 0.87 / 0.87 s | 0.85 / 0.84 s | 0.90 / 0.90 s |
| 8 kernels | 0.92 s | 1.72 / 1.69 s | 1.13 / 1.12 s | 1.15 / 1.16 s |
| 32 kernels | 3.74 s | 4.51 / 4.27 s | 3.84 / 3.90 s | 3.87 / 4.04 s |

Nothing moved, which is the expected result: read-only layers have no
write-backs to make asynchronous. `--prefetch 2` still matches
`--prefetch 1`. Two reloads share the stream for copies to the device, and
with one async engine a stream of their own would not help either; a depth
of 2 only reloads more (320 layers against 289 over the run). Measured on the
run phase alone, prefetching hides about 70% of the overhead at 8 and 32
kernels per layer; the "about half" and "about 60%" above were computed on
totals that included setup.

## Measuring

The benchmark now repeats itself. `--runs N` (default 5) does one warmup run
that is not reported, then N runs, each with a fresh manager, and prints the
median, minimum and maximum of the total, setup, run and page-locking times.
The counters and the trace are those of the last run. A workload that cannot
be admitted at all (`strict`) is reported once and not repeated.

```
time            0.367 s  (min 0.364, max 0.393)
  setup         0.181 s  (min 0.181, max 0.183)
  run           0.146 s  (min 0.145, max 0.172)
page-locking    0.192 s  (min 0.190, max 0.218)  for 704 MiB (as first spilled)
```

The tables above were measured by running the whole process six times and
dropping the first. The in-process medians agree with them (RTX 5080 Laptop,
WSL2, 2026-11-12; `spill --lookahead`, median of 5 after the warmup):

| | Processes | `--runs 5` |
|---|---|---|
| 3 epochs, total | 0.372 s | 0.367 s |
| 3 epochs, run | 0.148 s | 0.146 s |
| 3 epochs, run, `--host-alloc eager` | 0.086 s | 0.089 s |
| 30 epochs, run | 0.853 s | 0.871 s |
| 30 epochs, run, `--host-alloc eager` | 0.769 s | 0.788 s |

Each run page-locks its host memory again, so the one-time costs are still
in every run, reported on their own lines.

**The copies themselves are close to the link's limit.** Taken from the trace
of a 30-epoch run (`--lookahead --host-alloc eager --trace`), a 64 MiB
reload takes 2.46 ms (264 reloads, 10th to 90th percentile 2.44 to 2.50 ms),
which is 27 GB/s, and a 64 MiB copy back to the host takes 2.51 ms (272 of
them with `--dirty`). `nvidia-smi` reports the GPU's link as PCIe 5.0 x8,
which carries 31.5 GB/s before protocol overhead. So there is at most about
15% left to gain from faster copies. What remains has to come from avoiding
copies (fewer misses, clean copies) or hiding them behind compute (prefetch,
write-back).

## Tile kernels

`tilekernels` holds kernels written in cuTile Rust, each tested against a
single-threaded CPU loop in `f64` (`tilekernels/src/reference.rs`). The
design, and the order they are built in, is in
[docs/design/kernels.md](docs/design/kernels.md). So far: softmax.

### Softmax

`tilekernels::softmax(&x)` normalizes each row of a `[rows, cols]` tensor.
BF16 and F32 input give F32 output; F64 gives F64. One tile spans a whole
row, its width rounded up to a power of two with the padding masked out, and
up to 16 rows share a tile while it stays within 4096 elements. The tests
hold every element to within 1e-6 of the reference for F32 output and 1e-14
for F64.

```sh
cargo run --release -p tilekernels --example kernelbench -- softmax
```

RTX 5080 Laptop, WSL2, 2026-11-13. GPU and CPU are medians of 5 measurements
after a warmup call, with min to max in brackets; a GPU measurement is the
average of 20 calls, each waiting for its kernel. First call is that warmup:
one observation per case, including the JIT compile.

| Case | Tile | First call | GPU, per call | CPU reference |
|---|---|---|---|---|
| F32, 4096 × 1024 | 4 × 1024 | 0.22 s | 0.086 ms (0.084 to 0.095) | 19.3 ms (19.0 to 20.3) |
| F32, 16 × 32768 | 1 × 32768 | 0.68 s | 0.077 ms (0.074 to 0.078) | 1.5 ms (1.5 to 1.7) |
| BF16 in, 4096 × 1024 | 4 × 1024 | 0.18 s | 0.084 ms (0.082 to 0.087) | 20.4 ms (19.8 to 21.5) |

What this shows:

- **The GPU time does not follow the size.** 16 × 32768 has an eighth of
  the elements of 4096 × 1024 and takes 0.077 ms against 0.086 ms, a
  difference inside the variance. So most of the 0.08 ms is what any call
  costs: checking the arguments, launching, and waiting for the stream. The
  column is per-call latency; the kernel's own time is not measured here.
- **BF16 input costs the same as F32** (0.084 against 0.086 ms).
- **Against the CPU loop**, a call is about 220 times faster at
  4096 × 1024 and 19 times at 16 × 32768. The CPU loop is a naive
  single-threaded baseline in `f64`, not an optimized library, so the ratios
  show the kernel is worth having, not how fast softmax can be.
- **The first call is the compile**: 0.18 to 0.22 s for a 4096-element tile
  and 0.68 s for a 32,768-element one. The first case also pays cutile's
  check of the toolchain, which runs once per process, so its 0.22 s is not
  directly comparable with the 0.18 s of the BF16 case.

## Limitations

- The budget is the manager's own accounting, not the driver's free-memory
  counter, and unmanaged allocations (here the two accumulators) are outside it.
- Single GPU, single thread, `f32` tensors only in the benchmark.
- A modified tensor is copied back ahead of time only alongside automatic
  prefetch or when `write_back` is called. An eviction forced by an access
  still waits for the copy, and rebuilding a reconstructible tensor is
  synchronous.
- This GPU reports `asyncEngineCount = 1`: it runs one copy at a time, so
  uploads (reloads) and write-backs cannot overlap. They take turns even on
  separate streams. Whether they overlap on a device with more async
  engines is untested.
- Prefetching a workload that writes reloads more (10.8 to 12 layers per
  epoch here instead of 8.8, see issue #7), and is slower than not
  prefetching when a layer's compute is shorter than its copies (see the
  `--dirty` table).
- With `Policy::EvictLru`, a written workload gets write-backs from
  automatic prefetch but no prefetches: the tensor written back is the one
  the next access evicts. Seen in review, not measured in the benchmark.
- In lazy mode a write-back started by prefetch can page-lock a block, which
  takes as long as it would have at eviction.
- Writing to a tensor whose write-back is in flight waits for the copy and
  then throws it away.
- Page-locked host memory is unlocked only when the host limit needs the
  room for a block of another size, or when the manager is dropped. There
  is no call to give unused blocks back.
- The eager arena needs a host limit. With tensors of different sizes its
  free space can end up in pieces; the manager then moves blocks together,
  which waits for every copy in flight and copies host memory. That cost is
  not measured: the benchmark's layers are all the same size.
- Eager allocation moves page-locking into setup but does not make it
  cheaper (about 0.3 ms per MiB either way).
- A failed prefetch on the copy stream is reported by the next
  `ensure_resident` of that tensor, which leaves it spilled.
- Softmax needs a row to fit in one tile (at most 16,777,216 columns): in
  cutile 0.4 a block can only store its own tile, so a row cannot be reduced
  across blocks. The first call's compile time grows with the row width
  (0.68 s at 32,768 columns above).
- The kernels write F32 for BF16 input; there is no BF16 output.
- The kernels' first-call times are single observations, not medians.
- The kernels do not run on tensors the manager holds yet.

## Building

Requires an NVIDIA GPU and CUDA toolkit supported by `cutile` 0.4. The cutile
crates are pinned exactly in the workspace `Cargo.toml` because they are pre-1.0.

```sh
cargo test -p tiermem       # runs on the GPU
cargo test -p tilekernels   # runs on the GPU
```
