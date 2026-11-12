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

## Crates

| Crate | What it is |
|---|---|
| [`tiermem`](tiermem/) | VRAM budget manager: residency hints, eviction (spill to pinned host RAM or rebuild), lookahead eviction and prefetch on a separate copy stream, a pool of page-locked host memory (locked as needed or up front) with a limit, soft-limit pressure callbacks with hysteresis, recoverable budget errors with a diagnostic snapshot, Chrome-trace timeline. |
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
released.

## Host memory pool

Page-locked host memory now comes from a pool (`tiermem/src/host.rs`) instead
of one driver allocation per tensor. A block that is given back, because its
tensor was written to, unregistered, or lost its clean copy to the host
limit, goes on a free list and serves a later spill without being locked
again. The pool hands out bytes, so tensors of any element type can share it.

`Config::host_alloc` chooses when the memory is locked:

- `HostAlloc::Lazy` (the default) locks a block the first time a spill finds
  no free one, as before.
- `HostAlloc::Eager` locks one arena of `Config::host_limit_bytes` when the
  first spillable tensor is registered, and never grows it.

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

## Limitations

- The budget is the manager's own accounting, not the driver's free-memory
  counter, and unmanaged allocations (here the two accumulators) are outside it.
- Single GPU, single thread, `f32` tensors only in the benchmark.
- Copying a modified tensor back to the host is still synchronous, and so is
  rebuilding a reconstructible one; only reloads from the host are prefetched.
- Page-locked host memory is unlocked only when the host limit needs the
  room for a block of another size, or when the manager is dropped. There
  is no call to give unused blocks back.
- The eager arena needs a host limit, and with tensors of different sizes
  it can fragment: a spill may give up clean copies while the arena still
  has enough free bytes in total, just not in one piece. Not measured yet;
  the benchmark's layers are all the same size.
- Eager allocation moves page-locking into setup but does not make it
  cheaper (about 0.3 ms per MiB either way).
- A failed prefetch on the copy stream is reported by the next
  `ensure_resident` of that tensor, which leaves it spilled.

## Building

Requires an NVIDIA GPU and CUDA toolkit supported by `cutile` 0.4. The cutile
crates are pinned exactly in the workspace `Cargo.toml` because they are pre-1.0.

```sh
cargo test -p tiermem   # runs on the GPU
```
