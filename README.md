# vecadd_demo → tiermem

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
| [`tiermem`](tiermem/) | VRAM budget manager: residency hints, eviction (spill to host or rebuild), soft-limit pressure callbacks, recoverable budget errors with a diagnostic snapshot, Chrome-trace timeline. |
| [`vecadd`](vecadd/) | The original cuTile hello-world: a single tile kernel adding two vectors. |

## Milestone 1: run a working set larger than the budget

`tiermem/examples/overbudget.rs` streams 12 × 64 MiB "layers" (768 MiB)
through an add kernel, three passes, under a 256 MiB budget.

```sh
cargo run --release -p tiermem --example overbudget -- strict    # baseline: fails at layer 4
cargo run --release -p tiermem --example overbudget -- spill     # LRU, spill to host RAM
cargo run --release -p tiermem --example overbudget -- rebuild   # LRU, regenerate on device
cargo run --release -p tiermem --example overbudget -- spill --budget-mib 1024  # everything fits
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
  next (prefetch hints, or MRU eviction for scans).
- **Spilling re-copies unchanged weights.** Weights are read-only, yet each
  eviction copied them back to host RAM (2816 MiB of the traffic). Keeping a
  clean host copy for immutable tensors would make eviction just a free.
- **Migration is synchronous and goes through pageable memory.** Next step:
  pinned host buffers and a separate copy stream, so reloads overlap compute.
- **Pressure fires on every re-crossing** (45 times here). It needs
  hysteresis or rate limiting before it is useful to an application.

## Limitations

- The budget is the manager's own accounting, not the driver's free-memory
  counter, and unmanaged allocations (here the two accumulators) are outside it.
- Single GPU, single thread, `f32` tensors only in the benchmark.

## Building

Requires an NVIDIA GPU and CUDA toolkit supported by `cutile` 0.4. The cutile
crates are pinned exactly in the workspace `Cargo.toml` because they are pre-1.0.

```sh
cargo test -p tiermem   # runs on the GPU
```
