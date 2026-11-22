# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Plain static HTML, CSS and a small JavaScript file. No framework and no build step. Deployed to GitHub Pages from `docs/site/`. (Confirmed in the brief: "Static site, no framework, deployable to GitHub Pages from docs/site/. Don't add a build step or a framework unless I ask.")

## Users

Rust and GPU engineers whose ML workloads are larger than the VRAM they have. Confirmed as broader than that: ML practitioners who do not write Rust but run into the same limit also count, so the page explains the problem in plain terms before it shows Rust.

They arrive from GitHub, a link or a search, want to know in seconds what tiermem is, whether it works, and what state it is in, then clone it and run the benchmark on an NVIDIA GPU.

## Product Purpose

rustile is a project and brand; its first library is `tiermem`, a VRAM budget manager for tensors of any element type. It keeps the total size of managed tensors under a byte budget. When room is needed it evicts lower priority tensors, spilling them to page-locked host RAM or dropping them if they can be rebuilt, instead of failing. The workload can declare its access order so eviction follows Belady's rule.

The wider plan is an AI-aware OS layer, user space first: tiered memory, recoverable out-of-memory handling, workload-aware scheduling, observability.

Success for the website: a visitor understands what tiermem does, sees the evidence, sees honestly which milestones are done, and goes to the GitHub repository.

## Positioning

User-space tiering with a declared access order (Belady eviction) and a pool of page-locked host memory on the manager's own copy streams, built on cuTile Rust, with no kernel, driver or cross-process work. The proof it can show: a 768 MiB working set (12 x 64 MiB layers) runs correctly under a 256 MiB budget, where the strict baseline stops with a recoverable `BudgetExceeded` at layer 4.

## Operating Context

- Built on `cutile` 0.4 (pinned exactly in the workspace `Cargo.toml`), Rust edition 2024, an NVIDIA GPU and a CUDA toolkit that cutile 0.4 supports.
- Measured on an RTX 5080 Laptop under WSL2. Run-to-run variance is about plus or minus 10 percent.
- Install means cloning `github.com/JuL-sezaR/rustile`. The crate is not published to crates.io.
- Workspace crates: `tiermem`, `tilekernels` (softmax, RMSNorm, embedding gather, linear), `vecadd`.

## Capabilities and Constraints

- Done and measured (Milestone 1): lookahead eviction, clean host copies, pinned host pool with a limit, prefetch and asynchronous write-back on copy streams, soft-limit pressure callbacks with hysteresis, recoverable budget errors with a snapshot, Chrome-trace timeline, any element type under one budget.
- On hold (Milestone 2): a real model through tiermem. The ARC 1.0 route is on hold (issue #4); the model is chosen after the basic tile kernels exist.
- Proposed, not built (Milestone 3): memory-aware model routing (NOVELTIES.md, issue #6).
- Not started: cgroups, eBPF, HMM and multi-process enforcement.
- Limits to state honestly: single GPU, single thread; the budget is the manager's own accounting; element type is checked at runtime (`WrongDType`); the benchmark GPU reports one async copy engine; `kernelbench` cases are not comparable to better than about 35 percent.
- Terminology: "spill" (copy to host RAM), "rebuild" (drop and regenerate), "lookahead" (Belady), "pinned" meaning page-locked host memory. Note that `Hint::Pinned` is a different thing: a tensor that is never evicted. The page must not blur the two.

## Brand Commitments

The rustile brand kit (the user's `rustile-brand.zip`): a lowercase r built from square tiles, the wordmark on a 4 x 5 tile grid, two accent tiles (the dot of the i and the loose tile) that move in the animated wordmark. Colours: formwork `#121211`, concrete `#CFCBC2`, slab `#8F8B82` (decoration only), rust `#C23A17` on light grounds only, rust-hot `#E8601F` on dark grounds only. Tiles are always square: no rounded corners, outlines, gradients or drop shadows. Fonts: Big Shoulders Display for headings, JetBrains Mono for code and text. Direction in the user's words: "brutalist/industrial direction (hard grids, square tiles, the two rust tones by background, Big Shoulders Display and JetBrains Mono)". Use the animated wordmark in the hero and respect `prefers-reduced-motion`.

Voice: plain and specific, as in the README. Numbers come with hardware and date. Ideas that are not measured stay out of claims.

## Evidence on Hand

- The repository README holds every benchmark table, with hardware (RTX 5080 Laptop, WSL2) and date. The page may quote these and no others.
- `NOVELTIES.md` and `docs/milestone-2-feasibility.md` for status.
- Real API names from `tiermem/src/lib.rs` and `tiermem/examples/overbudget.rs`.
- Absent, and not to be invented: customers, testimonials, logos, download counts, a crates.io release, pricing, a published website URL, benchmarks on other GPUs.

## Product Principles

1. Show the mechanism working before explaining it; the Belady simulation uses the real policy and the real access pattern.
2. Every number carries its hardware, date and source, or it does not appear.
3. Say what is not done as plainly as what is.
4. One action: go to the repository.
5. Keep ideas (NOVELTIES) apart from results, as the README does.

## Accessibility & Inclusion

Target WCAG 2.2 AA. Looping motion must be pausable and absent under `prefers-reduced-motion`; text resizes to 200 percent and reflows at 320 CSS px; keyboard operable throughout; text contrast is checked per ground, because rust on concrete (3.3:1) and concrete or formwork on rust (3.3 to 3.5:1) only pass for large text.
