# Novelties

Ideas that could make tiermem more than a VRAM budget manager. Nothing here is
implemented or measured yet, which is why it lives here and not in the README.
An idea moves to the README only when it has code, a benchmark, and results.

Each entry says what the idea is, why it might be new, how we would know it
works, and what could kill it.

Status labels: **proposed** (an idea), **feasibility** (checking whether it can
be built), **in progress**, **measured** (results exist, ready for the README),
**dropped** (with the reason kept below).

---

## 1. Memory-aware model routing

**Status:** proposed. Prior work not yet checked.

**Idea.** Decision models that route agent requests between local and cloud
models, such as [ARC 1.0](https://huggingface.co/rayline-ai/ARC-1.0), choose by
predicted success minus expected cost. Their cost term covers prices and token
usage, but not whether the local model fits in GPU memory right now. tiermem
knows that: what is resident, what a spill would cost, and how long reloads take
under the current budget. Feeding those numbers into the routing cost would let
a router run locally only when the model fits, or when the spill penalty is
worth paying, and otherwise fall back to the cloud.

**Why it might be new.** Routers treat the local machine as a fixed resource.
This makes the local cost depend on live memory pressure. Before claiming
novelty, search for existing memory-aware or load-aware LLM routers.

**What tiermem needs to expose.** A cheap, read-only estimate: "running model X
now would need N bytes resident, cost M bytes of migration, and take about T
ms of copies." This builds on `Stats` and the access plan, without changing
eviction.

**How we would know it works.** Under a varying VRAM budget, memory-aware
routing should keep task quality close to the plain router while avoiding the
slow cases where a local model thrashes. Report latency, quality and cost
against the router with memory pressure ignored.

**Depends on.** Milestone 2 (ARC running through tiermem), and a way to compute
ARC's routing utility outside the hosted API.

**Risks.**
- ARC's open-source serving path is unfinished (the vllm-arc plugin is a work
  in progress), so reproducing its policy locally may be hard.
- Any gain may only show up with several local models competing for VRAM,
  which is a bigger setup than one laptop GPU.

---

## 2. Branching access plans

**Status:** proposed.

**Idea.** `expect_accesses` takes a linear order. Real models fork: ARC's
specialized heads read its frozen lower twelve layers, while its question head
uses all twenty-four. The lower layers are shared by every path, so they are
worth keeping resident more than any single upper layer. An access plan that
can express a shared trunk with branches, and the probability of taking each
branch, would let eviction and prefetch account for that.

**Why it might be new.** Belady's rule assumes a known sequence. With branches,
the optimal choice depends on which path is likely. This is close to existing
work on caching with probabilistic futures, so check that literature first.

**How we would know it works.** Fewer misses and less copy time than
flattening the branches into one linear plan, on the ARC workload, under the
same budget.

**Depends on.** Milestone 2.

---

## 3. A decision model as a coarse policy selector

**Status:** proposed. Weakest of the three; kept so the reasoning is recorded.

**Idea.** Once per job, not per access, a small decision model picks tiermem's
configuration: spill or rebuild, prefetch depth, how to split the budget.

**Why it is weak.** A forward pass of even a 0.6B model takes about as long as
a reload, so it cannot run per access, and with a declared access order
Belady's rule is already optimal. This only helps when the access order is
unknown and the workload has to be classified first. Revisit only if a real
workload shows that case.
