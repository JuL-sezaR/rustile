---
version: 1
slug: "index-html"
primary_target: "index.html"
related_targets: []
---

# Surface brief: rustile landing page

Scope: one static landing page, `docs/site/index.html`. Visitor mode: Persuade.

Audience, job, action: ML practitioners and Rust or GPU engineers who run workloads larger than their VRAM. They need to learn what tiermem is, believe it works, see which milestones are done, and go to the GitHub repository (the one primary action). Proof: README benchmark tables with hardware and date, plus a live simulation of the real eviction policy. Constraints: no framework, no build step, self-hosted fonts, `prefers-reduced-motion` respected, looping motion pausable.

Unresolved: the GitHub Pages URL (assumed `jul-sezar.github.io/rustile`) used for canonical and og:image; no custom 404 page yet.

## Direction contract

THESIS: The page is a concrete slab built from square tiles. The animated wordmark's loose tile and the evicted layer in the simulator are the same object. It refuses the category-default AI-infrastructure page: dark gradient, glowing cards, a row of hero metrics.

OWN-WORLD: Hard rectangular plates in formwork (#121211) and concrete (#CFCBC2); rust (#C23A17) only on concrete, rust-hot (#E8601F) only on formwork; one drenched rust plate for the tagline and one for the close. Big Shoulders Display 800 headings in sentence case, JetBrains Mono for text and data. Square tiles as bullets, status marks and simulator cells. Ruled ledgers instead of cards. No radius, shadow or gradient anywhere.

STORY: The visitor sees an oversized animated wordmark, the headline "Run a working set larger than your VRAM" and a ledger of strict, LRU and lookahead results. They scroll into a live Belady simulation that reproduces 36 and 27 misses, read plain milestone status including what is on hold, copy the clone and run commands, and click View on GitHub.

FIRST VIEWPORT: 56px header bar (mark, four anchors, GitHub). The animated wordmark spans the 12 column grid at about 1200px wide. Below it, columns 1 to 7 hold the headline (6rem maximum), a mono sentence and the rust-hot View on GitHub button; columns 8 to 12 hold the results ledger. The primary action is visible at 1440 x 900 and at 390 wide without scrolling past the wordmark.

FORM: Pinned by the brief, so no roll. seed: none (pinned) "brutalist/industrial direction (hard grids, square tiles, the two rust tones by background, Big Shoulders Display and JetBrains Mono)" and "a single landing page for now, with sections for what tiermem is, the Belady-eviction and pinned-memory design, the current milestone status, and install/quick start".

Signature interaction: a two-panel simulator (LRU against lookahead) over 12 layers, 4 slots, 3 epochs, with a scrubber, that computes 36 of 36 and 27 of 36 misses live. Motion grammar: one authored moment (the wordmark) plus a word by word tagline reveal; everything else is static.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
