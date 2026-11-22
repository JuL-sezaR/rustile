---
name: rustile
description: A concrete slab of square tiles in formwork, concrete and two rusts, set in Big Shoulders Display and JetBrains Mono.
colors:
  formwork: "#121211"
  concrete: "#CFCBC2"
  slab: "#8F8B82"
  rust: "#C23A17"
  rust-hot: "#E8601F"
  formwork-soft-text: "#9A9790"
  formwork-rule: "#575550"
  concrete-soft-text: "#434240"
typography:
  display:
    fontFamily: "Big Shoulders Display, Arial Narrow, system-ui, sans-serif"
    fontSize: "6rem"
    fontWeight: 800
    lineHeight: 1
  headline:
    fontFamily: "Big Shoulders Display, Arial Narrow, system-ui, sans-serif"
    fontSize: "3.75rem"
    fontWeight: 800
    lineHeight: 1
  title:
    fontFamily: "Big Shoulders Display, Arial Narrow, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.1
  body:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.5
rounded:
  none: "0"
spacing:
  s-50: "0.25rem"
  s-75: "0.5rem"
  s-100: "0.75rem"
  s-200: "1rem"
  s-300: "1.5rem"
  s-400: "2rem"
  s-500: "2.5rem"
  s-600: "3rem"
  s-700: "4rem"
  s-800: "5rem"
  s-900: "6rem"
components:
  button-primary:
    backgroundColor: "{colors.rust-hot}"
    textColor: "{colors.formwork}"
    rounded: "{rounded.none}"
    padding: "0.75rem 1.5rem"
    height: "3rem"
    typography: "{typography.body}"
  button-primary-hover:
    backgroundColor: "{colors.concrete}"
    textColor: "{colors.formwork}"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.concrete}"
    rounded: "{rounded.none}"
    padding: "0.75rem 1.5rem"
    height: "3rem"
  button-small:
    rounded: "{rounded.none}"
    padding: "0.5rem 1rem"
    height: "2.5rem"
  panel:
    backgroundColor: "#1D1D1B"
    textColor: "{colors.concrete}"
    rounded: "{rounded.none}"
    padding: "1.5rem"
  command-block:
    backgroundColor: "#1C1C1B"
    textColor: "{colors.concrete}"
    rounded: "{rounded.none}"
  copy-button:
    textColor: "{colors.concrete}"
    rounded: "{rounded.none}"
    padding: "0.25rem 0.75rem"
    height: "2.25rem"
  tile-hit:
    backgroundColor: "{colors.concrete}"
    textColor: "{colors.formwork}"
    rounded: "{rounded.none}"
  tile-miss:
    backgroundColor: "{colors.rust-hot}"
    textColor: "{colors.formwork}"
    rounded: "{rounded.none}"
---

# Design System: rustile

## Overview

**Creative North Star: "The Tile Slab"**

The page is a slab poured from square tiles. Hard rectangular plates of formwork, concrete and rust stack down the page, each one a full-bleed ground that sets every colour sitting on it. Content is ruled, not carded: ledgers, a ruled feature grid, a status timeline and command blocks, separated by one-pixel and two-pixel lines. The same tile is bullet, status mark, simulator cell and wordmark piece, so the loose tile in the hero and an evicted layer in the simulator read as one object.

Density is calm and technical. Headings are tall, condensed and heavy in sentence case; everything else is monospaced, which makes data, prose and commands feel like one working document. Depth is never faked. There is no radius, shadow or gradient anywhere in the shipped build.

The brand kit is binding: formwork, concrete, slab, rust and rust-hot are fixed, tiles are always square, and the wordmark stays on its tile grid.

**Key Characteristics:**
- Full-width plates in dark, light and rust, alternating down the page.
- Square tiles (0 radius) as bullets, marks, cells and buttons.
- Ruled structure (1px and 2px lines) in place of cards and shadows.
- Condensed 800-weight display headings over an all-mono body.
- Fixed spacing and type step scales; nothing is sized off-scale.
- One authored motion moment plus one scroll-driven word reveal; everything else is static.

## Colors

A five-colour brand palette used strictly by ground: two neutrals, one decorative grey, and two rusts that never swap grounds.

### Primary
- **Rust Hot** (#E8601F): the accent on formwork only. Primary button fill, simulator miss cells, step numbers, active tile bars, tooltips, focus ring, caret, selection and range thumb on dark plates. Text selection on the light and rust plates is formwork with concrete text, so rust-hot never lands on concrete or rust.
- **Rust** (#C23A17): the accent on concrete only, and the drenched ground of the tagline plate and the close plate. On the light plate it colours link underlines, bullet ticks and status marks.

### Neutral
- **Formwork** (#121211): the dark ground, the text colour on concrete and rust, and the ink for anything sitting on rust-hot.
- **Concrete** (#CFCBC2): the light ground, the text colour on formwork, hit cells and outline buttons on dark.
- **Slab** (#8F8B82): decoration only. It is mixed into rule lines and the scrollbar; it is never a text or fill colour of its own.
- **Formwork soft text** (#9A9790): secondary text on dark plates, derived as concrete 72% over formwork.
- **Formwork rule** (#575550): hairlines on dark plates, derived as slab 55% over formwork.
- **Concrete soft text** (#434240): secondary text on light plates, derived as formwork 74% over concrete. Light-plate rules are full formwork.

### Named Rules
**The Plate Sets Everything Rule.** A plate class defines `--bg`, `--fg`, `--fg-2`, `--accent`, `--rule`, `--tile` and `--focus` together. Components read those variables and never hard-code a colour that depends on the ground behind them.

**The Two Rusts Rule.** Rust (#C23A17) appears only on concrete; rust-hot (#E8601F) appears only on formwork. On the rust plate the accent becomes formwork. A close tile inside the rust plate resets to the dark variables so rust-hot is legal inside it.

**The Slab Is Decoration Rule.** Slab (#8F8B82) never carries meaning, text or state.

## Typography

**Display Font:** Big Shoulders Display (with Arial Narrow, system-ui)
**Body Font:** JetBrains Mono (with ui-monospace, SFMono-Regular, Menlo, Consolas)
**Label/Mono Font:** JetBrains Mono

**Character:** A tall, tightly set industrial headline face over a workmanlike monospace. The pairing reads as signage over a spec sheet. Both are self-hosted variable fonts (weights 100-900 and 100-800).

### Hierarchy
- **Display** (800, clamp 3rem to 4.5rem, line-height 1; 6rem for the close title): the hero headline and the closing line.
- **Headline** (800, clamp 2.25rem to 3.75rem, line-height 1): section titles. The tagline on the rust plate uses 700 at the display clamp.
- **Sub-headline** (800, clamp 1.875rem to 2.25rem, line-height 1.05): block titles inside the design section.
- **Title** (700, 1.5rem, line-height 1.1): feature names, panel titles, status entries. Panel titles drop to 1.25rem.
- **Body** (400, 1rem, line-height 1.5, 60ch max): prose, sentence case. Lead paragraph is 1.125rem at 1.55.
- **Data and small** (400 to 500, 0.875rem for table text, notes and commands; 0.75rem for captions, table heads, legends and fine print): tabular numerals on counts and table figures.
- **Simulator numerals** (Display, 800, 2.25rem): miss counts and slot values.

### Named Rules
**The Sentence Case Rule.** All headings, labels and buttons are sentence case. There is no uppercase transform and no letter-spacing on any role.

**The Step Scale Rule.** Type sizes come from one fixed set (0.75, 0.875, 1, 1.125, 1.25, 1.5, 1.875, 2.25, 3, 3.75, 4.5, 6 rem). Display sizes are fluid clamps between two steps.

## Layout

A 90rem (1440px) maximum wrapper with a fluid gutter of clamp(1rem, 4vw, 3rem) holds a 12-column grid with 1.5rem column gap and 2rem row gap. Everything is single-column below the breakpoints and arranges into columns at 64rem: the hero splits 7 and 5 columns (copy and ledger), section intros split headline 1 to 6 and copy 8 to 12, and the design blocks split copy 1 to 5 against data 7 to 12. The feature grid steps from one to two columns at 40rem and three at 64rem, with a ruled gutter between cells. Breakpoints used: 40rem, 48rem, 56rem and 64rem.

Plates carry 3rem vertical padding, 5rem from 48rem; the tagline and close plates run 4rem and 6rem. Spacing is a fixed scale (0.125 to 6rem, twelve steps); gaps, paddings and margins are taken from it, with the tile bar's 2px gap and the 3px focus and now-marker outlines as the only exceptions. The header is a 3.5rem bar, sticky from 48rem, with the four anchors on a second row below it on narrow screens.

## Elevation & Depth

Flat. The system has no shadows, gradients, blur or radius. Depth comes from stacking opaque plates in different grounds, from tonal panels (concrete mixed 5 to 14 percent into formwork) and from ruled lines. Emphasis uses a 2px full-strength rule at the head and foot of a table or list, with 1px softer rules between rows.

### Named Rules
**The Flat Slab Rule.** Nothing floats. The only layered element is the copy tooltip, which is an opaque rust-hot tile.

## Shapes

Every shape is a right-angled rectangle or square: buttons, panels, tiles, slots, range thumbs, status marks and command blocks all use border-radius 0. Tiles are solid squares; the only outlines in the system are 2px outline buttons, 3px hold and idea status marks (solid and dashed), the 3px focus ring with 3px offset and the 3px "now" marker on the current simulator cell. Status marks are 1.5rem squares: filled for done, ringed for on hold, dashed for idea. Bullets are 0.5rem squares in the accent colour.

## Components

### Buttons
- **Shape:** square, 2px border, 3rem minimum height (2.5rem small), mono 700 at 1rem.
- **Primary:** rust-hot fill with formwork text, padding 0.75rem 1.5rem, trailing arrow icon. Used for View on GitHub on dark plates and inside the dark close tile.
- **Outline:** transparent with a 2px border in the plate foreground; the small variant (0.875rem) drives the simulator transport.
- **Hover / Focus:** hover inverts to plate-foreground fill with plate-background text (the primary turns concrete). 0.3s ease-out colour transition; active nudges 1px down and is disabled under reduced motion. Focus is a 3px outline in the plate focus colour.

### Header and navigation
A plate-coloured bar with a one-pixel rule beneath. Brand mark and name at the left, four text anchors centered, and a GitHub link at the right. Anchors are soft text; hover inverts; the current section is underlined in the accent colour with a 2px stroke.

### Ledger and data tables
Collapsed-border tables at 0.875rem. A 2px foreground rule above the head and below the last row, 1px rules between rows, small soft-text column heads, bold row heads and tabular right-aligned figures. A 0.75rem square tile in the row labels carries hit, miss and stop state.

### Ruled feature grid and crate list
Definition items separated by hairlines under a 2px top rule: 1, 2 or 3 columns by width. The crate list is a two-column ruled list (12rem name column in bold mono, description beside it).

### Simulator panels
Two tonal panels, LRU and lookahead, each with a count in display numerals, a 12-cell strip, and four slots. Hit cells are concrete, miss cells are rust-hot, the current cell has a 3px concrete outline. The layer just evicted is drawn as a lone hollow rust-hot tile lifted above the slot it left, the wordmark's loose tile made literal. A scrubber with a square rust-hot thumb and a play button control the run.

### Status timeline
Ruled list with a square state mark at the left, title and note in the centre, bold state text at the right from 48rem.

### Command blocks
Bordered tonal blocks with a caption bar (rust-hot numbered square step), a pre-formatted mono body and a square outline copy button whose opaque tooltip confirms or reports failure.

### Rust close
The closing plate is rust with formwork text, a large two-line display title and a formwork tile that carries the dark-plate variables, a short sentence and the primary button.

### Motion
One authored moment: the animated wordmark in the hero, pausable by a visible text button and still under reduced motion (the button hides there). One scroll-driven word reveal on the tagline, from 30 percent to full opacity, staggered per word, gated behind view-timeline support and no-preference motion. All transitions use the ease cubic-bezier(.22, 1, .36, 1) at 0.2 to 0.3s.

## Do's and Don'ts

### Do:
- **Do** put every new surface on a plate and read colour from the plate variables.
- **Do** use rust-hot (#E8601F) on formwork and rust (#C23A17) on concrete, and nowhere else.
- **Do** keep tiles square at radius 0, with state carried by fill, ring or dashed ring plus a text label.
- **Do** build structure from 1px and 2px rules instead of cards.
- **Do** take every space and type size from the fixed step scales.
- **Do** write headings and buttons in sentence case with Big Shoulders Display 800 for headings and JetBrains Mono for everything else.
- **Do** keep a visible pause for looping motion and make it still under prefers-reduced-motion.

### Don't:
- **Don't** add radius, box-shadow, gradient or blur to any element.
- **Don't** use slab (#8F8B82) as a text or fill colour.
- **Don't** place rust-hot on concrete or rust on formwork.
- **Don't** reach for the default AI-infrastructure look: a dark gradient page, glowing cards or a row of hero metrics.
- **Don't** convey state by colour alone; pair it with shape or text.

## Not canonized

The built page carries no kickers, eyebrows, glyph icons, hard offset shadows or system display faces, so there is nothing to exclude. The panel and command-block tints (#1D1D1B, #1C1C1B) are derived mixes of concrete into formwork, recorded as component values rather than named palette colours.
