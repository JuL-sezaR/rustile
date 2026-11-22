/*
  rustile landing page: small enhancements, nothing here is needed to read the page.
  The page works with this file missing. No network calls, no storage, no cookies.
*/

/* ---- Eviction simulation (pure, no DOM) -------------------------------------------------
   The benchmark's access pattern: 12 layers swept in order, 3 passes, room for 4 in VRAM.
   Registering 12 layers under a 4 layer budget leaves the last four resident (8 to 11).
   'lru' evicts the least recently used layer. 'belady' evicts the resident layer whose next
   declared use is farthest away (or never), which is what Policy::EvictLookahead does. */
const SIM_LAYERS = 12, SIM_SLOTS = 4, SIM_EPOCHS = 3;

function simulate(policy) {
  const plan = [];
  for (let e = 0; e < SIM_EPOCHS; e++) for (let i = 0; i < SIM_LAYERS; i++) plan.push(i);

  const slots = [];
  for (let i = SIM_LAYERS - SIM_SLOTS; i < SIM_LAYERS; i++) slots.push(i);
  const recency = new Map(slots.map((layer, k) => [layer, k - SIM_SLOTS])); // 8 is oldest

  const steps = [];
  let misses = 0;
  plan.forEach((id, t) => {
    let slot = slots.indexOf(id);
    const hit = slot >= 0;
    let evicted = null;
    if (!hit) {
      misses++;
      let victim = 0, best = null;
      slots.forEach((layer, k) => {
        let score;
        if (policy === 'lru') score = -recency.get(layer);            // oldest use scores highest
        else { const next = plan.indexOf(layer, t + 1); score = next < 0 ? Infinity : next; }
        if (best === null || score > best) { best = score; victim = k; }
      });
      slot = victim;
      evicted = slots[slot];
      recency.delete(evicted);
      slots[slot] = id;
    }
    recency.set(id, t);
    steps.push({ id, hit, evicted, slot, slots: slots.slice(), misses });
  });
  return { steps, misses, initial: Array.from({ length: SIM_SLOTS }, (_, k) => SIM_LAYERS - SIM_SLOTS + k) };
}

(() => {
  'use strict';
  if (typeof document === 'undefined') return; // lets the simulation be tested in Node

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const reduceMotion = window.matchMedia ? matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

  /* ---- Wordmark: still by default, animated only when motion is welcome, always pausable ---- */
  function wordmark() {
    const img = $('#wm'), btn = $('#wm-toggle');
    if (!img || !btn) return;
    const label = $('span', btn), icon = $('use', btn);
    let playing = false;
    const set = (on) => {
      playing = on;
      img.src = on ? img.dataset.anim : img.dataset.still;
      label.textContent = on ? 'Pause animation' : 'Play animation';
      icon.setAttribute('href', on ? '#i-pause' : '#i-play');
    };
    const apply = () => {
      if (reduceMotion.matches) { btn.hidden = true; set(false); }
      else { btn.hidden = false; set(true); }
    };
    btn.addEventListener('click', () => set(!playing));
    if (reduceMotion.addEventListener) reduceMotion.addEventListener('change', apply);
    apply();
  }

  /* ---- Navigation: mark the section being read ---- */
  function spy() {
    if (!('IntersectionObserver' in window)) return;
    const links = $$('.nav a');
    const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
    const targets = ['top', ...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => a.removeAttribute('aria-current'));
        const link = byId.get(entry.target.id);
        if (link) link.setAttribute('aria-current', 'true');
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    targets.forEach((t) => io.observe(t));
  }

  /* ---- Tagline: split into words so CSS can light them one at a time on scroll ---- */
  function tagline() {
    const el = $('#tagline');
    if (!el) return;
    const words = el.textContent.trim().split(/\s+/);
    el.textContent = '';
    words.forEach((word, i) => {
      const span = document.createElement('span');
      span.className = 'w';
      span.style.setProperty('--i', i);
      span.textContent = word;
      el.append(span);
      if (i < words.length - 1) el.append(' ');
    });
    el.classList.add('is-split');
  }

  /* ---- Simulator ---- */
  function simulator() {
    const root = $('#sim');
    if (!root) return;
    const ui = $('.sim-ui', root), fallback = $('.sim-static', root);
    const range = $('#sim-scrub'), out = $('#sim-now'), play = $('#sim-play');
    const total = SIM_LAYERS * SIM_EPOCHS;
    const data = { lru: simulate('lru'), belady: simulate('belady') };

    const panels = $$('.panel', root).map((el) => ({
      policy: el.dataset.policy,
      count: $('.count', el),
      strip: $('[data-strip]', el),
      slotsEl: $('[data-slots]', el),
      note: $('[data-note]', el),
    }));

    panels.forEach((p) => {
      for (let i = 0; i < total; i++) {
        const li = document.createElement('li');
        li.textContent = i % SIM_LAYERS;
        p.strip.append(li);
      }
      const label = document.createElement('p');
      label.className = 'slots-label';
      label.textContent = 'In VRAM, 4 slots. The outlined tile above a slot is the layer that just left it.';
      p.slotsEl.before(label);
      p.lane = document.createElement('div');
      p.lane.className = 'lane';
      p.lane.setAttribute('aria-hidden', 'true');
      for (let k = 0; k < SIM_SLOTS; k++) p.lane.append(document.createElement('div'));
      p.slotsEl.before(p.lane);
      for (let k = 0; k < SIM_SLOTS; k++) p.slotsEl.append(Object.assign(document.createElement('div'), { className: 'slot' }));
    });

    function render(n) {
      out.textContent = n;
      range.value = n;
      range.setAttribute('aria-valuetext', `Access ${n} of ${total}`);
      panels.forEach((p) => {
        const d = data[p.policy];
        const step = n > 0 ? d.steps[n - 1] : null;
        Array.from(p.strip.children).forEach((li, i) => {
          li.className = i < n ? (d.steps[i].hit ? 'hit' : 'miss') : '';
          if (i === n - 1) li.classList.add('now');
        });
        const layers = step ? step.slots : d.initial;
        Array.from(p.slotsEl.children).forEach((slot, k) => {
          slot.textContent = layers[k];
          slot.className = 'slot' + (step && step.slot === k ? (step.hit ? ' hit' : ' miss') : '');
        });
        Array.from(p.lane.children).forEach((cell, k) => {
          cell.replaceChildren();
          if (step && step.evicted !== null && step.slot === k) {
            cell.append(Object.assign(document.createElement('div'), { className: 'gone', textContent: step.evicted }));
          }
        });
        const num = document.createElement('span');
        num.textContent = step ? step.misses : 0;
        p.count.replaceChildren(num, n === total ? ` of ${total} accesses miss` : ` of ${n} accesses so far miss`);
        p.note.textContent = step
          ? `Access ${n}: layer ${step.id}, ${step.hit ? 'hit' : 'miss'}${step.evicted !== null ? `, evicted layer ${step.evicted}` : ''}.`
          : 'Start: layers 8 to 11 are in VRAM.';
      });
    }

    let timer = null;
    const setPlaying = (on) => {
      $('span', play).textContent = on ? 'Pause' : 'Play';
      $('use', play).setAttribute('href', on ? '#i-pause' : '#i-play');
    };
    const stop = () => { clearInterval(timer); timer = null; setPlaying(false); };
    play.addEventListener('click', () => {
      if (timer) return stop();
      if (+range.value >= total) render(0);
      setPlaying(true);
      timer = setInterval(() => {
        const n = +range.value + 1;
        render(n);
        if (n >= total) stop();
      }, 450);
    });
    range.addEventListener('input', () => { stop(); render(+range.value); });

    fallback.hidden = true;
    ui.hidden = false;
    render(total);
  }

  /* ---- Tile bars: one square per half gigabyte per second ---- */
  function tilebars() {
    $$('.tilebar').forEach((el) => {
      const totalTiles = +el.dataset.total, filled = +el.dataset.filled;
      for (let i = 0; i < totalTiles; i++) el.append(Object.assign(document.createElement('i'), { className: i < filled ? 'on' : '' }));
    });
  }

  /* ---- Copy buttons: Invoker Commands where supported, a click handler where not ---- */
  const copyTimers = new WeakMap();
  let lastMessage = '';

  function legacyCopy(text) {
    const field = Object.assign(document.createElement('textarea'), { value: text });
    field.setAttribute('readonly', '');
    field.style.cssText = 'position:fixed;inset-block-start:0;opacity:0';
    document.body.append(field);
    field.select();
    try { return document.execCommand('copy'); } finally { field.remove(); }
  }

  function report(button, ok) {
    const tip = button.nextElementSibling;
    const icon = $('use', button);
    clearTimeout(copyTimers.get(button));
    button.removeAttribute('data-copied');
    button.removeAttribute('data-failed');
    let message = ok ? 'Copied' : 'Copy failed';
    if (message === lastMessage) message += ' '; // VoiceOver skips an identical repeat
    lastMessage = message;
    tip.textContent = '';
    button.setAttribute(ok ? 'data-copied' : 'data-failed', '');
    if (ok && icon) icon.setAttribute('href', '#i-check');
    requestAnimationFrame(() => { tip.textContent = message; });
    copyTimers.set(button, setTimeout(() => {
      button.removeAttribute('data-copied');
      button.removeAttribute('data-failed');
      if (icon) icon.setAttribute('href', '#i-copy');
      tip.textContent = '';
    }, 2000));
  }

  async function copy(button, target) {
    const text = target.matches('input, textarea') ? target.value : target.innerText.trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      report(button, true);
    } catch {
      report(button, legacyCopy(text));
    }
  }

  function copyButtons() {
    if ('CommandEvent' in window) {
      document.addEventListener('command', (event) => {
        if (event.command === '--copy') copy(event.source, event.target);
      }, true);
    } else {
      document.addEventListener('click', (event) => {
        const button = event.target.closest('button[command="--copy"]');
        if (!button) return;
        const target = document.getElementById(button.getAttribute('commandfor'));
        if (target) copy(button, target);
      });
    }
  }

  wordmark();
  spy();
  tagline();
  simulator();
  tilebars();
  copyButtons();
})();
