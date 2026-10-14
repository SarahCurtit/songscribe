// songscribe viewer: pick a grid, play it, compare chord timelines.
// No framework and no build step -- the server is stdlib-only and so is this.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const POLL_MS = 1000;
// Lane heights in px. The HTML label column is built from the same list, so
// the labels always line up with the SVG lanes.
const H = { ruler: 30, truth: 38, alt: 28, pred: 34, strip: 8, gap: 6 };
// Below this many px per bar, bar numbers would collide; show every Nth.
const MIN_BAR_LABEL_PX = 22;
// Follow mode keeps the playhead between these fractions of the visible width.
const FOLLOW_LEFT = 0.15;
const FOLLOW_RIGHT = 0.85;
const LEVEL_NAMES = {
  exact: "exact chord",
  majmin: "same triad",
  root: "same root",
  miss: "wrong",
};

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const svg = $("svg");
const scroller = $("scroller");

const state = {
  grids: [],
  backends: [],
  gridId: null,
  truth: null,
  selected: new Set(), // backends whose lane is shown
  predictions: {}, // backend -> latest status payload, for the current grid
  pxPerSec: Number($("zoom").value),
  playhead: null,
  loadToken: 0, // bumps on grid change, so stale polls are dropped
};

// --------------------------------------------------------------------------- //
// Data
// --------------------------------------------------------------------------- //

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function gridPath(id) {
  return id.split("/").map(encodeURIComponent).join("/");
}

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  return {
    grid: params.get("grid"),
    backends: (params.get("models") || "").split(",").filter(Boolean),
  };
}

function writeHash() {
  const params = new URLSearchParams();
  if (state.gridId) params.set("grid", state.gridId);
  if (state.selected.size) params.set("models", [...state.selected].join(","));
  history.replaceState(null, "", "#" + params.toString());
}

async function init() {
  try {
    [state.grids, state.backends] = await Promise.all([
      api("/api/grids"),
      api("/api/backends"),
    ]);
  } catch (error) {
    return showError(`Could not reach the server: ${error.message}`);
  }
  const wanted = readHash();
  wanted.backends.filter((b) => state.backends.includes(b)).forEach((b) => state.selected.add(b));

  const select = $("grid-select");
  for (const grid of state.grids) {
    const option = document.createElement("option");
    option.value = grid.id;
    const bpm = grid.bpm ? ` · ${Math.round(grid.bpm)} bpm` : "";
    option.textContent = `${grid.song} — grid ${grid.grid}${bpm}`;
    select.append(option);
  }
  select.addEventListener("change", () => loadGrid(select.value));

  buildBackendControls();
  const first = state.grids.find((g) => g.id === wanted.grid) || state.grids[0];
  if (first) {
    select.value = first.id;
    loadGrid(first.id);
  }
}

async function loadGrid(id) {
  const token = ++state.loadToken;
  state.gridId = id;
  state.truth = null;
  state.predictions = {};
  writeHash();
  hideError();

  const path = gridPath(id);
  audio.src = `/media/${path}/audio`;
  const grid = state.grids.find((g) => g.id === id);
  $("screenshot-panel").hidden = !grid.has_screenshot;
  if (grid.has_screenshot) $("screenshot").src = `/media/${path}/screenshot`;

  try {
    const truth = await api(`/api/grids/${path}`);
    if (token !== state.loadToken) return;
    state.truth = truth;
  } catch (error) {
    return showError(`Could not load the ground truth: ${error.message}`);
  }
  render();
  for (const backend of state.selected) requestPrediction(backend, false);
}

// Fetch a backend's prediction; with `run`, start it if nothing is cached.
async function requestPrediction(backend, run = true) {
  const token = state.loadToken;
  const url = `/api/grids/${gridPath(state.gridId)}/predict?backend=${encodeURIComponent(backend)}`;
  try {
    let reply = await api(url, run ? { method: "POST" } : undefined);
    if (!run && reply.status === "absent") reply = await api(url, { method: "POST" });
    while (reply.status === "running") {
      if (token !== state.loadToken || !state.selected.has(backend)) return;
      state.predictions[backend] = reply;
      updateBackendStatus();
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      reply = await api(url);
    }
    if (token !== state.loadToken) return;
    state.predictions[backend] = reply;
  } catch (error) {
    if (token !== state.loadToken) return;
    state.predictions[backend] = { status: "error", error: error.message };
  }
  updateBackendStatus();
  render();
}

// --------------------------------------------------------------------------- //
// Controls
// --------------------------------------------------------------------------- //

function buildBackendControls() {
  const box = $("backends");
  if (!state.backends.length) {
    box.append(Object.assign(document.createElement("span"), { textContent: "none registered" }));
    return;
  }
  for (const backend of state.backends) {
    const label = document.createElement("label");
    label.className = "backend";
    label.innerHTML = `<input type="checkbox"><span></span><span class="status"></span>`;
    label.dataset.backend = backend;
    label.children[1].textContent = backend;
    const input = label.querySelector("input");
    input.checked = state.selected.has(backend);
    input.addEventListener("change", () => {
      if (input.checked) {
        state.selected.add(backend);
        if (state.gridId) requestPrediction(backend);
      } else {
        state.selected.delete(backend);
      }
      writeHash();
      updateBackendStatus();
      render();
    });
    box.append(label);
  }
}

function updateBackendStatus() {
  for (const label of document.querySelectorAll(".backend")) {
    const backend = label.dataset.backend;
    const status = label.querySelector(".status");
    const reply = state.predictions[backend];
    status.classList.toggle("error", reply?.status === "error");
    status.title = reply?.error || "";
    if (!state.selected.has(backend) || !reply) status.textContent = "";
    else if (reply.status === "running") status.textContent = `running ${reply.elapsed ?? 0}s…`;
    else if (reply.status === "error") status.textContent = "failed (hover)";
    else if (reply.status === "done") status.textContent = `${reply.seconds}s`;
    else status.textContent = "";
  }
}

$("zoom").addEventListener("input", (event) => {
  const centre = (scroller.scrollLeft + scroller.clientWidth / 2) / state.pxPerSec;
  state.pxPerSec = Number(event.target.value);
  render();
  scroller.scrollLeft = centre * state.pxPerSec - scroller.clientWidth / 2;
});

scroller.addEventListener("click", (event) => {
  if (!state.truth) return;
  const x = event.clientX - svg.getBoundingClientRect().left;
  audio.currentTime = Math.max(0, x / state.pxPerSec);
});

document.addEventListener("keydown", (event) => {
  if (event.code !== "Space") return;
  if (event.target.closest("input, select, button, audio, textarea")) return;
  event.preventDefault();
  if (audio.paused) audio.play();
  else audio.pause();
});

audio.addEventListener("loadedmetadata", render);

// --------------------------------------------------------------------------- //
// Timeline
// --------------------------------------------------------------------------- //

function el(name, attributes = {}, parent = svg) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  parent.append(node);
  return node;
}

function duration() {
  const t = state.truth;
  if (Number.isFinite(audio.duration) && audio.duration > 0) return audio.duration;
  return (t && (t.audio_seconds || t.annotated_end)) || 0;
}

function lanes() {
  const list = [
    { kind: "ruler", height: H.ruler, label: "" },
    { kind: "truth", height: H.truth, label: "Ground truth" },
  ];
  if (state.truth?.alternates.length) list.push({ kind: "alt", height: H.alt, label: "Alternates" });
  for (const backend of state.backends) {
    if (state.selected.has(backend)) {
      list.push({ kind: "pred", backend, height: H.pred + H.strip + H.gap, label: backend });
    }
  }
  return list;
}

function fmt(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds - m * 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
}

function render() {
  const t = state.truth;
  svg.replaceChildren();
  const labels = $("lane-labels");
  labels.replaceChildren();
  if (!t) return;

  const px = state.pxPerSec;
  const width = Math.max(Math.ceil(duration() * px), scroller.clientWidth);
  const layout = lanes();
  const height = layout.reduce((sum, lane) => sum + lane.height, 0);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);

  // Unannotated tail: shaded, since nothing there is scored.
  if (t.annotated_end < duration()) {
    el("rect", {
      class: "tail",
      x: t.annotated_end * px,
      y: H.ruler,
      width: (duration() - t.annotated_end) * px,
      height: height - H.ruler,
    });
  }
  drawGrid(t, px, height);

  let y = 0;
  for (const lane of layout) {
    const div = document.createElement("div");
    div.style.height = `${lane.height}px`;
    div.textContent = lane.label;
    labels.append(div);

    if (lane.kind === "truth") {
      drawBlocks(t.events, y + 4, H.truth - 8, px, (e) => [e.short, e.label]);
    } else if (lane.kind === "alt") {
      drawBlocks(t.alternates, y + 4, H.alt - 8, px, (e) => [e.short, e.label], "alt");
    } else if (lane.kind === "pred") {
      const reply = state.predictions[lane.backend];
      if (reply?.status === "done") {
        drawBlocks(reply.events, y + 4, H.pred - 6, px, (e) => [e.label, null]);
        drawAgreement(reply.agreement.segments, y + H.pred, px);
        const score = document.createElement("span");
        score.className = "score";
        const totals = reply.agreement.totals;
        score.textContent = totals.root === null ? "" : pct(totals);
        score.title = "Share of annotated time: exact chord / same triad / same root";
        div.append(score);
      }
    }
    y += lane.height;
  }

  state.playhead = el("line", { class: "playhead", y1: 0, y2: height, x1: 0, x2: 0 });
  updatePlayhead();
}

function pct(totals) {
  const p = (x) => `${Math.round(100 * x)}%`;
  return `${p(totals.exact)} · ${p(totals.majmin)} · ${p(totals.root)}`;
}

function drawGrid(t, px, height) {
  const sectionStarts = new Set(t.sections.map((s) => s.start));
  const barWidth = t.bars.length > 1 ? (t.bars[1] - t.bars[0]) * px : Infinity;
  const every = Math.max(1, Math.ceil(MIN_BAR_LABEL_PX / barWidth));
  t.bars.forEach((start, index) => {
    const x = start * px;
    const section = sectionStarts.has(start);
    el("line", {
      class: section ? "barline section" : "barline",
      x1: x,
      x2: x,
      y1: H.ruler - 8,
      y2: height,
    });
    if (index < t.bars.length - 1 && index % every === 0) {
      el("text", { class: "tick", x: x + 3, y: H.ruler - 10 }).textContent = index + 1;
    }
  });
  for (const section of t.sections) {
    el("text", { class: "section-name", x: section.start * px + 3, y: 11 }).textContent =
      `[${section.name}]`;
  }
  if (!t.bars.length) {
    // No bar grid in this annotation: fall back to a seconds ruler.
    const step = px >= 40 ? 1 : 5;
    for (let s = 0; s <= duration(); s += step) {
      el("text", { class: "tick", x: s * px + 3, y: H.ruler - 10 }).textContent = `${s}s`;
      el("line", { class: "barline", x1: s * px, x2: s * px, y1: H.ruler - 8, y2: H.ruler });
    }
  }
}

function drawBlocks(events, y, height, px, labelOf, variant = "") {
  for (const event of events) {
    const [text, detail] = labelOf(event);
    const x = event.start * px;
    const w = Math.max((event.end - event.start) * px, 1);
    const noChord = text === "N";
    const rect = el("rect", {
      class: `block ${variant} ${noChord ? "no-chord" : ""}`.trim(),
      x,
      y,
      width: w,
      height,
      rx: 4,
    });
    const tip = `<b>${escape(text)}</b>${detail && detail !== text ? ` <span class="muted">${escape(detail)}</span>` : ""}<br><span class="muted">${fmt(event.start)} – ${fmt(event.end)}</span>`;
    hoverable(rect, tip);
    if (!noChord && w > 18) {
      const label = el("text", {
        class: `label ${variant}`.trim(),
        x: x + 6,
        y: y + height / 2 + 4,
      });
      label.textContent = text;
      // Clip by hand: an SVG clipPath per block is far heavier at this count.
      const room = w - 10;
      while (label.textContent.length > 1 && label.getComputedTextLength() > room) {
        label.textContent = label.textContent.slice(0, -2) + "…";
      }
    }
  }
}

function drawAgreement(segments, y, px) {
  for (const segment of segments) {
    const rect = el("rect", {
      class: `agree ${segment.level}`,
      x: segment.start * px,
      y,
      width: Math.max((segment.end - segment.start) * px, 1),
      height: H.strip,
    });
    hoverable(
      rect,
      `${LEVEL_NAMES[segment.level]}<br><span class="muted">${fmt(segment.start)} – ${fmt(segment.end)}</span>`,
    );
  }
}

function escape(text) {
  const span = document.createElement("span");
  span.textContent = text ?? "";
  return span.innerHTML;
}

// --------------------------------------------------------------------------- //
// Tooltip, playhead and "now" readout
// --------------------------------------------------------------------------- //

const tooltip = $("tooltip");

function hoverable(node, html) {
  node.addEventListener("mouseenter", () => {
    tooltip.innerHTML = html;
    tooltip.hidden = false;
  });
  node.addEventListener("mousemove", (event) => {
    const pad = 12;
    const box = tooltip.getBoundingClientRect();
    const left = Math.min(event.clientX + pad, window.innerWidth - box.width - pad);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${event.clientY - box.height - pad}px`;
  });
  node.addEventListener("mouseleave", () => {
    tooltip.hidden = true;
  });
}

function at(events, time) {
  return events?.find((e) => e.start <= time && time < e.end) || null;
}

function updatePlayhead() {
  const time = audio.currentTime || 0;
  const x = time * state.pxPerSec;
  if (state.playhead) {
    state.playhead.setAttribute("x1", x);
    state.playhead.setAttribute("x2", x);
  }
  if ($("follow").checked && !audio.paused) {
    const left = scroller.scrollLeft;
    const visible = scroller.clientWidth;
    if (x < left + FOLLOW_LEFT * visible || x > left + FOLLOW_RIGHT * visible) {
      scroller.scrollLeft = x - FOLLOW_LEFT * visible;
    }
  }
  updateNow(time);
}

function card(name, chord, detail) {
  return `<div class="card"><div class="name">${escape(name)}</div><div class="chord">${escape(chord || "—")}</div><div class="detail">${escape(detail || "")}</div></div>`;
}

let lastNow = "";
function updateNow(time) {
  const t = state.truth;
  if (!t) return;
  const truth = at(t.events, time);
  const cards = [
    card("Ground truth", truth?.short, truth && truth.label !== truth.short ? truth.label : ""),
  ];
  if (t.alternates.length) {
    const alt = at(t.alternates, time);
    cards.push(card("Alternate", alt?.short, alt && alt.label !== alt.short ? alt.label : ""));
  }
  for (const backend of state.backends) {
    if (!state.selected.has(backend)) continue;
    const reply = state.predictions[backend];
    const event = reply?.status === "done" ? at(reply.events, time) : null;
    const level = reply?.status === "done" ? at(reply.agreement.segments, time) : null;
    cards.push(card(backend, event?.label, level ? LEVEL_NAMES[level.level] : ""));
  }
  const html = cards.join("");
  if (html !== lastNow) {
    $("now").innerHTML = html;
    lastNow = html;
  }
}

function tick() {
  updatePlayhead();
  requestAnimationFrame(tick);
}

// --------------------------------------------------------------------------- //

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function hideError() {
  $("error").hidden = true;
}

function updateZoomLabel() {
  $("zoom-value").textContent = `${state.pxPerSec} px/s`;
}
$("zoom").addEventListener("input", updateZoomLabel);
updateZoomLabel();

window.addEventListener("resize", render);
requestAnimationFrame(tick);
init();
