// songscribe viewer: pick a recording, play it, compare what the models heard.
// No framework and no build step -- the server is stdlib-only and so is this.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const POLL_MS = 1000;
// The stages the server can run, in the order their lanes are stacked.
const STAGES = ["chords", "melody"];
// Lane heights in px. The HTML label column is built from the same list, so
// the labels always line up with the SVG lanes.
const H = { ruler: 30, truth: 38, alt: 28, pred: 34, strip: 8, gap: 6 };
// Staff geometry. `step` is half a staff space, which is what the server
// measures a note's position in, so one multiplication places a note head.
const STAFF = { step: 4, pad: 8, headRx: 5, headRy: 3.6, stem: 26, flagGap: 5 };
const STAFF_TOP_LINE = 8;
// Clear space above and below the five lines, for notes on ledger lines: eight
// steps is an octave either way, which is as far as an octave error goes.
const STAFF_HEIGHT = (2 * STAFF.pad + STAFF_TOP_LINE) * STAFF.step;
// Below this many px per bar, bar numbers would collide; show every Nth.
const MIN_BAR_LABEL_PX = 22;
// Follow mode keeps the playhead between these fractions of the visible width.
const FOLLOW_LEFT = 0.15;
const FOLLOW_RIGHT = 0.85;
// Characters of a backend's error shown inline before it is cut short.
const MAX_REASON = 60;
const LEVEL_NAMES = {
  exact: "exact chord",
  majmin: "same triad",
  root: "same root",
  miss: "wrong",
};
const NOTE_LEVEL_NAMES = {
  exact: "exact note",
  octave: "octave out",
  miss: "wrong note",
  extra: "not written",
  missing: "not transcribed",
};
const ACCIDENTALS = { "#": "♯", b: "♭" };
// The note each clef's bottom and top line carries, shown beside the lane's
// name so the staff can be read without relying on a clef glyph's font.
const CLEF_RANGE = { treble: "E4-F5", bass: "G2-A3" };

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const svg = $("svg");
const scroller = $("scroller");

const state = {
  grids: [],
  backends: { chords: [], melody: [] },
  gridId: null,
  truth: null,
  // Per stage: the backends whose lane is shown, and their latest replies.
  selected: { chords: new Set(), melody: new Set() },
  predictions: { chords: {}, melody: {} },
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
    // "models" is the chord stage's key, kept from when it was the only one.
    chords: (params.get("models") || "").split(",").filter(Boolean),
    melody: (params.get("melody") || "").split(",").filter(Boolean),
  };
}

function writeHash() {
  const params = new URLSearchParams();
  if (state.gridId) params.set("grid", state.gridId);
  if (state.selected.chords.size) params.set("models", [...state.selected.chords].join(","));
  if (state.selected.melody.size) params.set("melody", [...state.selected.melody].join(","));
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
  for (const stage of STAGES) {
    wanted[stage]
      .filter((b) => state.backends[stage].includes(b))
      .forEach((b) => state.selected[stage].add(b));
  }

  const select = $("grid-select");
  for (const grid of state.grids) {
    const option = document.createElement("option");
    option.value = grid.id;
    const bpm = grid.bpm ? ` · ${Math.round(grid.bpm)} bpm` : "";
    option.textContent = `${grid.song} — grid ${grid.grid}${bpm}`;
    select.append(option);
  }
  select.addEventListener("change", () => loadGrid(select.value));

  buildBackendControls("chords", "chord-backends");
  buildBackendControls("melody", "melody-backends");
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
  state.predictions = { chords: {}, melody: {} };
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
  for (const stage of STAGES) {
    for (const backend of state.selected[stage]) requestPrediction(stage, backend, false);
  }
}

// Fetch a backend's prediction; with `run`, start it if nothing is cached.
async function requestPrediction(stage, backend, run = true) {
  const token = state.loadToken;
  const url =
    `/api/grids/${gridPath(state.gridId)}/predict/${stage}` +
    `?backend=${encodeURIComponent(backend)}`;
  try {
    let reply = await api(url, run ? { method: "POST" } : undefined);
    if (!run && reply.status === "absent") reply = await api(url, { method: "POST" });
    while (reply.status === "running") {
      if (token !== state.loadToken || !state.selected[stage].has(backend)) return;
      state.predictions[stage][backend] = reply;
      updateBackendStatus();
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      reply = await api(url);
    }
    if (token !== state.loadToken) return;
    state.predictions[stage][backend] = reply;
  } catch (error) {
    if (token !== state.loadToken) return;
    state.predictions[stage][backend] = { status: "error", error: error.message };
  }
  updateBackendStatus();
  render();
}

// --------------------------------------------------------------------------- //
// Controls
// --------------------------------------------------------------------------- //

function buildBackendControls(stage, containerId) {
  const box = $(containerId);
  if (!state.backends[stage].length) {
    box.append(Object.assign(document.createElement("span"), { textContent: "none registered" }));
    return;
  }
  for (const backend of state.backends[stage]) {
    const label = document.createElement("label");
    label.className = "backend";
    label.innerHTML = `<input type="checkbox"><span></span><span class="status"></span>`;
    label.dataset.backend = backend;
    label.dataset.stage = stage;
    label.children[1].textContent = backend;
    const input = label.querySelector("input");
    input.checked = state.selected[stage].has(backend);
    input.addEventListener("change", () => {
      if (input.checked) {
        state.selected[stage].add(backend);
        if (state.gridId) requestPrediction(stage, backend);
      } else {
        state.selected[stage].delete(backend);
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
    const { backend, stage } = label.dataset;
    const status = label.querySelector(".status");
    const reply = state.predictions[stage][backend];
    status.classList.toggle("error", reply?.status === "error");
    status.title = reply?.error || "";
    if (!state.selected[stage].has(backend) || !reply) status.textContent = "";
    else if (reply.status === "running") status.textContent = `running ${reply.elapsed ?? 0}s…`;
    else if (reply.status === "error") status.textContent = failureReason(reply.error);
    else if (reply.status === "done") status.textContent = `${reply.seconds}s`;
    else status.textContent = "";
  }
}

// The short form shown beside the checkbox; the full message stays in the title.
function failureReason(error) {
  if (!error) return "failed";
  if (error.startsWith("NotImplementedError")) return "not implemented yet";
  const message = error.replace(/^\w+Error: /, "");
  return message.length > MAX_REASON ? `failed: ${message.slice(0, MAX_REASON)}…` : `failed: ${message}`;
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

// The reply for a staff lane, and the engraved score it should draw.
function staffData(lane) {
  if (lane.source === "written") return { score: state.truth.melody_score };
  const reply = state.predictions.melody[lane.backend];
  return reply?.status === "done" ? { score: reply.score, agreement: reply.agreement } : null;
}

function lanes() {
  const t = state.truth;
  const list = [{ kind: "ruler", height: H.ruler, label: "" }];
  if (t?.events.length) list.push({ kind: "truth", height: H.truth, label: "Ground truth" });
  if (t?.alternates.length) list.push({ kind: "alt", height: H.alt, label: "Alternates" });
  for (const backend of state.backends.chords) {
    if (state.selected.chords.has(backend)) {
      list.push({ kind: "pred", backend, height: H.pred + H.strip + H.gap, label: backend });
    }
  }
  const staff = STAFF_HEIGHT + H.gap;
  if (t?.melody_score) {
    list.push({ kind: "staff", source: "written", height: staff, label: "Melody (written)" });
  }
  for (const backend of state.backends.melody) {
    if (state.selected.melody.has(backend)) {
      list.push({ kind: "staff", source: "backend", backend, height: staff, label: backend });
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
  $("chord-legend").hidden = !layout.some((lane) => lane.kind === "pred");
  $("melody-legend").hidden = !layout.some((lane) => lane.kind === "staff" && lane.backend);

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
      const reply = state.predictions.chords[lane.backend];
      if (reply?.status === "done") {
        // Shown in the chart's spelling (Bb, not A#); the model's own label in the tooltip.
        drawBlocks(reply.events, y + 4, H.pred - 6, px, (e) => [e.display ?? e.label, e.label]);
        drawAgreement(reply.agreement.segments, y + H.pred, px);
        div.append(chordScore(reply.agreement.totals));
      }
    } else if (lane.kind === "staff") {
      const data = staffData(lane);
      if (data?.score) {
        drawStaff(data.score, y, px, width);
        div.append(staffNote(data));
      }
    }
    y += lane.height;
  }

  state.playhead = el("line", { class: "playhead", y1: 0, y2: height, x1: 0, x2: 0 });
  updatePlayhead();
}

function chordScore(totals) {
  const score = document.createElement("span");
  score.className = "score";
  const p = (x) => `${Math.round(100 * x)}%`;
  score.textContent =
    totals.root === null ? "" : `${p(totals.exact)} · ${p(totals.majmin)} · ${p(totals.root)}`;
  score.title =
    "Share of annotated time: exact chord / same triad / same root" +
    " (an alternate chord counts where the chart prints one)";
  return score;
}

// A staff lane's second line: the clef it is written in, and the melody score.
// Marked `staff` as well as `score` so a lane's figures can be told from a
// chord lane's, which are percentages of time rather than of notes.
function staffNote(data) {
  const note = document.createElement("span");
  note.className = "score staff";
  const clef = data.score.clef;
  const parts = [`${clef} ${CLEF_RANGE[clef] ?? ""}`.trim()];
  // Nothing measured the beat, so no note value was written.
  if (!data.score.beat_seconds) parts.push("rhythm unknown");
  const totals = data.agreement?.totals;
  if (totals && totals.exact !== null) {
    const p = (x) => `${Math.round(100 * x)}%`;
    parts.push(`${p(totals.exact)} · ${p(totals.octave)}`);
    // Transcribed against written. Several notes per written one is a held
    // note being split, which the percentages on their own cannot show.
    parts.push(`${totals.played_notes}/${totals.written_notes} notes`);
    if (totals.extra) parts.push(`+${totals.extra} unwritten`);
    note.title =
      "Share of the written notes: exact / right note in any octave." +
      " Then notes transcribed against notes written — more of the first means" +
      " a held note was split" +
      (totals.extra ? `; ${totals.extra} sounded where nothing was written` : "");
  } else if (!data.score.beat_seconds) {
    note.title = "No tempo, bar grid or .beats file, so note values are not written";
  }
  note.textContent = parts.join(" · ");
  return note;
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
      `${levelName(segment)}<br><span class="muted">${fmt(segment.start)} – ${fmt(segment.end)}</span>`,
    );
  }
}

function levelName(segment) {
  const name = LEVEL_NAMES[segment.level];
  return segment.alternate ? `${name} (alternate)` : name;
}

// --------------------------------------------------------------------------- //
// Staff
// --------------------------------------------------------------------------- //

// Where a staff step sits inside its lane. Step 0 is the bottom line and 8 the
// top one, so the y axis runs the other way from the step axis.
function stepY(step) {
  return (STAFF.pad + STAFF_TOP_LINE - step) * STAFF.step;
}

function drawStaff(score, laneY, px, width) {
  const staff = el("g", { class: "staff" });
  for (let step = 0; step <= STAFF_TOP_LINE; step += 2) {
    const y = laneY + stepY(step);
    el("line", { class: "staffline", x1: 0, x2: width, y1: y, y2: y }, staff);
  }
  for (const note of score.notes) drawNote(note, laneY, px, staff);
}

function drawNote(note, laneY, px, parent) {
  const x = note.start * px;
  const cy = laneY + stepY(note.step);
  const cx = x + STAFF.headRx;
  const group = el("g", { class: `note ${note.level || "plain"}` }, parent);

  // How long the note actually sounded. Proportional notation puts the head at
  // the onset but says nothing about the length, and a tracker's note lengths
  // are exactly what over- and under-segmentation shows up in.
  el(
    "rect",
    {
      class: "sounded",
      x,
      y: cy - 1.5,
      width: Math.max((note.end - note.start) * px, 1),
      height: 3,
      rx: 1.5,
    },
    group,
  );
  for (const step of note.ledger) {
    const y = laneY + stepY(step);
    el(
      "line",
      { class: "ledger", x1: cx - STAFF.headRx - 3, x2: cx + STAFF.headRx + 3, y1: y, y2: y },
      group,
    );
  }
  el(
    "ellipse",
    {
      class: note.filled ? "head filled" : "head",
      cx,
      cy,
      rx: STAFF.headRx,
      ry: STAFF.headRy,
      // Note heads are oval and tilted; upright ones read as dots.
      transform: `rotate(-20 ${cx} ${cy})`,
    },
    group,
  );
  if (note.stem) {
    const up = !note.stem_down;
    const sx = up ? cx + STAFF.headRx - 0.5 : cx - STAFF.headRx + 0.5;
    const tip = up ? cy - STAFF.stem : cy + STAFF.stem;
    el("line", { class: "stem", x1: sx, x2: sx, y1: cy, y2: tip }, group);
    // Flags curl back towards the head, so they follow the stem's direction.
    const towards = up ? 1 : -1;
    for (let flag = 0; flag < note.flags; flag++) {
      const from = tip + towards * flag * STAFF.flagGap;
      el(
        "path",
        { class: "flag", d: `M ${sx} ${from} q 6 ${3 * towards} 5 ${9 * towards}` },
        group,
      );
    }
  }
  if (note.accidental) {
    const glyph = el(
      "text",
      { class: "accidental", x: cx - STAFF.headRx - 2, y: cy + 4, "text-anchor": "end" },
      group,
    );
    glyph.textContent = ACCIDENTALS[note.accidental] ?? note.accidental;
  }
  for (let dot = 0; dot < note.dots; dot++) {
    el(
      "circle",
      { class: "dot", cx: cx + STAFF.headRx + 4 + dot * 4, cy, r: 1.5 },
      group,
    );
  }
  hoverable(group, noteTip(note));
}

function noteTip(note) {
  const written = note.value
    ? `${note.dots ? "dotted " : ""}${note.value}`
    : "rhythm unknown";
  const level = note.level ? `<br>${NOTE_LEVEL_NAMES[note.level] ?? note.level}` : "";
  return (
    `<b>${escape(note.name)}</b> <span class="muted">MIDI ${note.pitch} · ${escape(written)}` +
    `</span>${level}<br><span class="muted">${fmt(note.start)} – ${fmt(note.end)}</span>`
  );
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

function card(name, value, detail) {
  return `<div class="card"><div class="name">${escape(name)}</div><div class="chord">${escape(value || "—")}</div><div class="detail">${escape(detail || "")}</div></div>`;
}

let lastNow = "";
function updateNow(time) {
  const t = state.truth;
  if (!t) return;
  const cards = [];
  if (t.events.length) {
    const truth = at(t.events, time);
    cards.push(
      card("Ground truth", truth?.short, truth && truth.label !== truth.short ? truth.label : ""),
    );
  }
  if (t.alternates.length) {
    const alt = at(t.alternates, time);
    cards.push(card("Alternate", alt?.short, alt && alt.label !== alt.short ? alt.label : ""));
  }
  for (const backend of state.backends.chords) {
    if (!state.selected.chords.has(backend)) continue;
    const reply = state.predictions.chords[backend];
    const event = reply?.status === "done" ? at(reply.events, time) : null;
    const level = reply?.status === "done" ? at(reply.agreement.segments, time) : null;
    cards.push(card(backend, event?.display ?? event?.label, level ? levelName(level) : ""));
  }
  if (t.melody_score) {
    const note = at(t.melody_score.notes, time);
    cards.push(card("Melody (written)", note?.name, note ? `MIDI ${note.pitch}` : ""));
  }
  for (const backend of state.backends.melody) {
    if (!state.selected.melody.has(backend)) continue;
    const reply = state.predictions.melody[backend];
    const note = reply?.status === "done" ? at(reply.score.notes, time) : null;
    cards.push(
      card(backend, note?.name, note?.level ? NOTE_LEVEL_NAMES[note.level] ?? "" : ""),
    );
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
