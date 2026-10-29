// squeaktest page. Plain JavaScript, no dependencies, no inline code (the page's Content
// Security Policy allows scripts from this origin only). Text from the server is always set
// with textContent, never parsed as HTML.
"use strict";

const BAND_LABELS = {
  likely_synthetic: "Likely synthetic",
  uncertain: "Uncertain",
  likely_genuine: "Likely genuine",
  not_assessed: "Not assessed",
};
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);
let selectedFile = null;
let maxFileBytes = null;

function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function selectFile(file) {
  selectedFile = file || null;
  $("analyze-button").disabled = !selectedFile;
  $("drop-text").textContent = selectedFile
    ? "Selected: " + selectedFile.name
    : "Choose an audio file, or drop it here";
  setStatus("");
}

async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    if (!response.ok) return;
    const info = await response.json();
    maxFileBytes = info.limits.max_file_mb * 1024 * 1024;
    const minutes = info.limits.max_duration_s / 60;
    const duration = minutes >= 1 ? minutes + " min" : info.limits.max_duration_s + " s";
    $("limits").textContent =
      info.limits.formats + ". Up to " + info.limits.max_file_mb + " MB and " + duration + ".";
    $("score-note").textContent = info.score_note;
    $("model-line").textContent =
      "Model: " + info.model.name + " @ " + info.model.revision.slice(0, 7) +
      " (" + info.model.license + "). squeaktest " + info.version + ".";
  } catch {
    // The page still works without the limits text.
  }
}

async function analyzeFile(event) {
  event.preventDefault();
  if (!selectedFile) return;
  if (maxFileBytes !== null && selectedFile.size > maxFileBytes) {
    setStatus("That file is larger than the size limit.", true);
    return;
  }
  $("analyze-button").disabled = true;
  $("result").hidden = true;
  setStatus("Analyzing. On the free server this can take up to a minute...");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: selectedFile, // the bytes only: no file name is sent
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      setStatus((body && body.message) || "Analysis failed (HTTP " + response.status + ").", true);
      return;
    }
    setStatus("");
    showResult(body);
  } catch {
    setStatus("Couldn't reach the server. Check your connection and try again.", true);
  } finally {
    $("analyze-button").disabled = !selectedFile;
  }
}

function bandOf(score, bands) {
  if (score === null) return "skipped";
  if (score >= bands.likely_synthetic_at) return "likely_synthetic";
  if (score < bands.likely_genuine_below) return "likely_genuine";
  return "uncertain";
}

function showResult(report) {
  const band = $("band");
  band.textContent = BAND_LABELS[report.band] || report.band;
  band.className = "band " + report.band;

  $("score-line").textContent = report.score === null
    ? report.note
    : "Score " + report.score.toFixed(2) + " (raw model output, not a probability).";

  const audio = report.audio;
  $("audio-line").textContent =
    audio.duration_s.toFixed(1) + " s of audio (" + audio.container + "/" + audio.codec + "), " +
    audio.active_s.toFixed(1) + " s with sound" +
    (audio.tones_s ? ", " + audio.tones_s.toFixed(1) + " s of steady tones left out." : ".");

  const warnings = $("warnings");
  warnings.replaceChildren();
  for (const warning of report.warnings) {
    const p = document.createElement("p");
    p.className = "warning";
    p.textContent = warning.message;
    warnings.append(p);
  }

  drawTimeline(report);
  fillTable(report.windows);
  $("disclaimer").textContent = report.disclaimer;
  $("result").hidden = false;
  $("result-heading").focus({ preventScroll: true });
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function svg(name, attributes, parent) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (parent) parent.append(element);
  return element;
}

function drawTimeline(report) {
  const container = $("timeline");
  container.replaceChildren();
  if (report.windows.length === 0) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "No windows to show: " + report.note + ".";
    container.append(p);
    return;
  }

  const width = 600, height = 170;
  const left = 34, right = 10, top = 10, bottom = 26;
  const plotW = width - left - right, plotH = height - top - bottom;
  const duration = Math.max(report.audio.duration_s, 0.001);
  const x = (t) => left + (Math.min(Math.max(t, 0), duration) / duration) * plotW;
  const y = (score) => top + (1 - score) * plotH;

  const chart = svg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Score for each part of the clip; the table below has the same values.",
  }, container);

  for (const [value, label] of [
    [report.bands.likely_synthetic_at, "likely synthetic"],
    [report.bands.likely_genuine_below, "likely genuine"],
  ]) {
    const line = svg("line", {
      class: "threshold", x1: left, x2: width - right, y1: y(value), y2: y(value),
    }, chart);
    svg("title", {}, line).textContent = `Threshold: ${label} (${value.toFixed(2)})`;
  }

  // Windows overlap (one every hop_s), so each bar reaches halfway to its neighbours'
  // centres, and the first and last bars reach the edges of the scored audio. The bars then
  // tile what was scored instead of drawing on top of each other.
  const windows = report.windows;
  const centres = windows.map((w) => (w.start_s + w.end_s) / 2);
  windows.forEach((w, i) => {
    const from = i === 0 ? w.start_s : (centres[i - 1] + centres[i]) / 2;
    const to = i === windows.length - 1 ? w.end_s : (centres[i] + centres[i + 1]) / 2;
    const x0 = x(from), x1 = x(to);
    const scored = w.score !== null;
    const top_y = scored ? y(w.score) : y(0) - 3;
    const bar = svg("rect", {
      class: "bar " + bandOf(w.score, report.bands),
      x: x0 + 0.5,
      y: top_y,
      width: Math.max(x1 - x0 - 1, 1),
      height: Math.max(y(0) - top_y, 1),
    }, chart);
    svg("title", {}, bar).textContent = scored
      ? `${w.start_s.toFixed(1)}-${w.end_s.toFixed(1)} s: score ${w.score.toFixed(2)}`
      : `${w.start_s.toFixed(1)}-${w.end_s.toFixed(1)} s: mostly silence or tones, not scored`;
  });

  svg("line", { class: "axis", x1: left, x2: width - right, y1: y(0), y2: y(0) }, chart);
  svg("line", { class: "axis", x1: left, x2: left, y1: top, y2: y(0) }, chart);
  const ticks = [
    [left - 6, y(1) + 4, "end", "1"],
    [left - 6, y(0) + 4, "end", "0"],
    [left, height - 8, "start", "0 s"],
    [width - right, height - 8, "end", duration.toFixed(1) + " s"],
  ];
  for (const [tx, ty, anchor, text] of ticks) {
    svg("text", { class: "tick", x: tx, y: ty, "text-anchor": anchor }, chart).textContent = text;
  }
}

function fillTable(windows) {
  const tbody = $("window-table").querySelector("tbody");
  tbody.replaceChildren();
  for (const w of windows) {
    const row = document.createElement("tr");
    for (const value of [
      w.start_s.toFixed(1),
      w.end_s.toFixed(1),
      w.score === null ? "not scored" : w.score.toFixed(2),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    tbody.append(row);
  }
}

function setUpDropZone() {
  const zone = $("drop-zone");
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    zone.addEventListener(type, () => zone.classList.remove("dragging"));
  }
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) selectFile(file);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  $("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
  $("upload-form").addEventListener("submit", analyzeFile);
  $("result-heading").tabIndex = -1;
  setUpDropZone();
  loadInfo();
});
