// Interactive Playground for the bindcurve models (docs/playground.md).

import { MODELS, SPECIES_LABELS, response, species } from "./models.js";

const N_POINTS = 300;
const LOG_SLIDER = { min: -2, max: 5, step: 0.01 }; // log10(nM)

const PARAMETER_HELP = {
  ymin: "Response of fully free tracer, or at high inhibitor concentration",
  ymax: "Response of fully bound tracer, or at zero inhibitor concentration",
  IC50: "Concentration at the midpoint of the curve",
  hill_slope: "Steepness of the curve",
  Kds: "Dissociation constant of the tracer",
  Kd: "Dissociation constant of the competitor",
  Kd3: "Dissociation constant of tracer from the competitor-bound receptor",
  RT: "Total receptor concentration",
  LsT: "Total tracer concentration",
  Ns: "Nonspecific binding of tracer, relative to free tracer",
  N: "Nonspecific binding of competitor, relative to free competitor",
};

const SPECIES_COLORS = {
  R: "#1f77b4",
  Ls: "#ff7f0e",
  L: "#2ca02c",
  RLs: "#d62728",
  RL: "#9467bd",
  RLLs: "#8c564b",
  Ls_nonspecific: "#e377c2",
  L_nonspecific: "#17becf",
};

// Default x range in decades around the model's characteristic constant.
const X_DECADES = { ic50: [-3, 3], direct: [-3, 3], competition: [-2, 4] };

// yAuto: let Plotly fit the response axis; otherwise it is fixed to [ylo, yhi].
const state = {
  model: "ic50",
  params: {},
  xmin: 0,
  xmax: 0,
  yAuto: true,
  ylo: 0,
  yhi: 0,
  kept: null,
};

const $ = (id) => document.getElementById(id);

function format(value) {
  if (!Number.isFinite(value)) return "";
  return String(Number.parseFloat(value.toPrecision(3)));
}

function logspace(lo, hi, n) {
  const a = Math.log10(lo);
  const b = Math.log10(hi);
  return Array.from({ length: n }, (_, i) => 10 ** (a + ((b - a) * i) / (n - 1)));
}

function family(name) {
  if (name === "ic50") return "ic50";
  return name.startsWith("dir_") ? "direct" : "competition";
}

function defaultRange() {
  const center = Math.log10(state.params[MODELS[state.model].marker]);
  const [below, above] = X_DECADES[family(state.model)];
  state.xmin = 10 ** Math.floor(center + below);
  state.xmax = 10 ** Math.ceil(center + above);
  $("pg-xmin").value = format(state.xmin);
  $("pg-xmax").value = format(state.xmax);
}

// --- Controls --------------------------------------------------------------

function buildModelSelect() {
  const select = $("pg-model");
  for (const [name, model] of Object.entries(MODELS)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name}: ${model.title}`;
    select.append(option);
  }
  select.addEventListener("change", () => selectModel(select.value));
}

// Physical limits of typed-in values; the plateaus can take any value.
function isAllowed(name, value) {
  if (name === "ymin" || name === "ymax") return true;
  if (name === "Ns" || name === "N") return value >= 0;
  return value > 0;
}

function parameterRow(p) {
  const row = document.createElement("div");
  row.className = "pg-param";

  const label = document.createElement("label");
  label.htmlFor = `pg-p-${p.name}`;
  label.title = PARAMETER_HELP[p.name] ?? "";
  label.innerHTML = `${p.label} <code>${p.name}</code>`;

  const range = document.createElement("input");
  range.type = "range";
  range.setAttribute("aria-label", p.name);
  const box = document.createElement("input");
  box.type = "number";
  box.id = `pg-p-${p.name}`;
  box.step = "any";

  if (p.log) {
    Object.assign(range, LOG_SLIDER);
    box.min = "0";
  } else {
    Object.assign(range, { min: p.min, max: p.max, step: p.step });
  }

  const show = (value) => {
    range.value = p.log ? Math.log10(value) : value;
    box.value = format(value);
  };
  show(state.params[p.name]);

  range.addEventListener("input", () => {
    const value = p.log ? 10 ** Number(range.value) : Number(range.value);
    state.params[p.name] = Number.parseFloat(value.toPrecision(4));
    box.value = format(state.params[p.name]);
    scheduleDraw();
  });
  box.addEventListener("change", () => {
    const value = Number(box.value);
    if (Number.isFinite(value) && isAllowed(p.name, value)) state.params[p.name] = value;
    show(state.params[p.name]);
    scheduleDraw();
  });

  row.append(label, range, box);
  return row;
}

function selectModel(name) {
  state.model = name;
  state.kept = null;
  const model = MODELS[name];
  state.params = Object.fromEntries(model.parameters.map((p) => [p.name, p.value]));

  $("pg-model").value = name;
  $("pg-description").innerHTML =
    `${model.description} <a href="${model.theory}">Theory</a>`;
  const panel = $("pg-params");
  panel.replaceChildren(...model.parameters.map(parameterRow));
  $("pg-species-block").hidden = !model.species;
  $("pg-species-note").hidden = Boolean(model.species) || name === "ic50";
  defaultRange();
  state.yAuto = true;
  $("pg-yauto").checked = true;
  draw();
}

function buildRangeInputs() {
  for (const [id, key] of [["pg-xmin", "xmin"], ["pg-xmax", "xmax"]]) {
    $(id).addEventListener("change", () => {
      const value = Number($(id).value);
      const other = key === "xmin" ? state.xmax : state.xmin;
      const ok = value > 0 && (key === "xmin" ? value < other : value > other);
      if (ok) state[key] = value;
      $(id).value = format(state[key]);
      draw();
    });
  }
}

function showYRange(lo, hi) {
  $("pg-ymin").value = format(lo);
  $("pg-ymax").value = format(hi);
}

// Typing a bound fixes the axis at the two shown values; Auto releases it.
function buildYRangeInputs() {
  const read = (id) => ($(id).value === "" ? Number.NaN : Number($(id).value));
  const lock = () => {
    const lo = read("pg-ymin");
    const hi = read("pg-ymax");
    if (Number.isFinite(lo) && Number.isFinite(hi) && lo < hi) {
      Object.assign(state, { yAuto: false, ylo: lo, yhi: hi });
      $("pg-yauto").checked = false;
    }
    draw();
  };
  $("pg-ymin").addEventListener("change", lock);
  $("pg-ymax").addEventListener("change", lock);
  $("pg-yauto").addEventListener("change", () => {
    if ($("pg-yauto").checked) {
      state.yAuto = true;
      draw();
    } else {
      lock();
    }
  });
}

function buildButtons() {
  $("pg-keep").addEventListener("click", () => {
    state.kept = currentCurve();
    draw();
  });
  $("pg-clear").addEventListener("click", () => {
    state.kept = null;
    draw();
  });
  $("pg-reset").addEventListener("click", () => selectModel(state.model));
  $("pg-logy").addEventListener("change", draw);
}

// --- Computation and plots -------------------------------------------------

function grid() {
  return logspace(state.xmin, state.xmax, N_POINTS);
}

function currentCurve() {
  const x = grid();
  return { x, y: x.map((xi) => response(state.model, xi, state.params)) };
}

function theme() {
  const style = getComputedStyle(document.documentElement);
  const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
  return {
    text: read("--pst-color-text-base", "#222"),
    muted: read("--pst-color-text-muted", "#777"),
    grid: read("--pst-color-border", "#ddd"),
  };
}

function baseLayout(colors, xTitle, yTitle) {
  const axis = (title) => ({
    title: { text: title },
    color: colors.text,
    gridcolor: colors.grid,
    linecolor: colors.muted,
    zeroline: false,
    automargin: true,
    exponentformat: "power",
  });
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: colors.text, size: 12 },
    margin: { l: 72, r: 16, t: 24, b: 50 },
    xaxis: { ...axis(xTitle), type: "log", dtick: 1 },
    yaxis: axis(yTitle),
    legend: { orientation: "h", y: -0.25, x: 0 },
    hovermode: "x unified",
  };
}

const CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
};

function drawResponse(colors) {
  const model = MODELS[state.model];
  const curve = currentCurve();
  const traces = [];
  if (state.kept) {
    traces.push({
      ...state.kept,
      name: "kept curve",
      mode: "lines",
      line: { color: colors.muted, dash: "dash", width: 2 },
      hovertemplate: "%{y:.4g}<extra>kept</extra>",
    });
  }
  traces.push({
    ...curve,
    name: "response",
    mode: "lines",
    line: { color: "#1f77b4", width: 3 },
    hovertemplate: "%{y:.4g}<extra>response</extra>",
  });

  const marker = state.params[model.marker];
  const markerLabel = model.parameters.find((p) => p.name === model.marker).label;
  const layout = baseLayout(colors, model.x, "response");
  layout.showlegend = Boolean(state.kept);
  layout.shapes = [
    {
      type: "line", xref: "x", yref: "paper", x0: marker, x1: marker, y0: 0, y1: 1,
      line: { color: colors.muted, dash: "dash", width: 1 },
    },
    ...["ymin", "ymax"].map((key) => ({
      type: "line", xref: "paper", yref: "y", x0: 0, x1: 1,
      y0: state.params[key], y1: state.params[key],
      line: { color: colors.muted, dash: "dot", width: 1 },
    })),
  ];
  layout.annotations = [
    {
      x: Math.log10(marker), xref: "x", y: 1, yref: "paper", yanchor: "bottom",
      text: markerLabel, showarrow: false, font: { color: colors.muted },
    },
  ];
  if (!state.yAuto) {
    Object.assign(layout.yaxis, { range: [state.ylo, state.yhi], autorange: false });
  }
  Plotly.react("pg-response", traces, layout, CONFIG);
  // Show the axis range in use, which Plotly stores after autoranging.
  const [lo, hi] = $("pg-response").layout.yaxis.range;
  if (state.yAuto) showYRange(lo, hi);
  else showYRange(state.ylo, state.yhi);
}

function drawSpecies(colors) {
  const model = MODELS[state.model];
  if (!model.species) return;
  const x = grid();
  const values = x.map((xi) => species(state.model, xi, state.params));
  const traces = model.species.map((key) => ({
    x,
    y: values.map((v) => v[key]),
    name: SPECIES_LABELS[key],
    mode: "lines",
    line: { color: SPECIES_COLORS[key], width: 2 },
    hovertemplate: "%{y:.4g} nM",
  }));
  const layout = baseLayout(colors, model.x, "concentration (nM)");
  if ($("pg-logy").checked) Object.assign(layout.yaxis, { type: "log", dtick: 1 });
  Plotly.react("pg-species", traces, layout, CONFIG);
}

function draw() {
  const colors = theme();
  try {
    $("pg-error").hidden = true;
    drawResponse(colors);
    drawSpecies(colors);
  } catch (error) {
    $("pg-error").hidden = false;
    $("pg-error").textContent = `These parameters could not be evaluated: ${error.message}`;
  }
}

let pending = false;
function scheduleDraw() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    draw();
  });
}

// --- Start -----------------------------------------------------------------

function start() {
  if (typeof Plotly === "undefined") {
    $("pg-error").hidden = false;
    $("pg-error").textContent = "The plotting library could not be loaded.";
    return;
  }
  buildModelSelect();
  buildRangeInputs();
  buildYRangeInputs();
  buildButtons();
  selectModel(state.model);
  // Redraw with the new colors when the light/dark theme changes.
  new MutationObserver(draw).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
}

start();
