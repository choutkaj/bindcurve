// JavaScript port of the bindcurve models, for the documentation Playground.
//
// Every function mirrors the Python implementation in bindcurve/models/ and
// uses the same algorithm, so that results agree to rounding error. The test
// tests/js/test_models.mjs checks this against the Python package.

const TINY = 2.2250738585072014e-308; // np.finfo(float).tiny
const EPS = 2.220446049250313e-16; // np.finfo(float).eps

function expit(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

// Brent's root finder, following scipy.optimize.brentq (scipy/optimize/Zeros/brentq.c).
export function brentq(f, xa, xb, xtol = TINY, rtol = 4 * EPS, maxiter = 1000) {
  let xpre = xa;
  let xcur = xb;
  let xblk = 0;
  let fblk = 0;
  let spre = 0;
  let scur = 0;
  let fpre = f(xpre);
  let fcur = f(xcur);
  if (fpre === 0) return xpre;
  if (fcur === 0) return xcur;
  if (fpre * fcur > 0) throw new Error("brentq: f(a) and f(b) must have different signs");
  for (let i = 0; i < maxiter; i++) {
    if (fpre !== 0 && fcur !== 0 && Math.sign(fpre) !== Math.sign(fcur)) {
      xblk = xpre;
      fblk = fpre;
      spre = scur = xcur - xpre;
    }
    if (Math.abs(fblk) < Math.abs(fcur)) {
      xpre = xcur;
      xcur = xblk;
      xblk = xpre;
      fpre = fcur;
      fcur = fblk;
      fblk = fpre;
    }
    const delta = (xtol + rtol * Math.abs(xcur)) / 2;
    const sbis = (xblk - xcur) / 2;
    if (fcur === 0 || Math.abs(sbis) < delta) return xcur;
    if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
      let stry;
      if (xpre === xblk) {
        stry = (-fcur * (xcur - xpre)) / (fcur - fpre); // secant
      } else {
        const dpre = (fpre - fcur) / (xpre - xcur);
        const dblk = (fblk - fcur) / (xblk - xcur);
        stry = (-fcur * (fblk * dblk - fpre * dpre)) / (dblk * dpre * (fblk - fpre));
      }
      if (2 * Math.abs(stry) < Math.min(Math.abs(spre), 3 * Math.abs(sbis) - delta)) {
        spre = scur;
        scur = stry;
      } else {
        spre = sbis;
        scur = sbis;
      }
    } else {
      spre = sbis;
      scur = sbis;
    }
    xpre = xcur;
    fpre = fcur;
    xcur += Math.abs(scur) > delta ? scur : sbis > 0 ? delta : -delta;
    fcur = f(xcur);
  }
  return xcur;
}

// Free receptor at one total-competitor value; see free_receptor in three_state.py.
function freeReceptor(balance, lt, RT, constants) {
  const scaled = constants.map((c) => c / RT);
  return RT * brentq((r) => balance(r, lt / RT, ...scaled), 0, 1);
}

// --- Logistic -------------------------------------------------------------

function ic50Fraction(x, { IC50, hill_slope }) {
  return expit(hill_slope * (Math.log(IC50) - Math.log(x)));
}

// --- Direct binding --------------------------------------------------------

function dirSimpleSpecies(R, { Kds }) {
  return { R, Fbs: R / (Kds + R) };
}

function directSpecies(RT, { LsT, Kds, Ns = 0 }, nonspecific) {
  const K = (1 + Ns) * Kds;
  const a = K + LsT - RT;
  const root = Math.sqrt(a * a + 4 * K * RT);
  const R = a >= 0 ? (2 * K * RT) / (a + root) : (root - a) / 2;
  const Ls = LsT / (1 + Ns + R / Kds);
  const RLs = (R * Ls) / Kds;
  const species = { R, Ls, RLs, Fbs: RLs / LsT };
  if (nonspecific) species.Ls_nonspecific = Ns * Ls;
  return species;
}

// --- Three-state competition ----------------------------------------------

function balance3(r, lt, lst, kds, k) {
  return r + (lst * r) / (kds + r) + ((lt - 1) * r - k) / (k + r);
}

function threeStateSpecies(LT, { RT, LsT, Kds, Kd, N = 0 }, nonspecific) {
  const K = (1 + N) * Kd;
  const R = freeReceptor(balance3, LT, RT, [LsT, Kds, K]);
  const Ls = (LsT * Kds) / (Kds + R);
  const L = (LT * Kd) / (K + R);
  const species = { R, Ls, L, RLs: (R * Ls) / Kds, RL: (R * L) / Kd, Fbs: R / (Kds + R) };
  if (nonspecific) species.L_nonspecific = N * L;
  return species;
}

// --- Four-state competition -----------------------------------------------

function freeLigands(R, LT, LsT, Kds, Kd, Kd3) {
  const a = 1 + R / Kds;
  const b = 1 + R / Kd;
  const c = R / (Kd * Kd3);
  const A = b * c;
  const B = a * b + c * (LsT - LT);
  const C = -a * LT;
  const root = Math.sqrt(Math.max(B * B - 4 * A * C, 0));
  let L = B >= 0 ? (-2 * C) / (B + root) : (root - B) / (2 * A);
  if (!(A > TINY)) L = -C / B;
  L = Math.max(L, 0);
  return [L, LsT / (a + c * L)];
}

function balance4(r, lt, lst, kds, kd, kd3) {
  const [L, Ls] = freeLigands(r, lt, lst, kds, kd, kd3);
  return r * (1 + Ls / kds + L / kd + (L * Ls) / (kd * kd3)) - 1;
}

function fourStateSpecies(LT, { RT, LsT, Kds, Kd3, Kd, N = 0 }, nonspecific) {
  const K = (1 + N) * Kd;
  const R = freeReceptor(balance4, LT, RT, [LsT, Kds, K, Kd3]);
  const [Lapp, Ls] = freeLigands(R, LT, LsT, Kds, K, Kd3);
  const RLLs = (R * Lapp * Ls) / (K * Kd3);
  const RLs = (R * Ls) / Kds;
  const L = Lapp / (1 + N);
  const species = { R, Ls, L, RLs, RL: (R * Lapp) / K, RLLs, Fbs: (RLs + RLLs) / LsT };
  if (nonspecific) species.L_nonspecific = N * L;
  return species;
}

// --- Model definitions -----------------------------------------------------

const conc = (name, label, value) => ({ name, label, value, log: true });
const lin = (name, label, value, min, max, step) => ({ name, label, value, min, max, step });

const FP_PLATEAUS = [
  lin("ymin", "y<sub>min</sub>", 35, 0, 300, 1),
  lin("ymax", "y<sub>max</sub>", 220, 0, 300, 1),
];
const KDS = conc("Kds", "K<sub>d</sub><sup>*</sup>", 15);
const KD = conc("Kd", "K<sub>d</sub>", 50);
const KD3 = conc("Kd3", "K<sub>d3</sub>", 150);
const RT = conc("RT", "[R]<sub>T</sub>", 40);
const LST = conc("LsT", "[L<sup>*</sup>]<sub>T</sub>", 10);
const NS = lin("Ns", "N<sup>*</sup>", 0.25, 0, 5, 0.01);
const N = lin("N", "N", 0.5, 0, 5, 0.01);

const DIRECT_SPECIES = ["R", "Ls", "RLs"];
const COMP3_SPECIES = ["R", "Ls", "L", "RLs", "RL"];
const COMP4_SPECIES = [...COMP3_SPECIES, "RLLs"];

export const MODELS = {
  ic50: {
    title: "Logistic IC₅₀ model",
    description: "Empirical four-parameter inhibition curve.",
    theory: "theory/logistic.html",
    x: "inhibitor concentration (nM)",
    marker: "IC50",
    parameters: [
      lin("ymin", "y<sub>min</sub>", 0, -20, 150, 1),
      lin("ymax", "y<sub>max</sub>", 100, -20, 150, 1),
      conc("IC50", "IC<sub>50</sub>", 100),
      lin("hill_slope", "h", 1, 0.2, 5, 0.01),
    ],
    species: null,
    fraction: (x, p) => ic50Fraction(x, p),
  },
  dir_simple: {
    title: "Simplified direct binding",
    description: "Receptor titrated into tracer; the x axis is taken as free receptor.",
    theory: "theory/equilibrium.html#simplified-direct-binding",
    x: "free receptor concentration (nM)",
    marker: "Kds",
    parameters: [...FP_PLATEAUS, KDS],
    species: null,
    compute: (x, p) => dirSimpleSpecies(x, p),
  },
  dir_specific: {
    title: "Direct binding with depletion",
    description: "Receptor titrated into tracer; exact mass balances.",
    theory: "theory/equilibrium.html#direct-binding-with-depletion",
    x: "total receptor concentration (nM)",
    marker: "Kds",
    parameters: [...FP_PLATEAUS, LST, KDS],
    species: DIRECT_SPECIES,
    compute: (x, p) => directSpecies(x, p, false),
  },
  dir_total: {
    title: "Direct binding with nonspecific tracer binding",
    description: "As dir_specific, with tracer also bound nonspecifically in proportion to free tracer.",
    theory: "theory/equilibrium.html#direct-binding-with-nonspecific-tracer-binding",
    x: "total receptor concentration (nM)",
    marker: "Kds",
    parameters: [...FP_PLATEAUS, LST, NS, KDS],
    species: [...DIRECT_SPECIES, "Ls_nonspecific"],
    compute: (x, p) => directSpecies(x, p, true),
  },
  comp_3st_specific: {
    title: "Complete competition (three states)",
    description: "Competitor titrated into receptor and tracer; the two ligands cannot bind together.",
    theory: "theory/equilibrium.html#complete-competition-three-states",
    x: "total competitor concentration (nM)",
    marker: "Kd",
    parameters: [...FP_PLATEAUS, RT, LST, KDS, KD],
    species: COMP3_SPECIES,
    compute: (x, p) => threeStateSpecies(x, p, false),
  },
  comp_3st_total: {
    title: "Complete competition with nonspecific competitor binding",
    description: "As comp_3st_specific, with competitor also bound nonspecifically.",
    theory: "theory/equilibrium.html#complete-competition-three-states",
    x: "total competitor concentration (nM)",
    marker: "Kd",
    parameters: [...FP_PLATEAUS, RT, LST, KDS, N, KD],
    species: [...COMP3_SPECIES, "L_nonspecific"],
    compute: (x, p) => threeStateSpecies(x, p, true),
  },
  comp_4st_specific: {
    title: "Incomplete competition (four states)",
    description: "Competitor titrated into receptor and tracer; a ternary complex RLL* can form.",
    theory: "theory/equilibrium.html#incomplete-competition-four-states",
    x: "total competitor concentration (nM)",
    marker: "Kd",
    parameters: [...FP_PLATEAUS, RT, LST, KDS, KD3, KD],
    species: COMP4_SPECIES,
    compute: (x, p) => fourStateSpecies(x, p, false),
  },
  comp_4st_total: {
    title: "Incomplete competition with nonspecific competitor binding",
    description: "As comp_4st_specific, with competitor also bound nonspecifically.",
    theory: "theory/equilibrium.html#incomplete-competition-four-states",
    x: "total competitor concentration (nM)",
    marker: "Kd",
    parameters: [...FP_PLATEAUS, RT, LST, KDS, KD3, N, KD],
    species: [...COMP4_SPECIES, "L_nonspecific"],
    compute: (x, p) => fourStateSpecies(x, p, true),
  },
};

export const SPECIES_LABELS = {
  R: "R (free receptor)",
  Ls: "L* (free tracer)",
  L: "L (free competitor)",
  RLs: "RL*",
  RL: "RL",
  RLLs: "RLL*",
  Ls_nonspecific: "L*<sub>NS</sub> (nonspecific tracer)",
  L_nonspecific: "L<sub>NS</sub> (nonspecific competitor)",
};

// Equilibrium species at a single concentration x (binding models only).
export function species(name, x, params) {
  const model = MODELS[name];
  if (!model.compute) throw new Error(`${name} has no species`);
  return model.compute(x, params);
}

// Modeled fraction at x: the logistic fraction or the tracer-bound fraction Fbs.
export function fraction(name, x, params) {
  const model = MODELS[name];
  return model.fraction ? model.fraction(x, params) : model.compute(x, params).Fbs;
}

// Response at x, ymin + (ymax - ymin) * fraction, as Model.evaluate.
export function response(name, x, params) {
  return params.ymin + (params.ymax - params.ymin) * fraction(name, x, params);
}
