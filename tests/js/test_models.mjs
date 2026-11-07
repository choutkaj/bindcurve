// Check the Playground's JavaScript models against bindcurve's Python models.
//
// Usage:
//   python tests/js/make_reference.py reference.json
//   node tests/js/test_models.mjs reference.json

import { readFileSync } from "node:fs";

import { response, species } from "../../docs/_static/playground/models.js";

const RTOL = 1e-9;
// Absolute floor relative to the largest value of a series, for values that
// are zero up to rounding (e.g. free receptor at saturating ligand).
const ATOL_SCALE = 1e-12;

const cases = JSON.parse(readFileSync(process.argv[2], "utf-8"));
let failures = 0;
let checked = 0;

function compare(label, expected, actual) {
  const scale = Math.max(...expected.map(Math.abs));
  expected.forEach((e, i) => {
    const a = actual[i];
    checked++;
    if (!(Math.abs(a - e) <= RTOL * Math.abs(e) + ATOL_SCALE * scale)) {
      failures++;
      if (failures <= 20) {
        console.error(`${label}[${i}]: expected ${e}, got ${a}`);
      }
    }
  });
}

for (const { model, params, x, response: expected, species: expectedSpecies } of cases) {
  const label = `${model} ${JSON.stringify(params)}`;
  compare(`${label} response`, expected, x.map((xi) => response(model, xi, params)));
  if (expectedSpecies) {
    for (const [key, values] of Object.entries(expectedSpecies)) {
      compare(`${label} ${key}`, values, x.map((xi) => species(model, xi, params)[key]));
    }
  }
}

console.log(`${checked - failures} of ${checked} values agree (${cases.length} cases).`);
process.exit(failures ? 1 : 0);
