---
html_theme.sidebar_secondary.remove: true
---

# Playground

The Playground draws each `bindcurve` model and redraws it as you change the
parameters. It evaluates the same equations as the Python package, described
in the [theory](theory/index.md) section, directly in your browser.
Concentrations are shown in nM, but the curves look the same in any unit, as
long as all concentrations use it.

For the equilibrium binding models, a second plot shows the equilibrium
concentration of every species across the titration. Hover over a parameter
name for a short description. **Keep curve** freezes the current curve for
comparison while you change parameters.

```{raw} html
<link rel="stylesheet" href="_static/playground/playground.css">
<div class="playground">
  <div class="pg-controls">
    <label for="pg-model">Model</label>
    <select id="pg-model"></select>
    <p id="pg-description" class="pg-description"></p>
    <div id="pg-params"></div>
    <fieldset>
      <legend>Concentration range (nM)</legend>
      <div class="pg-range">
        <label for="pg-xmin">from</label>
        <input id="pg-xmin" type="number" min="0" step="any">
        <label for="pg-xmax">to</label>
        <input id="pg-xmax" type="number" min="0" step="any">
      </div>
    </fieldset>
    <fieldset>
      <legend>Response range</legend>
      <div class="pg-range pg-range-y">
        <label for="pg-ymin">from</label>
        <input id="pg-ymin" type="number" step="any">
        <label for="pg-ymax">to</label>
        <input id="pg-ymax" type="number" step="any">
        <label class="pg-auto"><input id="pg-yauto" type="checkbox" checked> Auto</label>
      </div>
    </fieldset>
    <div class="pg-buttons">
      <button id="pg-keep" type="button">Keep curve</button>
      <button id="pg-clear" type="button">Clear</button>
      <button id="pg-reset" type="button">Reset</button>
    </div>
  </div>
  <div class="pg-plots">
    <p id="pg-error" class="pg-error" hidden></p>
    <noscript><p class="pg-error">The Playground needs JavaScript.</p></noscript>
    <div id="pg-response" class="pg-plot" role="img" aria-label="Response curve"></div>
    <div id="pg-species-block">
      <div class="pg-species-head">
        <strong>Species at equilibrium</strong>
        <label><input id="pg-logy" type="checkbox" checked> logarithmic concentration axis</label>
      </div>
      <div id="pg-species" class="pg-plot" role="img" aria-label="Species concentrations"></div>
    </div>
    <p id="pg-species-note" class="pg-note" hidden>
      The simplified direct-binding model has no tracer concentration, so only
      the bound fraction is defined and there is no species plot.
    </p>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@2.35.2/plotly-basic.min.js" charset="utf-8"></script>
<script type="module" src="_static/playground/playground.js"></script>
```
