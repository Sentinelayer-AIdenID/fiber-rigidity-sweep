/* SPDX-License-Identifier: MIT */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const positive = value => finite(value) && value > 0;
  const nonnegative = value => finite(value) && value >= 0;
  const onsetMethods = [['gamma_c_K', 'Modulus upturn'], ['gamma_c_nonaffinity', 'Nonaffinity peak']];
  const unique = values => [...new Set(values)].sort((a, b) => a - b);
  const fmt = value => !finite(value) ? 'unresolved' : value === 0 ? '0' :
    Math.abs(value) < 0.01 || Math.abs(value) >= 1000 ? value.toExponential(2) : Number(value.toPrecision(4)).toString();
  const state = {data: null, frames: [], curve: null, energy: 'stretch_energy', scale: 0, frameIndex: 0};
  const colors = ['#153249', '#26bad2', '#ffe58c', '#ff725f'];
  let renderer = null;

  function message(text, error = false) {
    $('message').textContent = text;
    $('message').className = error ? 'message error' : 'message';
    $('message').hidden = !text;
  }

  function validate(data) {
    if (!data || data.schema_version !== 1 || !Array.isArray(data.curves) || !data.curves.length ||
        !Array.isArray(data.network_frames)) throw new Error('Expected schema_version 1 with curves and network_frames arrays.');
    for (const curve of data.curves) {
      if (!finite(curve.z_target) || !finite(curve.z_actual) || !nonnegative(curve.kappa) ||
          !Array.isArray(curve.gamma) || !curve.gamma.length || !curve.gamma.every(positive)) {
        throw new Error('Each curve needs finite connectivity, nonnegative κ and positive strain values.');
      }
      for (const key of ['K', 'nonaffinity', 'stretch_energy', 'bend_energy', 'converged_count']) {
        if (!Array.isArray(curve[key]) || curve[key].length !== curve.gamma.length ||
            !curve[key].every(value => value === null || finite(value))) {
          throw new Error(`Curve ${key} must have one finite number or null per strain.`);
        }
      }
      if (curve.critical_exponents !== undefined) for (const [key] of onsetMethods) {
        const fit = curve.critical_exponents?.[key];
        const pair = value => Array.isArray(value) && value.length === 2 && value.every(finite);
        if (!fit || !['determined', 'not determined'].includes(fit.status) ||
            !Number.isInteger(fit.n_points) || fit.n_points < 0 ||
            (fit.status === 'determined' && (!positive(fit.f) || !nonnegative(fit.f_stderr) ||
              !pair(fit.f_ci95) || !pair(fit.fit_window) || !finite(fit.delta_f_literature)))) {
          throw new Error('Invalid critical exponent result. Expected a fit with uncertainty and window, or not determined.');
        }
      }
    }
    for (const frame of data.network_frames) {
      if (!finite(frame.z_target) || !nonnegative(frame.kappa) || !finite(frame.seed) || !positive(frame.gamma) ||
          typeof frame.converged !== 'boolean' || !Array.isArray(frame.positions) || !Array.isArray(frame.edges) ||
          !Array.isArray(frame.edge_vectors) || frame.edge_vectors.length !== frame.edges.length) {
        throw new Error('A network frame has missing coordinates, parameters or convergence status.');
      }
      const pair = value => Array.isArray(value) && value.length === 2 && value.every(finite);
      if (!frame.positions.every(pair) || !frame.edge_vectors.every(pair) ||
          !frame.edges.every(edge => pair(edge) && edge.every(i => Number.isInteger(i) && i >= 0 && i < frame.positions.length))) {
        throw new Error('Network coordinates must be finite pairs and edge indices must reference existing nodes.');
      }
      for (const key of ['stretch_energy', 'bend_energy']) {
        if (!Array.isArray(frame[key]) || frame[key].length !== frame.edges.length ||
            !frame[key].every(value => finite(value) && value >= 0)) {
          throw new Error(`Network ${key} must contain one nonnegative number per edge.`);
        }
      }
      if (!Array.isArray(frame.box) || frame.box.length !== 2 || !frame.box.every(pair)) {
        throw new Error('The periodic cell must contain two finite box vectors.');
      }
    }
    return data;
  }

  function options(id, values, previous, format = fmt) {
    const select = $(id);
    select.replaceChildren();
    for (const value of values) {
      const option = document.createElement('option');
      option.value = String(value);
      option.textContent = format(value);
      select.appendChild(option);
    }
    if (values.includes(previous)) select.value = String(previous);
    select.disabled = values.length === 0;
    return values.includes(previous) ? previous : values[0];
  }

  function load(data, name) {
    const checked = validate(data);
    state.data = checked;
    $('dataset-label').textContent = name;
    options('kappa-select', unique(data.curves.map(curve => curve.kappa)), Number($('kappa-select').value));
    message(`${data.curves.length} response curves · ${data.network_frames.length} saved network frames. Only converged samples contribute to transition estimates.`);
    selectKappa();
  }

  function selectKappa() {
    const kappa = Number($('kappa-select').value);
    options('z-select', unique(state.data.curves.filter(curve => curve.kappa === kappa).map(curve => curve.z_target)), Number($('z-select').value));
    drawHeatmap();
    drawCriticalTable();
    selectZ();
  }

  function drawCriticalTable() {
    const kappa = Number($('kappa-select').value);
    const curves = state.data.curves.filter(curve => curve.kappa === kappa).sort((a, b) => a.z_actual - b.z_actual);
    const body = $('critical-fit-rows');
    body.replaceChildren();
    for (const curve of curves) for (const [key, method] of onsetMethods) {
      const fit = curve.critical_exponents?.[key];
      const determined = fit?.status === 'determined';
      const row = document.createElement('tr');
      const values = [fmt(curve.z_actual), method, positive(curve[key]) ? fmt(curve[key]) : 'not determined',
        determined ? `${fmt(fit.f)} ± ${fmt(fit.f_stderr)}` : `not determined${fit?.reason ? `: ${fit.reason}` : ''}`,
        fit?.fit_window ? `${fit.fit_window.map(fmt).join(' – ')} (${fit.n_points})` : `not determined (${fit?.n_points ?? 0})`,
        determined ? fmt(fit.delta_f_literature) : 'not determined'];
      for (const [index, value] of values.entries()) {
        const cell = document.createElement('td');
        cell.textContent = value;
        if (index === 3 && determined) cell.title = `Conditional 95% interval: ${fit.f_ci95.map(fmt).join(' – ')}`;
        row.appendChild(cell);
      }
      body.appendChild(row);
    }
    const trends = state.data.critical_strain_trends?.filter(trend => trend.kappa === kappa) ?? [];
    const descriptions = onsetMethods.map(([key, name]) => {
      const trend = trends.find(value => value.method === key);
      return `${name}: ${trend?.status ?? 'not determined'} (${trend?.n_points ?? 0} resolved connectivities)`;
    });
    $('critical-trend').textContent = `γc versus increasing measured z below 4 — ${descriptions.join('; ')}. Expected direction: decreasing. The limit γc → 0 at z = 4 is not determined by this finite grid.`;
  }

  function selectZ() {
    const kappa = Number($('kappa-select').value);
    const z = Number($('z-select').value);
    state.curve = state.data.curves.find(curve => curve.kappa === kappa && curve.z_target === z);
    const seeds = unique(state.data.network_frames.filter(frame => frame.kappa === kappa && frame.z_target === z).map(frame => frame.seed));
    options('seed-select', seeds, Number($('seed-select').value), String);
    const curve = state.curve;
    $('actual-z').textContent = fmt(curve.z_actual);
    $('critical-k').textContent = fmt(curve.gamma_c_K);
    $('critical-na').textContent = fmt(curve.gamma_c_nonaffinity);
    $('agreement').textContent = fmt(curve.agreement_log10);
    $('curve-label').textContent = `z target ${fmt(z)} · κ ${fmt(kappa)}`;
    drawResponse();
    selectSeed();
  }

  function selectSeed() {
    const previous = state.frames[state.frameIndex]?.gamma;
    state.frames = state.data.network_frames.filter(frame => frame.kappa === Number($('kappa-select').value) &&
      frame.z_target === Number($('z-select').value) && frame.seed === Number($('seed-select').value)).sort((a, b) => a.gamma - b.gamma);
    state.frameIndex = Math.max(0, state.frames.findIndex(frame => frame.gamma === previous));
    $('strain-slider').max = Math.max(0, state.frames.length - 1);
    $('strain-slider').value = state.frameIndex;
    $('strain-slider').disabled = state.frames.length < 2;
    $('strain-min').textContent = state.frames.length ? fmt(state.frames[0].gamma) : '—';
    $('strain-max').textContent = state.frames.length ? fmt(state.frames[state.frames.length - 1].gamma) : '—';
    updateEnergyScale();
    drawNetwork();
  }

  function updateEnergyScale() {
    state.scale = 0;
    for (const frame of state.frames) for (const value of frame[state.energy]) state.scale = Math.max(state.scale, value);
    $('energy-max').textContent = fmt(state.scale);
    $('energy-kind').textContent = `${state.energy === 'stretch_energy' ? 'stretch' : 'bend'} energy / bond`;
  }

  const plotStyle = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#121b29',
    font: {family: 'system-ui, sans-serif', size: 10, color: '#a6b6cd'},
    margin: {l: 62, r: 24, t: 24, b: 66},
    hoverlabel: {bgcolor: '#182638', bordercolor: '#416078', font: {size: 11}},
    legend: {orientation: 'h', x: 0, y: 1.13, font: {size: 10}},
  };
  const axis = {gridcolor: '#253247', zeroline: false, linecolor: '#40516a', tickfont: {size: 10}, automargin: true};
  const plotConfig = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    toImageButtonOptions: {format: 'png', scale: 2}};

  function plot(id, traces, layout) {
    if (!window.Plotly) {
      $(id).classList.add('plot-unavailable');
      $(id).textContent = 'Plot library unavailable. Connect to the internet to load the pinned Plotly CDN, then reload this page. Network controls and numerical summaries remain available.';
      return;
    }
    window.Plotly.react(id, traces, {...plotStyle, ...layout}, plotConfig);
  }

  function drawHeatmap() {
    const curves = state.data.curves.filter(curve => curve.kappa === Number($('kappa-select').value)).sort((a, b) => a.z_target - b.z_target);
    const gammas = unique(curves.flatMap(curve => curve.gamma));
    const values = curves.map(curve => gammas.map(gamma => {
      const index = curve.gamma.indexOf(gamma);
      const value = curve.K[index];
      return positive(value) && curve.converged_count[index] > 0 ? Math.log10(value) : null;
    }));
    const custom = curves.map(curve => gammas.map(gamma => {
      const index = curve.gamma.indexOf(gamma);
      return [gamma, curve.K[index] ?? null, curve.z_actual, curve.converged_count[index] ?? 0];
    }));
    const traces = [{type: 'heatmap', x: gammas.map(Math.log10), y: curves.map(curve => curve.z_target), z: values,
      customdata: custom, colorscale: [[0, '#152a45'], [.3, '#1d768e'], [.6, '#40c5be'], [.8, '#dfdd8a'], [1, '#f18d5a']],
      colorbar: {title: {text: 'log₁₀ K′', side: 'right'}, thickness: 9, len: .8, tickfont: {size: 9}, outlinewidth: 0},
      hoverongaps: false, hovertemplate: 'γ %{customdata[0]:.4g}<br>z target %{y:.3f}<br>z actual %{customdata[2]:.3f}<br>K′ %{customdata[1]:.4g}<br>Converged seeds %{customdata[3]}<extra></extra>'}];
    for (const [key, name, color, dash, symbol] of [
      ['gamma_c_K', 'γc · modulus upturn', '#ffffff', 'solid', 'circle'],
      ['gamma_c_nonaffinity', 'γc · nonaffinity peak', '#f7bb78', 'dot', 'diamond'],
    ]) {
      traces.push({type: 'scatter', mode: 'lines+markers', x: curves.map(curve => positive(curve[key]) ? Math.log10(curve[key]) : null),
        y: curves.map(curve => curve.z_target), customdata: curves.map(curve => curve[key]), name,
        line: {color, width: 2, dash}, marker: {color, size: 6, symbol}, connectgaps: false,
        hovertemplate: `${name}<br>γ %{customdata:.4g}<br>z target %{y:.3f}<extra></extra>`});
    }
    const low = Math.min(...gammas.map(Math.log10));
    const high = Math.max(...gammas.map(Math.log10));
    const ticks = unique([low, ...Array.from({length: Math.max(0, Math.floor(high) - Math.ceil(low) + 1)}, (_, i) => Math.ceil(low) + i), high]);
    plot('heatmap', traces, {margin: {l: 59, r: 68, t: 44, b: 53},
      xaxis: {...axis, title: {text: 'Shear strain γ'}, tickvals: ticks, ticktext: ticks.map(value => fmt(10 ** value))},
      yaxis: {...axis, title: {text: 'Target connectivity z'}, tickvals: curves.map(curve => curve.z_target)},
      legend: {...plotStyle.legend, y: 1.17}, uirevision: `heatmap-${$('kappa-select').value}`});
  }

  function drawResponse() {
    const curve = state.curve;
    const traces = [
      {type: 'scatter', mode: 'lines+markers', name: 'K′ · differential modulus', x: curve.gamma,
        y: curve.K.map((value, i) => positive(value) && curve.converged_count[i] > 0 ? value : null),
        line: {color: '#52d4e8', width: 2}, marker: {size: 5}, connectgaps: false, yaxis: 'y',
        customdata: curve.converged_count, hovertemplate: 'γ %{x:.4g}<br>K′ %{y:.4g}<br>Converged seeds %{customdata}<extra></extra>'},
      {type: 'scatter', mode: 'lines+markers', name: 'Γ · nonaffinity', x: curve.gamma,
        y: curve.nonaffinity.map((value, i) => positive(value) && curve.converged_count[i] > 0 ? value : null),
        line: {color: '#f7bb78', width: 2}, marker: {size: 5, symbol: 'diamond'}, connectgaps: false, yaxis: 'y2',
        hovertemplate: 'γ %{x:.4g}<br>Γ %{y:.4g}<extra></extra>'},
    ];
    const shapes = [['gamma_c_K', '#52d4e8'], ['gamma_c_nonaffinity', '#f7bb78']].filter(([key]) => positive(curve[key])).map(([key, color]) =>
      ({type: 'line', xref: 'x', yref: 'paper', x0: curve[key], x1: curve[key], y0: 0, y1: 1, line: {color, width: 1, dash: 'dot'}}));
    plot('response-plot', traces, {margin: {l: 65, r: 70, t: 41, b: 48},
      xaxis: {...axis, type: 'log', title: {text: 'Shear strain γ'}},
      yaxis: {...axis, type: 'log', title: {text: 'Differential modulus K′', font: {color: '#52d4e8'}}},
      yaxis2: {...axis, type: 'log', title: {text: 'Nonaffinity Γ', font: {color: '#f7bb78'}}, overlaying: 'y', side: 'right', showgrid: false},
      legend: {...plotStyle.legend, y: 1.3}, shapes, uirevision: `${curve.kappa}-${curve.z_target}`});
  }

  function energyColor(value) {
    const ratio = state.scale > 0 ? Math.min(1, Math.max(0, value / state.scale)) : 0;
    const scaled = ratio * (colors.length - 1);
    const index = Math.min(colors.length - 2, Math.floor(scaled));
    const weight = scaled - index;
    const rgb = hex => [1, 3, 5].map(start => parseInt(hex.slice(start, start + 2), 16) / 255);
    const a = rgb(colors[index]), b = rgb(colors[index + 1]);
    return a.map((channel, i) => channel * (1 - weight) + b[i] * weight);
  }

  function makeRenderer() {
    let canvas = $('network-canvas');
    if (window.createREGL) {
      try {
        const regl = window.createREGL({canvas, attributes: {antialias: true, alpha: false}});
        const draw = regl({
          vert: 'precision mediump float; attribute vec2 position; attribute vec3 color; varying vec3 vColor; void main(){vColor=color;gl_Position=vec4(position,0.0,1.0);}',
          frag: 'precision mediump float; varying vec3 vColor; void main(){gl_FragColor=vec4(vColor,1.0);}',
          attributes: {position: regl.prop('positions'), color: regl.prop('colors')}, count: regl.prop('count'),
          primitive: 'lines', depth: {enable: false},
        });
        $('renderer-label').textContent = 'regl / WebGL';
        return geometry => {
          regl.poll();
          regl.clear({color: [0.063, 0.098, 0.145, 1]});
          if (geometry.positions.length) draw({...geometry, count: geometry.positions.length});
        };
      } catch (_) {
        // A canvas with an initialized WebGL context cannot become a 2D canvas.
        const replacement = canvas.cloneNode();
        canvas.replaceWith(replacement);
        canvas = replacement;
      }
    }
    const ctx = canvas.getContext('2d');
    $('renderer-label').textContent = 'Canvas fallback';
    return geometry => {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
      for (let i = 0; i < geometry.positions.length; i += 2) {
        const a = geometry.positions[i], b = geometry.positions[i + 1];
        const color = geometry.colors[i].map(value => Math.round(value * 255));
        ctx.strokeStyle = `rgb(${color.join(',')})`;
        ctx.beginPath();
        ctx.moveTo((a[0] + 1) * canvas.width / 2, (1 - a[1]) * canvas.height / 2);
        ctx.lineTo((b[0] + 1) * canvas.width / 2, (1 - b[1]) * canvas.height / 2);
        ctx.stroke();
      }
    };
  }

  function drawNetwork() {
    const frame = state.frames[state.frameIndex];
    $('network-empty').hidden = Boolean(frame);
    $('strain-value').textContent = frame ? fmt(frame.gamma) : '—';
    $('network-status').textContent = frame ? (frame.converged ? 'Converged' : 'NOT CONVERGED · excluded from fits') : 'No saved frame';
    $('network-status').className = `network-badge${frame && !frame.converged ? ' failed' : ''}`;
    if (!renderer) renderer = makeRenderer();
    const canvas = $('network-canvas');
    const width = Math.max(1, canvas.clientWidth), height = Math.max(1, canvas.clientHeight);
    canvas.width = Math.round(width * (window.devicePixelRatio || 1));
    canvas.height = Math.round(height * (window.devicePixelRatio || 1));
    if (!frame) { renderer({positions: [], colors: []}); return; }
    const a = frame.box[0], b = frame.box[1];
    const corners = [[0, 0], a, [a[0] + b[0], a[1] + b[1]], b];
    const segments = frame.edges.map((edge, i) => {
      const p = frame.positions[edge[0]], delta = frame.edge_vectors[i];
      return [p, [p[0] + delta[0], p[1] + delta[1]]];
    });
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const point of [...corners, ...segments.flat()]) {
      minX = Math.min(minX, point[0]); maxX = Math.max(maxX, point[0]);
      minY = Math.min(minY, point[1]); maxY = Math.max(maxY, point[1]);
    }
    const scale = 1.72 / Math.max((maxX - minX) / width, (maxY - minY) / height, 1e-12);
    const transform = point => [(point[0] - (minX + maxX) / 2) * scale / width, (point[1] - (minY + maxY) / 2) * scale / height];
    const geometry = {positions: [], colors: []};
    for (let i = 0; i < 4; i++) {
      geometry.positions.push(transform(corners[i]), transform(corners[(i + 1) % 4]));
      geometry.colors.push([.22, .3, .4], [.22, .3, .4]);
    }
    segments.forEach((segment, i) => {
      geometry.positions.push(transform(segment[0]), transform(segment[1]));
      const color = energyColor(frame[state.energy][i]);
      geometry.colors.push(color, color);
    });
    renderer(geometry);
    $('frame-note').textContent = `${frame.edges.length.toLocaleString()} displayed bonds · seed ${frame.seed}. Periodic bonds use image-corrected displacements. Colors share one scale across saved strains; bend energy is allocated to adjacent bonds.`;
  }

  $('file-input').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    try { load(JSON.parse(await file.text()), file.name); }
    catch (error) { message(`Could not load this file: ${error.message}`, true); }
    event.target.value = '';
  });
  $('sample-button').addEventListener('click', () => {
    if (!window.RIGIDITY_SAMPLE) { message('No bundled sample is available. Open the viewer JSON from a completed sweep.', true); return; }
    try { load(window.RIGIDITY_SAMPLE, 'Bundled CPU sample'); }
    catch (error) { message(`Could not load the bundled sample: ${error.message}`, true); }
  });
  $('kappa-select').addEventListener('change', selectKappa);
  $('z-select').addEventListener('change', selectZ);
  $('seed-select').addEventListener('change', selectSeed);
  $('strain-slider').addEventListener('input', event => {
    state.frameIndex = Number(event.target.value);
    drawNetwork();
  });
  for (const [id, energy] of [['stretch-button', 'stretch_energy'], ['bend-button', 'bend_energy']]) {
    $(id).addEventListener('click', () => {
      state.energy = energy;
      $('stretch-button').setAttribute('aria-pressed', String(energy === 'stretch_energy'));
      $('bend-button').setAttribute('aria-pressed', String(energy === 'bend_energy'));
      updateEnergyScale();
      drawNetwork();
    });
  }
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.data) drawNetwork(); }, 100);
  });
  if (window.RIGIDITY_SAMPLE) {
    try { load(window.RIGIDITY_SAMPLE, 'Bundled CPU sample'); }
    catch (error) { message(`Could not load the bundled sample: ${error.message}`, true); }
  } else {
    message('Open the viewer JSON from a sweep, or generate the bundled sample using the README commands. Files are read locally in your browser.');
  }
})();
