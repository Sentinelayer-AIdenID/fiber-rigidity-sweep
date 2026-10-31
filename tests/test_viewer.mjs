// SPDX-License-Identifier: MIT
// DOM-level checks only: these do not replace a real browser/WebGL inspection.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const app = fs.readFileSync(path.join(root, 'viewer/app.js'), 'utf8');

function fixture() {
  const curve = (z, kappa) => ({z_target: z, z_actual: z - 0.05, kappa,
    gamma: [0.01, 0.1, 1], K: [0.001, null, 0.5], nonaffinity: [1, 2, 0.5],
    stretch_energy: [0.1, 0.2, 0.3], bend_energy: [0.1, 0.2, 0.3], converged_count: [2, 0, 2],
    gamma_c_K: 0.1, gamma_c_nonaffinity: 0.1, agreement_log10: 0});
  const curves = [curve(3.2, 0.001), curve(3.4, 0.001), curve(3.2, 0.01)];
  const network_frames = [];
  for (const curve of curves) for (const seed of [1, 2]) for (const gamma of curve.gamma) {
    network_frames.push({z_target: curve.z_target, kappa: curve.kappa, seed, gamma, converged: gamma !== 0.1,
      positions: [[0, 0], [9, 0]], edges: [[0, 1]], edge_vectors: [[-1, 0]],
      stretch_energy: [gamma], bend_energy: [2 * gamma], box: [[10, 0], [2, 8]]});
  }
  return {schema_version: 1, metadata: {}, curves, network_frames};
}

function mount(data, {webgl = false, plotly = true} = {}) {
  const elements = new Map(), strokes = [], plots = new Map(), drawCalls = [];
  let start;
  const ctx = {clearRect() {strokes.length = 0;}, beginPath() {}, moveTo(x, y) {start = [x, y];},
    lineTo(x, y) {strokes.push([start, [x, y]]);}, stroke() {}};
  class Element {
    constructor(id) {
      Object.assign(this, {id, _value: '', textContent: '', hidden: false, disabled: false, children: [],
        handlers: {}, attributes: {}, clientWidth: 600, clientHeight: 300, classList: {add() {}}});
    }
    get value() {return this._value;}
    set value(value) {this._value = String(value);}
    replaceChildren() {this.children = []; this.value = '';}
    appendChild(item) {this.children.push(item); if (this.children.length === 1) this.value = item.value;}
    addEventListener(type, fn) {this.handlers[type] = fn;}
    setAttribute(key, value) {this.attributes[key] = value;}
    getContext() {return ctx;}
  }
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  };
  const window = {RIGIDITY_SAMPLE: data, devicePixelRatio: 1, addEventListener() {}};
  if (plotly) window.Plotly = {react(id, traces, layout) {plots.set(id, {traces, layout});}};
  if (webgl) window.createREGL = ({canvas}) => {
    assert.equal(canvas.id, 'network-canvas');
    const regl = command => {
      assert.equal(command.primitive, 'lines');
      assert.equal(command.depth.enable, false);
      assert.match(command.vert, /gl_Position/);
      assert.match(command.frag, /gl_FragColor/);
      return props => {
        assert.equal(props.count, props.positions.length);
        assert.equal(props.colors.length, props.count);
        assert(props.positions.every(point => point.length === 2 && point.every(Number.isFinite)));
        assert(props.colors.every(color => color.length === 3 && color.every(value => value >= 0 && value <= 1)));
        drawCalls.push(props);
      };
    };
    regl.prop = name => name;
    regl.poll = () => {};
    regl.clear = () => {};
    return regl;
  };
  vm.runInNewContext(app, {document: {getElementById: get, createElement: () => new Element('option')},
    window, console, setTimeout, clearTimeout});
  const upload = async value => get('file-input').handlers.change({target: {
    files: [{name: 'uploaded.json', text: async () => JSON.stringify(value)}], value: ''}});
  return {get, plots, strokes, drawCalls, upload};
}

function assertLoaded(ui) {
  assert(!ui.get('message').className.includes('error'), ui.get('message').textContent);
  assert.equal(ui.get('dataset-label').textContent, 'Bundled CPU sample');
  assert.equal(ui.get('kappa-select').disabled, false);
  assert.equal(ui.plots.size, 2);
}

test('selectors, null masks, transition overlays, periodic vectors and fixed energy scales', () => {
  const {get, plots, strokes} = mount(fixture());
  assert.equal(get('kappa-select').value, '0.001');
  assert.equal(get('z-select').children.length, 2);
  assert.equal(get('seed-select').children.length, 2);
  assert.equal(get('actual-z').textContent, '3.15');
  assert.equal(plots.get('heatmap').traces.length, 3);
  assert.equal(plots.get('heatmap').traces[0].z[0][1], null);
  assert.equal(plots.get('heatmap').traces[1].x[0], -1);
  assert.equal(plots.get('response-plot').traces[0].y[1], null);
  assert(strokes.at(-1)[1][0] < strokes.at(-1)[0][0], 'Periodic vector must point left, not toward the naive endpoint.');
  assert.equal(get('energy-max').textContent, '1');
  get('strain-slider').handlers.input({target: {value: '1'}});
  assert.match(get('network-status').textContent, /NOT CONVERGED/);
  assert.equal(get('energy-max').textContent, '1', 'The color scale must stay fixed when strain changes.');
  get('bend-button').handlers.click();
  assert.equal(get('energy-max').textContent, '2');
  assert.equal(get('bend-button').attributes['aria-pressed'], 'true');
  get('kappa-select').value = '0.01';
  get('kappa-select').handlers.change();
  assert.equal(get('z-select').children.length, 1);
  assert.equal(get('strain-value').textContent, '0.1');
});

test('zero bending stiffness is a valid central-force limit', () => {
  const data = fixture();
  data.curves = [data.curves[0]];
  data.curves[0].kappa = 0;
  data.network_frames = [data.network_frames[0]];
  data.network_frames[0].kappa = 0;
  data.network_frames[0].bend_energy = [0];
  const ui = mount(data);
  assertLoaded(ui);
  assert.equal(ui.get('kappa-select').value, '0');
  ui.get('bend-button').handlers.click();
  assert.equal(ui.get('energy-max').textContent, '0');
});

test('invalid imports show errors, and missing network frames remain explicit', async () => {
  const ui = mount(fixture());
  await ui.upload({schema_version: 2});
  assert.match(ui.get('message').textContent, /Expected schema_version 1/);
  const invalid = fixture();
  invalid.network_frames[0].edges = [[0, 1000]];
  await ui.upload(invalid);
  assert.match(ui.get('message').textContent, /edge indices/);
  await ui.upload({...fixture(), network_frames: []});
  assert.equal(ui.get('dataset-label').textContent, 'uploaded.json');
  assert.equal(ui.get('network-empty').hidden, false);
  assert.equal(ui.get('strain-slider').disabled, true);
  assert.equal(ui.get('seed-select').disabled, true);
});

test('regl command receives finite line geometry and colors', () => {
  const ui = mount(fixture(), {webgl: true});
  assertLoaded(ui);
  assert.equal(ui.get('renderer-label').textContent, 'regl / WebGL');
  assert.equal(ui.drawCalls.length, 1);
  assert.equal(ui.drawCalls[0].count, 10, 'Four cell-outline segments and one network bond.');
});

test('CDN failure preserves numerical summaries and Canvas network rendering', () => {
  const ui = mount(fixture(), {plotly: false});
  assert.equal(ui.get('actual-z').textContent, '3.15');
  assert.match(ui.get('heatmap').textContent, /Plot library unavailable/);
  assert.equal(ui.get('renderer-label').textContent, 'Canvas fallback');
  assert.equal(ui.strokes.length, 5);
});

test('bundled sample JSON and script agree and render through all saved selections', () => {
  const script = fs.readFileSync(path.join(root, 'viewer/sample-data.js'), 'utf8');
  const json = JSON.parse(fs.readFileSync(path.join(root, 'viewer/sample.json'), 'utf8'));
  const context = {window: {}};
  vm.runInNewContext(script, context);
  assert.deepEqual(JSON.parse(JSON.stringify(context.window.RIGIDITY_SAMPLE)), json);
  assert(json.network_frames.length > 0, 'The committed sample must have actual network frames.');
  const ui = mount(context.window.RIGIDITY_SAMPLE, {webgl: true});
  assertLoaded(ui);
  for (const curve of json.curves) {
    ui.get('kappa-select').value = curve.kappa;
    ui.get('kappa-select').handlers.change();
    ui.get('z-select').value = curve.z_target;
    ui.get('z-select').handlers.change();
    for (const option of ui.get('seed-select').children) {
      ui.get('seed-select').value = option.value;
      ui.get('seed-select').handlers.change();
      for (let index = 0; index <= Number(ui.get('strain-slider').max); index++) {
        ui.get('strain-slider').handlers.input({target: {value: String(index)}});
        assert.equal(ui.get('network-empty').hidden, true);
        assert(!ui.get('network-status').textContent.includes('No saved frame'));
      }
    }
  }
  assert(ui.drawCalls.length >= json.network_frames.length);
});

test('critical table shows uncertainty, window, failed fits and per-kappa measured-z trend', () => {
  const data = fixture();
  const failed = {status: 'not determined', n_points: 2, f: null, fit_window: [.11, .12], reason: 'only 2 usable near-transition points; need 5'};
  for (const curve of data.curves) curve.critical_exponents = {gamma_c_K: failed, gamma_c_nonaffinity: failed};
  data.curves[0].critical_exponents.gamma_c_nonaffinity = {status: 'determined', f: .93, f_stderr: .01,
    f_ci95: [.905, .955], n_points: 10, fit_window: [.11, .15], delta_f_literature: .23};
  data.critical_strain_trends = [
    {kappa: .001, method: 'gamma_c_nonaffinity', status: 'increasing', n_points: 2},
    {kappa: .01, method: 'gamma_c_nonaffinity', status: 'not determined', n_points: 1}];
  const ui = mount(data);
  assertLoaded(ui);
  const rows = ui.get('critical-fit-rows').children;
  assert.equal(rows.length, 4);
  assert.equal(rows[0].children[0].textContent, '3.15');
  assert.match(rows[0].children[3].textContent, /not determined: only 2/);
  assert.equal(rows[1].children[3].textContent, '0.93 ± 0.01');
  assert.equal(rows[1].children[3].title, 'Conditional 95% interval: 0.905 – 0.955');
  assert.equal(rows[1].children[4].textContent, '0.11 – 0.15 (10)');
  assert.equal(rows[1].children[5].textContent, '0.23');
  assert.match(ui.get('critical-trend').textContent, /Nonaffinity peak: increasing \(2 resolved/);
  assert.match(ui.get('critical-trend').textContent, /limit.*not determined/);
  ui.get('kappa-select').value = '.01';
  ui.get('kappa-select').handlers.change();
  assert.equal(ui.get('critical-fit-rows').children.length, 2);
  assert.match(ui.get('critical-trend').textContent, /Nonaffinity peak: not determined \(1 resolved/);
});

test('critical table keeps legacy exports unresolved and rejects an invalid fit', async () => {
  const ui = mount(fixture());
  assert.equal(ui.get('critical-fit-rows').children[0].children[3].textContent, 'not determined');
  const data = fixture();
  data.curves[0].critical_exponents = {gamma_c_K: {status: 'determined', f: .7}};
  await ui.upload(data);
  assert.match(ui.get('message').textContent, /Invalid critical exponent result/);
});
