import { useCallback, useEffect, useMemo, useState } from 'react';
import { BarChart } from './charts.jsx';
import { Button, Segmented, Slider } from './controls.jsx';
import { DECODERS, api, fmtLer, fmtP, fmtUs, indexLer } from './data.js';
import { useSpring } from './motion.js';

const DEMO_SEED = 4000;

// ------------------------------------------------------------------ lattice
function EventDot({ cx, cy, r, count }) {
  const [on, setOn] = useState(false);
  useEffect(() => { const id = requestAnimationFrame(() => setOn(true)); return () => cancelAnimationFrame(id); }, []);
  const s = useSpring(on ? 1 : 0, { damping: 1, response: 0.32 });
  return (
    <g transform={`translate(${cx},${cy}) scale(${s})`}>
      <circle r={r * 1.9} className="event-halo" />
      <circle r={r} className="event-dot" />
      {count > 1 && <text className="event-count" dy="0.35em" textAnchor="middle">{count}</text>}
    </g>
  );
}

function Lattice({ layout, shot, layer }) {
  const d = layout.d;
  const pad = 1.4;
  const S = 2 * d;
  const unit = 44;
  const view = `${-pad * unit} ${-pad * unit} ${(S + 2 * pad) * unit} ${(S + 2 * pad) * unit}`;
  const P = (v) => v * unit;
  const dataSet = new Set(layout.data.map(([x, y]) => `${x},${y}`));
  const checkType = new Map(layout.checks.map((c) => [`${c.x},${c.y}`, c.type]));

  const polys = layout.checks.map((c) => {
    const nb = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([dx, dy]) => [c.x + dx, c.y + dy]).filter(([x, y]) => dataSet.has(`${x},${y}`));
    const pts = nb.length === 4 ? nb : [...nb, [c.x, c.y]];
    return { ...c, pts: pts.map(([x, y]) => `${P(x)},${P(y)}`).join(' ') };
  });

  const det = layout.detectors;
  const lastT = Math.max(...det.map((q) => q[2]));
  const inLayer = (i) => layer === 'all' || det[i][2] === layer;
  const fired = shot ? shot.fired.filter(inLayer) : [];
  const counts = new Map();
  fired.forEach((i) => { const k = `${det[i][0]},${det[i][1]}`; counts.set(k, (counts.get(k) || 0) + 1); });

  // Matching edges. Boundary edges go to the nearest boundary that this check type's error chains end on:
  // Z checks (detect X errors) terminate top/bottom, X checks left/right, in Stim's rotated_memory_z layout.
  const edges = !shot ? [] : shot.edges.filter(([a, b]) => inLayer(a) || (b >= 0 && inLayer(b))).map(([a, b], k) => {
    const A = det[a];
    if (b < 0) {
      const z = checkType.get(`${A[0]},${A[1]}`) === 'Z';
      const to = z ? [A[0], A[1] < d ? -0.6 : S + 0.6] : [A[0] < d ? -0.6 : S + 0.6, A[1]];
      return { k, kind: 'boundary', x1: P(A[0]), y1: P(A[1]), x2: P(to[0]), y2: P(to[1]) };
    }
    const B = det[b];
    if (A[0] === B[0] && A[1] === B[1]) return { k, kind: 'time', x1: P(A[0]), y1: P(A[1]) };
    return { k, kind: 'space', x1: P(A[0]), y1: P(A[1]), x2: P(B[0]), y2: P(B[1]) };
  });

  return (
    <svg className="lattice" viewBox={view} role="img"
      aria-label={`Distance ${d} rotated surface code${shot ? ` with ${fired.length} detection events shown` : ''}`}>
      {polys.map((c) => <polygon key={`${c.x},${c.y}`} points={c.pts} className={`plaq plaq-${c.type}`} />)}
      {layout.data.map(([x, y]) => <circle key={`${x},${y}`} cx={P(x)} cy={P(y)} r={7} className="data-qubit" />)}
      {edges.map((e) => (e.kind === 'time'
        ? <circle key={e.k} cx={e.x1} cy={e.y1} r={21} className="match-edge match-time" />
        : <line key={e.k} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} className={`match-edge match-${e.kind}`} />))}
      {[...counts.entries()].map(([k, c]) => {
        const [x, y] = k.split(',').map(Number);
        return <EventDot key={`${shot.seed}-${layer}-${k}`} cx={P(x)} cy={P(y)} r={11} count={c} />;
      })}
      {layer !== 'all' && layer === lastT && (
        <text x={P(d)} y={P(S + 1.05)} textAnchor="middle" className="lattice-note">final data-qubit readout</text>
      )}
    </svg>
  );
}

// ------------------------------------------------------------------ verdict cards
function Verdict({ name, dec, truth }) {
  const meta = DECODERS[name];
  const ok = dec.flip === truth;
  return (
    <div className="verdict">
      <div className="verdict-head">
        <span className="swatch" style={{ background: meta.color }} aria-hidden />
        <span className="verdict-name">{meta.short}</span>
        <span className={`status ${ok ? 'good' : 'bad'}`}>
          <span aria-hidden>{ok ? '✓' : '✗'}</span>{ok ? 'Correct' : 'Wrong'}
        </span>
      </div>
      <div className="verdict-pred">{dec.flip ? 'Flip' : 'No flip'}</div>
      {dec.prob != null ? (
        <div className="confidence" title="Network's estimated probability that the logical qubit flipped">
          <div className="confidence-track"><div className="confidence-fill" style={{ width: `${dec.prob * 100}%`, background: meta.color }} /></div>
          <span className="num">P(flip) {(dec.prob * 100).toFixed(1)}%</span>
        </div>
      ) : (
        <div className="confidence muted">exact matching, no probability</div>
      )}
      <div className="verdict-time num">{fmtUs(dec.us)} incl. Python overhead</div>
    </div>
  );
}

// ------------------------------------------------------------------ section
export default function Live({ summary }) {
  const [d, setD] = useState(3);
  const [p, setP] = useState(0.01);
  const [seed, setSeed] = useState(DEMO_SEED);
  const [layouts, setLayouts] = useState({});
  const [shot, setShot] = useState(null);
  const [layer, setLayer] = useState('all');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [bench, setBench] = useState(null);
  const [benchShots, setBenchShots] = useState(50000);
  const [benchBusy, setBenchBusy] = useState(false);

  const layout = layouts[d];
  useEffect(() => {
    if (layouts[d]) return;
    api('layout', { d }).then((l) => setLayouts((m) => ({ ...m, [d]: l }))).catch((e) => setErr(e.message));
  }, [d, layouts]);

  const fetchShot = useCallback(async (dd, pp, ss) => {
    setBusy(true); setErr(null);
    try {
      const s = await api('shot', { d: dd, p: pp, seed: ss, nonempty: 1 });
      setShot(s); setSeed(s.seed);
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }, []);

  useEffect(() => { fetchShot(d, p, seed); }, [d]); // eslint-disable-line react-hooks/exhaustive-deps

  const runBench = async () => {
    setBenchBusy(true); setErr(null);
    try { setBench(await api('benchmark', { d, p, shots: benchShots, seed: seed + 1 })); } catch (e) { setErr(e.message); } finally { setBenchBusy(false); }
  };

  const times = layout ? [...new Set(layout.detectors.map((q) => q[2]))].sort((a, b) => a - b) : [];
  const layerOpts = [{ value: 'all', label: 'All' }, ...times.map((t) => ({ value: t, label: t === times.length - 1 ? 'Final' : `R${t + 1}` }))];
  const perLayer = shot && layout ? times.map((t) => shot.fired.filter((i) => layout.detectors[i][2] === t).length) : [];

  const getLer = useMemo(() => indexLer(summary.ler_results), [summary]);
  const saved = summary.config.p_sweep.includes(p) ? summary.config.p_sweep.find((v) => v === p) : null;

  const benchRows = bench ? ['no_decoder', 'pymatching', 'mlp', 'cnn'].map((k) => ({
    id: k, label: DECODERS[k].short, value: bench.results[k].ler, lo: bench.results[k].ci_low, hi: bench.results[k].ci_high, color: DECODERS[k].color,
  })) : [];

  return (
    <section id="live" className="section">
      <header className="section-head">
        <p className="eyebrow">Live decoder</p>
        <h2>Watch one shot get decoded</h2>
        <p className="lede">
          Stim samples a fresh memory experiment at the noise level you pick. PyMatching and the two saved networks then
          decide whether the logical qubit flipped. Red dots are detection events. Blue lines are the pairs PyMatching matched.
        </p>
      </header>

      <div className="live-grid">
        <div className="card controls-card">
          <div className="field">
            <span className="field-label">Code distance</span>
            <Segmented label="Code distance" value={d} onChange={(v) => { setD(v); setLayer('all'); setBench(null); }}
              options={[{ value: 3, label: 'd = 3' }, { value: 5, label: 'd = 5' }]} />
          </div>
          <div className="field">
            <div className="field-row">
              <span className="field-label">Physical error rate p</span>
              <span className="field-value num">{fmtP(p)}</span>
            </div>
            <Slider label="Physical error rate" min={0.001} max={0.02} step={0.0005} value={p}
              onChange={(v) => setP(+v.toFixed(4))} onCommit={(v) => { setBench(null); fetchShot(d, +v.toFixed(4), seed); }} format={fmtP} />
            <div className="field-hint">
              Estimated PyMatching threshold (d=3/5 crossing): {summary.threshold[0]?.crossing_found ? fmtP(summary.threshold[0].p_threshold) : 'not found'}
            </div>
          </div>
          <div className="btn-row">
            <Button onClick={() => fetchShot(d, p, seed + 1)} busy={busy}>Next shot</Button>
            <Button kind="secondary" onClick={() => fetchShot(d, p, seed - 1 > 0 ? seed - 1 : 1)} disabled={busy}>Previous</Button>
          </div>
          {layout && (
            <dl className="stats">
              <div><dt>Qubits</dt><dd className="num">{layout.stats.qubits}</dd></div>
              <div><dt>Detectors</dt><dd className="num">{layout.stats.detectors}</dd></div>
              <div><dt>Rounds</dt><dd className="num">{layout.rounds}</dd></div>
              <div><dt>Seed</dt><dd className="num">{seed}</dd></div>
            </dl>
          )}
          <p className="fine">Noise: circuit-level depolarizing, so every gate, reset, measurement and idle step fails with probability p.</p>
        </div>

        <div className="card lattice-card">
          <div className="lattice-top">
            <Segmented size="sm" label="Time layer" value={layer} onChange={setLayer} options={layerOpts} />
            {shot && <span className="muted num">{shot.fired.length} of {layout?.stats.detectors} detectors fired</span>}
          </div>
          {layout ? <Lattice layout={layout} shot={shot} layer={layer} /> : <div className="lattice skeleton" />}
          {perLayer.length > 0 && (
            <div className="round-strip" aria-label="Detection events per time layer">
              {perLayer.map((n, i) => (
                <button type="button" key={i} className={`round-cell${layer === times[i] ? ' on' : ''}`} onClick={() => setLayer(times[i])}>
                  <span className="round-bar" style={{ height: `${Math.min(100, n * 18)}%` }} />
                  <span className="round-n num">{n}</span>
                </button>
              ))}
            </div>
          )}
          <div className="legend-inline">
            <span><i className="key key-z" />Z check</span><span><i className="key key-x" />X check</span>
            <span><i className="key key-q" />data qubit</span><span><i className="key key-ev" />detection event</span>
            <span><i className="key key-edge" />PyMatching pair</span>
          </div>
        </div>

        <div className="verdicts">
          <div className={`truth card ${shot?.truth ? 'flip' : ''}`}>
            <span className="eyebrow">Ground truth</span>
            <strong>{shot ? (shot.truth ? 'Logical qubit flipped' : 'No logical flip') : '…'}</strong>
            <span className="muted">Known from the simulation, hidden from the decoders.</span>
          </div>
          {shot && ['pymatching', 'mlp', 'cnn'].map((k) => <Verdict key={k} name={k} dec={shot.decoders[k]} truth={shot.truth} />)}
        </div>
      </div>

      {err && <div className="error" role="alert">{err}</div>}

      <div className="card bench">
        <div className="bench-head">
          <div>
            <h3>Mini-benchmark at p = {fmtP(p)}, d = {d}</h3>
            <p className="muted">Fresh shots (not the test set). Bars show logical error rate with 95% Wilson intervals. Lower is better.</p>
          </div>
          <div className="bench-controls">
            <Segmented size="sm" label="Shots" value={benchShots} onChange={setBenchShots}
              options={[{ value: 20000, label: '20k' }, { value: 50000, label: '50k' }, { value: 200000, label: '200k' }]} />
            <Button onClick={runBench} busy={benchBusy}>Run</Button>
          </div>
        </div>
        {bench ? (
          <>
            <BarChart rows={benchRows} log valueFormat={fmtLer} axisLabel="Logical error rate per shot (log scale)" labelWidth={110} />
            <div className="bench-foot">
              {['pymatching', 'mlp', 'cnn'].map((k) => (
                <span key={k}>{DECODERS[k].short}: <b className="num">{fmtUs(bench.results[k].us_per_shot)}</b>/shot batched</span>
              ))}
              {saved != null && (
                <span>
                  Saved test set at this p: PyMatching {fmtLer(getLer(d, saved, 'pymatching').ler)}, MLP {fmtLer(getLer(d, saved, 'mlp').ler)}, CNN {fmtLer(getLer(d, saved, 'cnn').ler)}
                </span>
              )}
            </div>
          </>
        ) : (
          <div className="empty">Pick a noise level, then press <b>Run</b>. 50k shots take about 2 s at d = 5.</div>
        )}
      </div>
    </section>
  );
}
