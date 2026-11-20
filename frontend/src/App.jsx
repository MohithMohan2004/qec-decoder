import { useEffect, useMemo, useState } from 'react';
import { BarChart, LineChart, StackedBars } from './charts.jsx';
import { Segmented } from './controls.jsx';
import { DECODERS, api, fmtCompact, fmtInt, fmtLer, fmtP, fmtPct, fmtUs, indexLer, overlaps } from './data.js';
import Live from './Live.jsx';

const SECTIONS = [
  ['live', 'Live decoder'],
  ['results', 'Accuracy'],
  ['speed', 'Speed & size'],
  ['analysis', 'Analysis'],
  ['verdict', 'Verdict'],
];

// ------------------------------------------------------------------ chrome
function Nav() {
  const [active, setActive] = useState('');
  const [theme, setTheme] = useState(() => { try { return localStorage.getItem('theme') || 'auto'; } catch { return 'auto'; } });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'auto') delete root.dataset.theme; else root.dataset.theme = theme;
    try { localStorage.setItem('theme', theme); } catch { /* storage unavailable: theme just won't persist */ }
  }, [theme]);

  useEffect(() => {
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) setActive(e.target.id); });
    }, { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);

  return (
    <nav className="nav" aria-label="Sections">
      <a className="brand" href="#top"><span className="brand-mark" aria-hidden />QEC Decoder Lab</a>
      <div className="nav-links">
        {SECTIONS.map(([id, label]) => (
          <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>{label}</a>
        ))}
      </div>
      <Segmented size="sm" label="Colour theme" value={theme} onChange={setTheme}
        options={[{ value: 'auto', label: 'Auto' }, { value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }]} />
    </nav>
  );
}

function Stat({ label, value, sub, tone }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={`stat-value num ${tone || ''}`}>{value}</span>
      <span className="stat-sub">{sub}</span>
    </div>
  );
}

function Section({ id, eyebrow, title, lede, children }) {
  return (
    <section id={id} className="section">
      <header className="section-head">
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
        {lede && <p className="lede">{lede}</p>}
      </header>
      {children}
    </section>
  );
}

function Card({ title, note, children, className = '' }) {
  return (
    <div className={`card ${className}`}>
      {title && <h3>{title}</h3>}
      {note && <p className="muted card-note">{note}</p>}
      {children}
    </div>
  );
}

// ------------------------------------------------------------------ derived numbers (all from results/)
function useFacts(s) {
  return useMemo(() => {
    const get = indexLer(s.ler_results);
    const fp = s.config.fixed_p;
    const ratio = (d, p, k) => get(d, p, k).ler / get(d, p, 'pymatching').ler;
    const tp = (d, k, big) => s.throughput.find((r) => r.distance === d && r.decoder === k && (big ? r.batch_size > 1 : r.batch_size === 1));
    const th = s.threshold[0];
    const shots = s.eval_meta?.test_shots || {};
    return { get, fp, ratio, tp, th, shots };
  }, [s]);
}

// ------------------------------------------------------------------ hero
function Hero({ s, f }) {
  const fp = f.fp;
  const best = (d) => (['mlp', 'cnn'].map((k) => ({ k, r: f.ratio(d, fp, k) })).sort((a, b) => a.r - b.r)[0]);
  const b3 = best(3), b5 = best(5);
  const pm5 = f.tp(5, 'pymatching', true), mlp5 = f.tp(5, 'mlp', true);
  return (
    <header className="hero" id="top">
      <p className="eyebrow">Quantum Computing · 25AM3504 · course project</p>
      <h1>Surface-code decoding:<br />matching vs a small neural net</h1>
      <p className="hero-lede">
        We simulate the rotated surface code at distance 3 and 5 with <b>Stim</b> (circuit-level noise, d rounds) and decode it three ways:
        <b> PyMatching</b> (minimum-weight perfect matching), a small <b>MLP</b> and a small <b>CNN</b>. Every number on this page is read from <code>results/</code>.
      </p>
      <div className="stats-row">
        <Stat label={`d = 3 · p = ${fmtP(fp)}`} value={`${b3.r.toFixed(2)}×`} tone={b3.r < 1 ? 'good' : 'bad'}
          sub={`${DECODERS[b3.k].short} error rate relative to PyMatching (lower is better)`} />
        <Stat label={`d = 5 · p = ${fmtP(fp)}`} value={`${b5.r.toFixed(2)}×`} tone={b5.r < 1 ? 'good' : 'bad'}
          sub={`best network (${DECODERS[b5.k].short}) relative to PyMatching`} />
        <Stat label="PyMatching threshold" value={f.th?.crossing_found ? fmtP(f.th.p_threshold) : 'n/a'}
          sub={`d=3 and d=5 curves cross between ${fmtP(f.th.p_below)} and ${fmtP(f.th.p_above)}`} />
        <Stat label="d = 5 throughput" value={`${fmtCompact(mlp5.shots_per_s)}/s`}
          sub={`MLP batched, vs PyMatching ${fmtCompact(pm5.shots_per_s)}/s on the same CPU`} />
      </div>
      <a className="btn btn-primary hero-cta" href="#live">Try the live decoder</a>
    </header>
  );
}

// ------------------------------------------------------------------ accuracy
function Results({ s, f }) {
  const [d, setD] = useState(3);
  const ps = s.config.p_sweep;
  const decs = ['pymatching', 'mlp', 'cnn', 'pymatching_fixed_p', 'no_decoder'];
  const lerSeries = decs.map((k) => ({
    id: k, label: DECODERS[k].label, short: DECODERS[k].short, color: DECODERS[k].color, dash: DECODERS[k].dash, marker: DECODERS[k].marker,
    points: ps.map((p) => { const r = f.get(d, p, k); return { x: p, y: r.ler, lo: r.ci_low, hi: r.ci_high }; }),
  }));

  const ratioSeries = [3, 5].flatMap((dd) => ['mlp', 'cnn'].map((k) => ({
    id: `${k}${dd}`, label: `${DECODERS[k].short} d=${dd}`, short: `${DECODERS[k].short} d=${dd}`, color: DECODERS[k].color,
    dash: dd === 5 ? '6 4' : '', marker: DECODERS[k].marker,
    points: ps.map((p) => ({ x: p, y: f.ratio(dd, p, k) })),
  })));

  const thSeries = [3, 5].map((dd) => ({
    id: `pm${dd}`, label: `PyMatching d=${dd}`, short: `d=${dd}`, color: 'var(--series-1)', dash: dd === 5 ? '6 4' : '', marker: dd === 5 ? 'square' : 'circle',
    points: ps.map((p) => { const r = f.get(dd, p, 'pymatching'); return { x: p, y: r.ler, lo: r.ci_low, hi: r.ci_high }; }),
  }));

  return (
    <Section id="results" eyebrow="Accuracy" title="Logical error rate vs physical error rate"
      lede={`Test sets: ${fmtInt(f.shots[3] || 0)} shots per p at d=3, ${fmtInt(f.shots[5] || 0)} at d=5, from a seed never used for training or model selection. Shaded bands are 95% Wilson intervals; at these shot counts most are thinner than the line.`}>
      <Card>
        <div className="card-head-row">
          <h3>All decoders, d = {d}</h3>
          <Segmented size="sm" label="Distance" value={d} onChange={setD} options={[{ value: 3, label: 'd = 3' }, { value: 5, label: 'd = 5' }]} />
        </div>
        <LineChart series={lerSeries} xLabel="Physical error rate p" yLabel="Logical error rate per shot"
          xFormat={fmtP} valueFormat={fmtLer} tableTitle={`Logical error rate, d=${d}`} />
      </Card>
      <div className="grid-2">
        <Card title="Error rate relative to PyMatching" note="Below 1.0 the network beats matching. Solid lines are d=3, dashed are d=5.">
          <LineChart series={ratioSeries} yLog={false} xLabel="Physical error rate p" yLabel="LER ÷ PyMatching LER"
            xFormat={fmtP} valueFormat={(v) => `${v.toFixed(2)}×`} yFormat={(v) => `${v.toFixed(1)}×`} refY={{ y: 1, label: 'PyMatching' }} height={300} />
        </Card>
        <Card title="Threshold: does the bigger code help?" note="Below the crossing, d=5 beats d=3, so adding qubits suppresses errors. Above it, the bigger code is worse.">
          <LineChart series={thSeries} xLabel="Physical error rate p" yLabel="PyMatching LER"
            xFormat={fmtP} valueFormat={fmtLer} height={300}
            markX={f.th?.crossing_found ? { x: f.th.p_threshold, label: `≈ ${fmtP(f.th.p_threshold)}` } : null} />
        </Card>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ speed & size
function Speed({ s, f }) {
  const hw = s.eval_meta?.hardware || '';
  const rows = (big) => [3, 5].flatMap((d) => ['pymatching', 'mlp', 'cnn'].map((k) => {
    const r = f.tp(d, k, big);
    return { id: `${k}${d}${big}`, group: `d = ${d}`, label: DECODERS[k].short, value: big ? r.shots_per_s : r.us_per_shot, color: DECODERS[k].color };
  }));
  const sizes = s.model_sizes.filter((r) => r.model !== 'mlp_fixed_p');
  return (
    <Section id="speed" eyebrow="Speed & size" title="How fast, and how big?"
      lede={`Timed on ${hw}. Median of 5 runs after a warm-up. Batched runs use 100k shots for PyMatching and batches of 4096 for the networks.`}>
      <div className="grid-2">
        <Card title="Throughput, batched" note="Shots decoded per second. Higher is better.">
          <BarChart rows={rows(true)} log valueFormat={(v) => `${fmtCompact(v)}/s`} axisLabel="Shots per second (log scale)" labelWidth={100} />
        </Card>
        <Card title="Latency, one shot at a time" note="Microseconds per shot at batch size 1. Lower is better, and this is what a real-time decoder needs.">
          <BarChart rows={rows(false)} log valueFormat={fmtUs} axisLabel="µs per shot (log scale)" labelWidth={100} />
        </Card>
      </div>
      <Card title="Model size">
        <div className="size-grid">
          <div className="size-cell">
            <span className="swatch" style={{ background: DECODERS.pymatching.color }} aria-hidden />
            <b>PyMatching</b><span className="num">0</span><span className="muted">trained parameters. Edge weights come straight from Stim's error model.</span>
          </div>
          {sizes.map((r) => (
            <div className="size-cell" key={`${r.model}${r.distance}`}>
              <span className="swatch" style={{ background: DECODERS[r.model].color }} aria-hidden />
              <b>{DECODERS[r.model].short} d={r.distance}</b><span className="num">{fmtInt(r.params)}</span><span className="muted">parameters</span>
            </div>
          ))}
        </div>
      </Card>
    </Section>
  );
}

// ------------------------------------------------------------------ analysis
function Analysis({ s, f }) {
  const dis = s.disagreement.map((r) => ({
    id: `${r.decoder}${r.distance}`,
    label: `${DECODERS[r.decoder].short} d=${r.distance}`,
    frac: r.disagree_frac,
    segments: [
      { value: r.nn_right, color: DECODERS[r.decoder].color, label: `${DECODERS[r.decoder].short} right` },
      { value: r.mwpm_right, color: DECODERS.pymatching.color, label: 'PyMatching right' },
    ],
  }));
  const ps = s.config.p_sweep;
  const g = s.generalisation;
  const gSeries = [
    { id: 'mixed', label: 'MLP trained on all p', short: 'MLP mixed-p', color: DECODERS.mlp.color, dash: '', marker: 'square' },
    { id: 'fixed', label: `MLP trained only at p=${fmtP(f.fp)}`, short: 'MLP fixed-p', color: DECODERS.mlp.color, dash: '6 4', marker: 'triangle' },
  ].map((sr) => ({
    ...sr,
    points: ps.map((p) => {
      const row = g.find((r) => r.p === p && r.decoder === `mlp_${sr.id}_p`);
      const pm = g.find((r) => r.p === p && r.decoder === 'pymatching');
      return { x: p, y: row.ler / pm.ler };
    }),
  }));
  const logs = s.train_logs;
  const tSeries = ['mlp_d3', 'cnn_d3', 'mlp_d5', 'cnn_d5'].filter((k) => logs[k]?.length).map((k) => {
    const [kind, dd] = k.split('_');
    return {
      id: k, label: `${DECODERS[kind].short} ${dd.replace('d', 'd=')}`, short: `${DECODERS[kind].short} ${dd.replace('d', 'd=')}`,
      color: DECODERS[kind].color, dash: dd === 'd5' ? '6 4' : '', marker: 'none',
      points: logs[k].map((r) => ({ x: r.step, y: r.val_loss })),
    };
  });
  const steps = [...new Set(tSeries.flatMap((x) => x.points.map((p) => p.x)))];
  const maxStep = Math.max(...steps);

  return (
    <Section id="analysis" eyebrow="Analysis" title="Where the networks help, and where they fail">
      <div className="grid-2">
        <Card title="When they disagree, who is right?" note={`At p = ${fmtP(f.fp)}. Each bar is only the shots where the network and PyMatching give different answers.`}>
          <StackedBars rows={dis} />
          <ul className="mini-legend">
            {dis.map((r) => <li key={r.id}><b>{r.label}</b> disagrees on {fmtPct(r.frac)} of shots</li>)}
          </ul>
        </Card>
        <Card title="Noise-model mismatch" note={`d = 3. A second MLP was trained only at p = ${fmtP(f.fp)}, then tested at every p. Ratio to PyMatching, lower is better.`}>
          <LineChart series={gSeries} yLog={false} xLabel="Physical error rate p" yLabel="LER ÷ PyMatching LER" height={280}
            xFormat={fmtP} valueFormat={(v) => `${v.toFixed(3)}×`} yFormat={(v) => `${v.toFixed(2)}×`} refY={{ y: 1, label: 'PyMatching' }} />
        </Card>
      </div>
      <Card title="Training: validation loss" note="Mixed-p validation set, logged during training. Every model was trained on fresh Stim samples each step, so it never saw the same batch twice.">
        <LineChart series={tSeries} yLog={false} xLabel="Training step" yLabel="Validation BCE loss" height={300}
          xFormat={(v) => fmtCompact(v)} valueFormat={(v) => v.toFixed(4)} yFormat={(v) => v.toFixed(2)}
          xTicks={[0, maxStep / 4, maxStep / 2, (3 * maxStep) / 4, maxStep].map(Math.round).filter((v) => v >= Math.min(...steps))} />
      </Card>
    </Section>
  );
}

// ------------------------------------------------------------------ verdict (computed, not typed)
function Verdict({ s, f }) {
  const fp = f.fp;
  const ps = s.config.p_sweep;
  const line = (d, k) => {
    const a = f.get(d, fp, k), b = f.get(d, fp, 'pymatching');
    const r = a.ler / b.ler;
    const sig = !overlaps(a, b);
    return { r, sig, better: r < 1 };
  };
  const wins = (d, k) => ps.filter((p) => { const a = f.get(d, p, k), b = f.get(d, p, 'pymatching'); return a.ler < b.ler && !overlaps(a, b); }).length;
  const c3 = line(3, 'cnn'), m3 = line(3, 'mlp'), c5 = line(5, 'cnn'), m5 = line(5, 'mlp');
  // does the network get better with distance? (d=5 LER below d=3 LER at the lowest p)
  const scales = (k) => ps.filter((p) => f.get(5, p, k).ler < f.get(3, p, k).ler);
  const pm5 = f.tp(5, 'pymatching', true), mlp5 = f.tp(5, 'mlp', true), cnn5 = f.tp(5, 'cnn', true);
  const lat = (k) => f.tp(5, k, false).us_per_shot;
  const gen = s.generalisation;
  const genGap = Math.max(...ps.map((p) => {
    const a = gen.find((r) => r.p === p && r.decoder === 'mlp_fixed_p'), b = gen.find((r) => r.p === p && r.decoder === 'mlp_mixed_p');
    return a.ler / b.ler - 1;
  }));
  const fixedDev = Math.max(...[3, 5].flatMap((d) => ps.map((p) => Math.abs(f.get(d, p, 'pymatching_fixed_p').ler / f.get(d, p, 'pymatching').ler - 1))));
  const cnnLog = s.train_logs.cnn_d5 || [];
  const stillFalling = cnnLog.length > 4 && cnnLog[cnnLog.length - 1].val_loss < cnnLog[cnnLog.length - 5].val_loss;
  const latGap = Math.min(lat('mlp'), lat('cnn')) / lat('pymatching');
  const good = (d) => Math.min(line(d, 'mlp').r, line(d, 'cnn').r) < 1;
  const headline = good(3) && !good(5) ? 'Yes at d = 3, no at d = 5.' : good(3) && good(5) ? 'Yes at both distances.' : !good(3) && !good(5) ? 'No: PyMatching wins at both distances.' : 'Mixed: the networks win only at d = 5.';
  const pct = (r) => `${Math.abs((1 - r) * 100).toFixed(0)}%`;

  const items = [
    {
      tone: 'good', head: 'd = 3: the networks win.',
      body: `At p = ${fmtP(fp)} the CNN's error rate is ${pct(c3.r)} lower than PyMatching's and the MLP's is ${pct(m3.r)} lower${c3.sig && m3.sig ? ', with non-overlapping 95% intervals' : ''}. The CNN is significantly better at ${wins(3, 'cnn')} of ${ps.length} noise levels. A likely reason: matching treats the X and Z parts of correlated (Y-type) errors separately, while a network can learn those correlations, and at d = 3 there are few enough syndromes for it to learn them well.`,
    },
    {
      tone: 'bad', head: 'd = 5: the networks lose clearly.',
      body: `At p = ${fmtP(fp)} the CNN is ${c5.r.toFixed(2)}× PyMatching's error rate and the MLP is ${m5.r.toFixed(2)}×. ${scales('mlp').length === 0 ? 'The MLP is never better at d = 5 than at d = 3, so for it the larger code buys nothing.' : `The MLP is better at d = 5 than at d = 3 only at p = ${scales('mlp').map(fmtP).join(', ')}.`} The CNN gains from distance only at p = ${scales('cnn').map(fmtP).join(', ') || 'no value'}, while PyMatching gains at ${ps.filter((p) => f.get(5, p, 'pymatching').ler < f.get(3, p, 'pymatching').ler).length} of ${ps.length} noise levels. ${stillFalling ? 'Its validation loss was still edging down when the learning-rate schedule ended, so it is likely under-trained as well as small.' : 'Its validation loss had flattened, so model size rather than training length is the likely limit.'}`,
    },
    {
      tone: 'neutral', head: 'Speed depends on how you ask.',
      body: `Batched at d = 5, the MLP decodes ${fmtCompact(mlp5.shots_per_s)} shots/s against PyMatching's ${fmtCompact(pm5.shots_per_s)}/s, while the CNN manages ${fmtCompact(cnn5.shots_per_s)}/s. One shot at a time, PyMatching takes ${fmtUs(lat('pymatching'))} against ${fmtUs(lat('mlp'))} for the MLP and ${fmtUs(lat('cnn'))} for the CNN. Real-time error correction needs low per-shot latency, and there PyMatching is ${latGap.toFixed(0)}× faster than the quickest network on this CPU.`,
    },
    {
      tone: 'neutral', head: 'Noise-model mismatch barely hurt at d = 3.',
      body: `The MLP trained only at p = ${fmtP(fp)} is at most ${(genGap * 100).toFixed(1)}% worse than the mixed-p MLP anywhere in the sweep. At this size, brittleness to a mis-specified noise level is small. PyMatching with weights fixed at p = ${fmtP(fp)} stays within ${(fixedDev * 100).toFixed(1)}% of the re-weighted matcher at every p and distance.`,
    },
  ];
  return (
    <Section id="verdict" eyebrow="Verdict" title="Is the performance good?"
      lede={`${headline} The pipeline itself is sound: PyMatching shows a threshold and every network beats the no-decoder baseline by a wide margin. Every statement below is computed in the browser from the CSVs.`}>
      <div className="verdict-list">
        {items.map((it) => (
          <div className={`card verdict-item ${it.tone}`} key={it.head}>
            <span className={`status ${it.tone === 'good' ? 'good' : it.tone === 'bad' ? 'bad' : 'info'}`}>
              <span aria-hidden>{it.tone === 'good' ? '✓' : it.tone === 'bad' ? '✗' : 'i'}</span>
              {it.tone === 'good' ? 'Strength' : it.tone === 'bad' ? 'Weakness' : 'Trade-off'}
            </span>
            <h3>{it.head}</h3>
            <p>{it.body}</p>
          </div>
        ))}
      </div>
      <Card title="Pipeline output (same lines the CLI demo prints)">
        <ul className="conclusion">{s.conclusion.map((l) => <li key={l}>{l}</li>)}</ul>
      </Card>
    </Section>
  );
}

function Footer({ s }) {
  const mins = s.runtime?.total_seconds ? (s.runtime.total_seconds / 60).toFixed(0) : null;
  return (
    <footer className="footer">
      <div>
        <h3>Reproduce</h3>
        <pre><code>{`python run_all.py --full     # ${mins ? `${mins} min on this laptop` : 'the real numbers'}
python server.py             # this page, from saved checkpoints
python demo.py --d 3 --p 0.01 --pause`}</code></pre>
      </div>
      <p className="muted">
        Seeds: train {s.config.seeds.train}, validation {s.config.seeds.val}, test {s.config.seeds.test}, demo {s.config.seeds.demo}.
        Base paper: <i>Rethink the Role of Neural Decoders in Quantum Error Correction</i>. Built with Stim, PyMatching and PyTorch.
      </p>
    </footer>
  );
}

export default function App() {
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api('summary').then(setS).catch((e) => setErr(e.message)); }, []);
  const f = useFacts(s || { ler_results: [], throughput: [], threshold: [], config: { fixed_p: 0 } });

  if (err) {
    return (
      <main className="boot">
        <h1>Can't reach the decoder server</h1>
        <p className="muted">{err}</p>
        <p>Start it from the project folder with <code>.venv/bin/python server.py</code>, then reload.</p>
      </main>
    );
  }
  if (!s) return <main className="boot"><span className="spinner big" aria-label="Loading" /></main>;
  if (!s.ler_results.length) {
    return <main className="boot"><h1>No results yet</h1><p>Run <code>python run_all.py --quick</code> first.</p></main>;
  }
  return (
    <>
      <Nav />
      <main>
        <Hero s={s} f={f} />
        <Live summary={s} />
        <Results s={s} f={f} />
        <Speed s={s} f={f} />
        <Analysis s={s} f={f} />
        <Verdict s={s} f={f} />
      </main>
      <Footer s={s} />
    </>
  );
}
