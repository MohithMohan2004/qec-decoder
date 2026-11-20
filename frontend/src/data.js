// API client, formatting helpers and the decoder identity table (colour follows the decoder, always).

export async function api(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`/api/${path}${qs ? `?${qs}` : ''}`);
  const body = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok || body.error) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

// Categorical slots 1-4 of the validated palette (see styles.css), fixed order, never cycled.
// Secondary encoding (dash + marker) so identity never relies on colour alone.
export const DECODERS = {
  pymatching: { label: 'PyMatching', short: 'PyMatching', color: 'var(--series-1)', dash: '', marker: 'circle' },
  mlp: { label: 'MLP', short: 'MLP', color: 'var(--series-2)', dash: '', marker: 'square' },
  cnn: { label: 'CNN', short: 'CNN', color: 'var(--series-3)', dash: '', marker: 'diamond' },
  pymatching_fixed_p: { label: 'PyMatching (fixed p=0.005 weights)', short: 'PyM fixed-p', color: 'var(--series-4)', dash: '6 4', marker: 'triangle' },
  no_decoder: { label: 'No decoder', short: 'No decoder', color: 'var(--neutral-series)', dash: '2 4', marker: 'none' },
};

export const fmtLer = (x) => (x == null ? '–' : x >= 0.1 ? x.toFixed(3) : x >= 0.01 ? x.toFixed(4) : x.toPrecision(3));
export const fmtPct = (x, digits = 2) => `${(x * 100).toFixed(digits)}%`;
export const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
export const fmtCompact = (x) => {
  if (x >= 1e6) return `${(x / 1e6).toFixed(x >= 1e7 ? 0 : 1)}M`;
  if (x >= 1e3) return `${(x / 1e3).toFixed(x >= 1e4 ? 0 : 1)}k`;
  return `${Math.round(x)}`;
};
export const fmtUs = (us) => (us >= 1000 ? `${(us / 1000).toFixed(1)} ms` : us >= 10 ? `${us.toFixed(0)} µs` : `${us.toFixed(1)} µs`);
export const fmtP = (p) => `${+(p * 100).toFixed(2)}%`;

/** Index ler_results rows as get(d, p, decoder). */
export function indexLer(rows) {
  const m = new Map();
  rows.forEach((r) => m.set(`${r.distance}|${r.p}|${r.decoder}`, r));
  return (d, p, dec) => m.get(`${d}|${p}|${dec}`);
}

/** Two Wilson intervals overlap -> "not statistically distinguishable". */
export const overlaps = (a, b) => a.ci_low <= b.ci_high && b.ci_low <= a.ci_high;
