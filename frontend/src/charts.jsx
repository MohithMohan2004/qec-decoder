// Hand-rolled SVG charts (no chart library: none is installable offline, and SVG keeps full control).
// Conventions from the dataviz method: one y-axis, thin 2px lines, >=8px markers with a surface ring,
// recessive grid, legend for >=2 series plus direct end labels, hover crosshair + tooltip, table view.
import { useMemo, useState } from 'react';
import { useSpring, useWidth } from './motion.js';

// ------------------------------------------------------------------ scales & ticks
const linear = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
const logScale = (d0, d1, r0, r1) => {
  const a = Math.log10(d0), b = Math.log10(d1);
  return (v) => r0 + ((Math.log10(Math.max(v, 1e-12)) - a) / (b - a || 1)) * (r1 - r0);
};

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  return out.length > 7 ? out.filter((v) => /^1/.test(v.toExponential())) : out;
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = 10 ** Math.floor(Math.log10(span / n));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * step0).find((s) => span / s <= n) || step0 * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}

const fmtTick = (v) => {
  if (v === 0) return '0';
  if (v >= 1000) return `${v / 1000}k`.replace('.0k', 'k');
  if (v >= 1) return `${+v.toPrecision(3)}`;
  return `${+v.toPrecision(2)}`;
};

// ------------------------------------------------------------------ markers
export function Marker({ kind, x, y, color, r = 4.5 }) {
  const common = { fill: color, stroke: 'var(--surface-1)', strokeWidth: 2 };
  if (kind === 'none') return null;
  if (kind === 'square') return <rect x={x - r} y={y - r} width={r * 2} height={r * 2} rx={1.5} {...common} />;
  if (kind === 'diamond') return <path d={`M${x},${y - r * 1.25}L${x + r * 1.25},${y}L${x},${y + r * 1.25}L${x - r * 1.25},${y}Z`} {...common} />;
  if (kind === 'triangle') return <path d={`M${x},${y - r * 1.2}L${x + r * 1.15},${y + r * 0.9}L${x - r * 1.15},${y + r * 0.9}Z`} {...common} />;
  return <circle cx={x} cy={y} r={r} {...common} />;
}

export function LegendKey({ s }) {
  return (
    <svg width="26" height="12" aria-hidden className="legend-key">
      <line x1="1" x2="25" y1="6" y2="6" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash || undefined} />
      <Marker kind={s.marker} x={13} y={6} color={s.color} r={3.5} />
    </svg>
  );
}

function Legend({ series, hidden, toggle }) {
  return (
    <div className="legend" role="group" aria-label="Series (click to show or hide)">
      {series.map((s) => (
        <button
          type="button"
          key={s.id}
          className={`legend-item${hidden.has(s.id) ? ' off' : ''}`}
          aria-pressed={!hidden.has(s.id)}
          onClick={() => toggle(s.id)}
        >
          <LegendKey s={s} />
          {s.label}
        </button>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ line chart
export function LineChart({
  series, xLabel, yLabel, yLog = true, xFormat = (v) => v, yFormat = fmtTick, valueFormat = (v) => v,
  height = 340, refY, markX, xTicks, tableTitle = 'Table',
}) {
  const [wrap, width] = useWidth();
  const [hidden, setHidden] = useState(new Set());
  const [hover, setHover] = useState(null);
  const [table, setTable] = useState(false);
  const toggle = (id) => setHidden((h) => { const n = new Set(h); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const vis = series.filter((s) => !hidden.has(s.id));

  const narrow = width < 560;
  const m = { top: 14, right: narrow ? 14 : 112, bottom: 46, left: 58 };
  const iw = width - m.left - m.right, ih = height - m.top - m.bottom;

  const xs = useMemo(() => [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b), [series]);
  const [ylo, yhi] = useMemo(() => {
    const vals = (vis.length ? vis : series).flatMap((s) => s.points.flatMap((p) => [p.lo ?? p.y, p.hi ?? p.y, p.y]))
      .concat(refY != null ? [refY.y] : []).filter((v) => v > 0 || !yLog);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (yLog) { lo /= 1.35; hi *= 1.35; } else { const pad = (hi - lo) * 0.08 || 0.1; lo -= pad; hi += pad; }
    return [lo, hi];
  }, [vis, series, refY, yLog]);

  const x = linear(xs[0], xs[xs.length - 1], 0, iw);
  const y = (yLog ? logScale : linear)(ylo, yhi, ih, 0);
  const yt = yLog ? logTicks(ylo, yhi) : niceTicks(ylo, yhi);
  const xt = xTicks || xs;

  // direct labels at the right end, nudged apart so they never collide
  const labels = useMemo(() => {
    if (narrow) return [];
    const ls = vis.map((s) => { const last = s.points[s.points.length - 1]; return { s, y: y(last.y) }; }).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ls.length; i += 1) if (ls[i].y - ls[i - 1].y < 15) ls[i].y = ls[i - 1].y + 15;
    return ls;
  }, [vis, narrow, ylo, yhi, ih]); // eslint-disable-line react-hooks/exhaustive-deps

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best = xs[0];
    xs.forEach((v) => { if (Math.abs(x(v) - px) < Math.abs(x(best) - px)) best = v; });
    setHover(best);
  };

  const path = (pts, key = 'y') => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(1)},${y(p[key]).toFixed(1)}`).join('');
  const band = (pts) => `${path(pts, 'hi')}${pts.slice().reverse().map((p) => `L${x(p.x).toFixed(1)},${y(p.lo).toFixed(1)}`).join('')}Z`;

  const tipRows = hover == null ? [] : vis.map((s) => ({ s, p: s.points.find((p) => p.x === hover) })).filter((r) => r.p).sort((a, b) => b.p.y - a.p.y);
  const tipLeft = hover != null && m.left + x(hover) > width * 0.6;

  return (
    <div className="chart" ref={wrap}>
      <div className="chart-bar">
        <Legend series={series} hidden={hidden} toggle={toggle} />
        <button type="button" className="link-btn" onClick={() => setTable((t) => !t)} aria-pressed={table}>
          {table ? 'Chart view' : 'Table view'}
        </button>
      </div>
      {table ? (
        <DataTable series={series} xs={xs} xFormat={xFormat} valueFormat={valueFormat} title={tableTitle} />
      ) : (
        <div className="chart-plot">
          <svg width={width} height={height} role="img" aria-label={`${yLabel} versus ${xLabel}`}>
            <g transform={`translate(${m.left},${m.top})`}>
              {yt.map((t) => (
                <g key={t} transform={`translate(0,${y(t)})`}>
                  <line x2={iw} className="grid" />
                  <text x={-8} dy="0.32em" textAnchor="end" className="tick">{yFormat(t)}</text>
                </g>
              ))}
              <line y1={ih} y2={ih} x2={iw} className="axis" />
              {xt.map((t) => (
                <text key={t} x={x(t)} y={ih + 18} textAnchor="middle" className="tick">{xFormat(t)}</text>
              ))}
              <text x={iw / 2} y={ih + 40} textAnchor="middle" className="axis-label">{xLabel}</text>
              <text transform={`translate(${-46},${ih / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{yLabel}</text>

              {refY && (
                <g>
                  <line x2={iw} y1={y(refY.y)} y2={y(refY.y)} className="ref-line" />
                  <text x={4} y={y(refY.y) - 6} className="ref-label">{refY.label}</text>
                </g>
              )}
              {vis.map((s) => s.points[0].lo != null && (
                <path key={`b${s.id}`} d={band(s.points)} fill={s.color} opacity={0.13} />
              ))}
              {vis.map((s) => (
                <path key={`l${s.id}`} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash || undefined} strokeLinejoin="round" />
              ))}
              {vis.map((s) => s.points.map((p) => <Marker key={`${s.id}${p.x}`} kind={s.marker} x={x(p.x)} y={y(p.y)} color={s.color} />))}
              {markX && (
                <g transform={`translate(${x(markX.x)},0)`}>
                  <line y2={ih} className="mark-line" />
                  <text y={-2} x={6} className="mark-label">{markX.label}</text>
                </g>
              )}
              {labels.map(({ s, y: ly }) => (
                <text key={`t${s.id}`} x={iw + 8} y={ly} dy="0.32em" className="direct-label">{s.short || s.label}</text>
              ))}
              {hover != null && <line x1={x(hover)} x2={x(hover)} y2={ih} className="crosshair" />}
              <rect width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} onPointerDown={onMove} />
            </g>
          </svg>
          {hover != null && tipRows.length > 0 && (
            <div
              className="tooltip"
              style={{ top: m.top + 8, left: tipLeft ? undefined : m.left + x(hover) + 14, right: tipLeft ? width - (m.left + x(hover)) + 14 : undefined }}
            >
              <div className="tooltip-title">{xLabel}: {xFormat(hover)}</div>
              {tipRows.map(({ s, p }) => (
                <div key={s.id} className="tooltip-row">
                  <LegendKey s={s} />
                  <span className="tooltip-name">{s.short || s.label}</span>
                  <span className="num">{valueFormat(p.y)}</span>
                  {p.lo != null && <span className="tooltip-ci num">[{valueFormat(p.lo)}, {valueFormat(p.hi)}]</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DataTable({ series, xs, xFormat, valueFormat, title }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <caption className="sr-only">{title}</caption>
        <thead>
          <tr><th scope="col">x</th>{series.map((s) => <th scope="col" key={s.id}>{s.short || s.label}</th>)}</tr>
        </thead>
        <tbody>
          {xs.map((xv) => (
            <tr key={xv}>
              <th scope="row">{xFormat(xv)}</th>
              {series.map((s) => {
                const p = s.points.find((q) => q.x === xv);
                return <td key={s.id} className="num">{p ? valueFormat(p.y) : '–'}{p?.lo != null && <span className="ci"> [{valueFormat(p.lo)}–{valueFormat(p.hi)}]</span>}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ horizontal bars
function Bar({ x0, w, y, h, color }) {
  const sw = useSpring(Math.max(0, w), { damping: 1, response: 0.4 });
  const r = Math.min(4, sw / 2, h / 2);
  // flat at the baseline, 4px rounded data-end
  return <path d={`M${x0},${y}H${x0 + sw - r}Q${x0 + sw},${y} ${x0 + sw},${y + r}V${y + h - r}Q${x0 + sw},${y + h} ${x0 + sw - r},${y + h}H${x0}Z`} fill={color} />;
}

/**
 * rows: [{id, group?, label, value, lo?, hi?, color, marker?}]  log: log x-axis.
 * Rows sharing `group` are drawn as a block with the group name above.
 */
export function BarChart({ rows, log = false, valueFormat, axisLabel, refX, rowHeight = 26, labelWidth = 120, domain }) {
  const [wrap, width] = useWidth();
  const [hover, setHover] = useState(null);
  const groups = [...new Set(rows.map((r) => r.group ?? ''))];
  const lw = width < 480 ? Math.min(labelWidth, 92) : labelWidth;
  const m = { top: 8, right: 74, bottom: 40, left: lw };
  let cursor = 0;
  const layout = [];
  groups.forEach((g) => {
    if (g) cursor += 22;
    rows.filter((r) => (r.group ?? '') === g).forEach((r) => { layout.push({ ...r, cy: cursor }); cursor += rowHeight; });
    cursor += 8;
  });
  const ih = cursor;
  const height = ih + m.top + m.bottom;
  const iw = width - m.left - m.right;
  const vals = rows.flatMap((r) => [r.value, r.hi ?? r.value, r.lo ?? r.value]).concat(refX ? [refX.x] : []).filter((v) => v > 0);
  const lo = domain ? domain[0] : log ? 10 ** Math.floor(Math.log10(Math.min(...vals))) : 0;
  const hi = domain ? domain[1] : log ? 10 ** Math.ceil(Math.log10(Math.max(...vals))) : Math.max(...vals) * 1.1;
  const x = (log ? logScale : linear)(lo, hi, 0, iw);
  const ticks = log ? logTicks(lo, hi).filter((t) => /^1/.test(t.toExponential())) : niceTicks(lo, hi, 5);
  const barH = 12;

  return (
    <div className="chart" ref={wrap}>
      <div className="chart-plot">
        <svg width={width} height={height} role="img" aria-label={axisLabel}>
          <g transform={`translate(${m.left},${m.top})`}>
            {ticks.map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y2={ih} className="grid" />
                <text y={ih + 16} textAnchor="middle" className="tick">{fmtTick(t)}</text>
              </g>
            ))}
            <text x={iw / 2} y={ih + 34} textAnchor="middle" className="axis-label">{axisLabel}</text>
            {groups.map((g) => {
              const first = layout.find((r) => (r.group ?? '') === g);
              return g ? <text key={g} x={-lw + 2} y={first.cy - 8} className="group-label">{g}</text> : null;
            })}
            {layout.map((r) => {
              const by = r.cy + (rowHeight - barH) / 2;
              return (
                <g key={r.id} onPointerEnter={() => setHover(r.id)} onPointerLeave={() => setHover(null)}>
                  <rect x={-lw} y={r.cy} width={width - m.right} height={rowHeight} fill={hover === r.id ? 'var(--hover-wash)' : 'transparent'} />
                  <text x={-10} y={r.cy + rowHeight / 2} dy="0.32em" textAnchor="end" className="bar-label">{r.label}</text>
                  <Bar x0={0} w={x(Math.max(r.value, lo))} y={by} h={barH} color={r.color} />
                  {r.lo != null && (
                    <g className="whisker">
                      <line x1={x(r.lo)} x2={x(r.hi)} y1={by + barH / 2} y2={by + barH / 2} />
                      <line x1={x(r.lo)} x2={x(r.lo)} y1={by + 2} y2={by + barH - 2} />
                      <line x1={x(r.hi)} x2={x(r.hi)} y1={by + 2} y2={by + barH - 2} />
                    </g>
                  )}
                  <text x={x(Math.max(r.hi ?? r.value, r.value, lo)) + 8} y={r.cy + rowHeight / 2} dy="0.32em" className="bar-value num">{valueFormat(r.value)}</text>
                </g>
              );
            })}
            {refX && (
              <g transform={`translate(${x(refX.x)},0)`}>
                <line y2={ih} className="ref-line" />
                <text y={ih - 4} x={4} className="ref-label">{refX.label}</text>
              </g>
            )}
          </g>
        </svg>
      </div>
    </div>
  );
}

/** 100%-stacked horizontal bars with a 2px surface gap between segments. */
export function StackedBars({ rows }) {
  const [wrap, width] = useWidth();
  const [hover, setHover] = useState(null);
  const lw = width < 480 ? 86 : 120;
  const iw = width - lw - 8;
  return (
    <div className="chart" ref={wrap}>
      <svg width={width} height={rows.length * 40 + 4} role="img" aria-label="Who is right when decoders disagree">
        {rows.map((r, i) => {
          const total = r.segments.reduce((a, s) => a + s.value, 0);
          let acc = 0;
          return (
            <g key={r.id} transform={`translate(0,${i * 40})`}>
              <text x={lw - 10} y={16} dy="0.32em" textAnchor="end" className="bar-label">{r.label}</text>
              {r.segments.map((s, j) => {
                const w = (s.value / total) * iw;
                const x0 = lw + acc;
                acc += w;
                const id = `${r.id}-${j}`;
                return (
                  <g key={id} onPointerEnter={() => setHover(id)} onPointerLeave={() => setHover(null)}>
                    <rect x={x0 + (j ? 1 : 0)} y={4} width={Math.max(0, w - (j ? 1 : 0) - (j < r.segments.length - 1 ? 1 : 0))} height={24} rx={j === 0 || j === r.segments.length - 1 ? 4 : 0} fill={s.color} opacity={hover && hover !== id ? 0.55 : 1} />
                    {w > 70 && <text x={x0 + 10} y={16} dy="0.32em" className="seg-label">{`${Math.round((s.value / total) * 100)}%`}</text>}
                    <title>{`${s.label}: ${s.value.toLocaleString('en-US')} shots (${((s.value / total) * 100).toFixed(1)}%)`}</title>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
