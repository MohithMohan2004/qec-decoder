import { useLayoutEffect, useRef, useState } from 'react';
import { rubberband, useSpring } from './motion.js';

/** Segmented control: the selection thumb glides between options on a critically damped spring. */
export function Segmented({ options, value, onChange, label, size = 'md' }) {
  const refs = useRef([]);
  const [box, setBox] = useState({ x: 0, w: 0 });
  const idx = Math.max(0, options.findIndex((o) => o.value === value));

  useLayoutEffect(() => {
    const measure = () => {
      const el = refs.current[idx];
      if (el) setBox({ x: el.offsetLeft, w: el.offsetWidth });
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [idx, options.length]);

  const x = useSpring(box.x, { damping: 1, response: 0.3 });
  const w = useSpring(box.w, { damping: 1, response: 0.3 });

  const onKey = (e) => {
    const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    const next = options[(idx + dir + options.length) % options.length];
    onChange(next.value);
    refs.current[(idx + dir + options.length) % options.length]?.focus();
  };

  return (
    <div className={`segmented segmented-${size}`} role="radiogroup" aria-label={label} onKeyDown={onKey}>
      {box.w > 0 && <span className="segmented-thumb" style={{ transform: `translateX(${x}px)`, width: w }} aria-hidden />}
      {options.map((o, i) => (
        <button
          key={String(o.value)}
          ref={(el) => { refs.current[i] = el; }}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          tabIndex={o.value === value ? 0 : -1}
          className={o.value === value ? 'on' : ''}
          onPointerDown={() => o.value !== value && onChange(o.value)} /* respond on press, not release */
          onClick={() => o.value !== value && onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/**
 * Slider with 1:1 pointer tracking (respects the grab offset), rubber-banding past the ends and a
 * spring settle on release. `onChange` fires continuously; `onCommit` once on release / key press.
 */
export function Slider({ min, max, step, value, onChange, onCommit, format, label }) {
  const track = useRef(null);
  const drag = useRef(null);
  const [overshoot, setOvershoot] = useState(0);
  const [dragging, setDragging] = useState(false);
  const frac = (value - min) / (max - min);
  const settle = useSpring(dragging ? overshoot : 0, { damping: 0.8, response: 0.3, immediate: dragging });

  const snap = (v) => Math.min(max, Math.max(min, Math.round((v - min) / step) * step + min));
  const fromX = (clientX) => {
    const r = track.current.getBoundingClientRect();
    return { raw: min + ((clientX - r.left) / r.width) * (max - min), width: r.width, left: r.left };
  };

  const down = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const onThumb = e.target.classList.contains('slider-thumb');
    const { raw, width, left } = fromX(e.clientX);
    const thumbX = left + frac * width;
    drag.current = { offset: onThumb ? e.clientX - thumbX : 0, width };
    setDragging(true);
    if (!onThumb) onChange(snap(raw));
  };
  const move = (e) => {
    if (!drag.current) return;
    const { raw, width } = fromX(e.clientX - drag.current.offset);
    const over = raw < min ? (raw - min) / (max - min) * width : raw > max ? (raw - max) / (max - min) * width : 0;
    setOvershoot(rubberband(over, width, 0.25));
    const v = snap(raw);
    if (v !== value) onChange(v);
  };
  const up = () => {
    if (!drag.current) return;
    drag.current = null;
    setDragging(false);
    setOvershoot(0);
    onCommit?.(value);
  };
  const key = (e) => {
    const k = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step, PageUp: step * 5, PageDown: -step * 5 }[e.key];
    let v;
    if (k) v = snap(value + k);
    else if (e.key === 'Home') v = min;
    else if (e.key === 'End') v = max;
    else return;
    e.preventDefault();
    onChange(v);
    onCommit?.(v);
  };

  return (
    <div className="slider">
      <div
        ref={track}
        className={`slider-track${dragging ? ' dragging' : ''}`}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={up}
      >
        <div className="slider-fill" style={{ width: `calc(${frac * 100}% + ${settle}px)` }} />
        <div
          className="slider-thumb"
          role="slider"
          tabIndex={0}
          aria-label={label}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={value}
          aria-valuetext={format(value)}
          onKeyDown={key}
          style={{ left: `calc(${frac * 100}% + ${settle}px)` }}
        />
      </div>
    </div>
  );
}

export function Button({ children, onClick, kind = 'primary', disabled, busy, ...rest }) {
  return (
    <button type="button" className={`btn btn-${kind}${busy ? ' busy' : ''}`} onClick={onClick} disabled={disabled || busy} {...rest}>
      {busy && <span className="spinner" aria-hidden />}
      <span>{children}</span>
    </button>
  );
}
