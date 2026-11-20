// Motion primitives following skills.md (Apple "Designing Fluid Interfaces"):
//  - springs described by damping ratio + response (seconds), not mass/stiffness/duration
//  - always animate from the *current* value and velocity, so every animation is interruptible
//  - prefers-reduced-motion: jump straight to the target (no slide, no overshoot)
import { useEffect, useRef, useState } from 'react';

const reducedQuery = typeof window !== 'undefined' && window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)')
  : null;

export function prefersReducedMotion() {
  return !!(reducedQuery && reducedQuery.matches);
}

export function useReducedMotion() {
  const [reduced, setReduced] = useState(prefersReducedMotion());
  useEffect(() => {
    if (!reducedQuery) return undefined;
    const on = () => setReduced(reducedQuery.matches);
    reducedQuery.addEventListener('change', on);
    return () => reducedQuery.removeEventListener('change', on);
  }, []);
  return reduced;
}

/** Apple's parameterisation -> unit-mass physics. */
function physics(damping, response) {
  const stiffness = (2 * Math.PI / response) ** 2;
  const friction = (4 * Math.PI * damping) / response;
  return { stiffness, friction };
}

/**
 * Animate a number towards `target` with a spring. Re-targeting mid-flight keeps position
 * AND velocity, so there is never a jump or a velocity "brick wall".
 * `velocity` (units/s) may be handed off from a gesture on release.
 */
export function useSpring(target, { damping = 1, response = 0.35, velocity, immediate = false } = {}) {
  const reduced = useReducedMotion();
  const [value, setValue] = useState(target);
  const state = useRef({ x: target, v: 0, raf: 0, last: 0 });

  useEffect(() => {
    const s = state.current;
    if (velocity !== undefined) s.v = velocity;
    if (immediate || reduced) {
      cancelAnimationFrame(s.raf);
      s.x = target; s.v = 0; s.raf = 0;
      setValue(target);
      return undefined;
    }
    const { stiffness, friction } = physics(damping, response);
    const step = (now) => {
      const dt = Math.min(0.032, (now - (s.last || now)) / 1000) || 1 / 60;
      s.last = now;
      // semi-implicit Euler in 4 sub-steps: stable for stiff, snappy springs
      for (let i = 0; i < 4; i += 1) {
        const h = dt / 4;
        s.v += (-stiffness * (s.x - target) - friction * s.v) * h;
        s.x += s.v * h;
      }
      const settled = Math.abs(s.v) < 1e-3 * (1 + Math.abs(target)) && Math.abs(s.x - target) < 1e-3 * (1 + Math.abs(target));
      if (settled) {
        s.x = target; s.v = 0; s.raf = 0; s.last = 0;
        setValue(target);
        return;
      }
      setValue(s.x);
      s.raf = requestAnimationFrame(step);
    };
    if (!s.raf) { s.last = 0; s.raf = requestAnimationFrame(step); }
    else { cancelAnimationFrame(s.raf); s.raf = requestAnimationFrame(step); }
    return () => { cancelAnimationFrame(s.raf); s.raf = 0; };
  }, [target, damping, response, velocity, immediate, reduced]);

  return value;
}

/** Progressive resistance past a boundary instead of a hard stop (skills.md §9). */
export function rubberband(overshoot, dimension, constant = 0.55) {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}

/** Track width of an element (for responsive SVG charts). */
export function useWidth(initial = 640) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    if (!ref.current) return undefined;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(260, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}
