import { Vector3 } from "three";

const DEG = Math.PI / 180;

/** Lat/lon (degrees) → point on a sphere of `radius`, in the frame where an
 * equirectangular texture on `SphereGeometry` lands at the right place. */
export function toVector(lat: number, lon: number, radius = 1, target = new Vector3()): Vector3 {
  const phi = (90 - lat) * DEG;
  const theta = (lon + 180) * DEG;
  return target.set(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

/** Points of a great-circle arc lifted above the surface, for the propagation arcs. */
export function arcPoints(
  a: { lat: number; lon: number },
  b: { lat: number; lon: number },
  segments = 48,
  lift = 0.18,
): Vector3[] {
  const start = toVector(a.lat, a.lon);
  const end = toVector(b.lat, b.lon);
  const angle = start.angleTo(end);
  const points: Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const s = i / segments;
    const point = new Vector3().copy(start).lerp(end, s);
    // slerp-ish: renormalise the chord point back to the sphere then lift it
    const height = 1 + lift * Math.sin(Math.PI * s) * Math.min(1, angle / 1.2);
    points.push(point.normalize().multiplyScalar(height));
  }
  return points;
}

/** Index of the first element greater than `value` in an ascending array. */
export function countBefore(times: number[], value: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export const easeOut = (x: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 3);
