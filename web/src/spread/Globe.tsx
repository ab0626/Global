import { useEffect, useMemo, useRef, useState } from "react";
import { geoOrthographic, geoPath, geoGraticule10, type GeoPermissibleObjects } from "d3-geo";
import { feature } from "topojson-client";
import type { Topology, GeometryCollection } from "topojson-specification";
import land110 from "world-atlas/countries-110m.json";

export type Marker = {
  key: string;
  lat: number;
  lon: number;
  /** Seconds since animation start at which the marker appears. */
  t: number;
  size: number;
  label: string;
  kind: "publisher" | "event";
};

type Props = {
  markers: Marker[];
  /** Current playhead in the same units as Marker.t. */
  now: number;
  focus?: [number, number] | null;
  onHover?: (marker: Marker | null) => void;
};

const WORLD = land110 as unknown as Topology<{ countries: GeometryCollection }>;
const COUNTRIES = feature(WORLD, WORLD.objects.countries) as GeoPermissibleObjects;
const SPHERE: GeoPermissibleObjects = { type: "Sphere" };
const GRATICULE = geoGraticule10();
const POP_SECONDS = 1.2;

export function Globe({ markers, now, focus, onHover }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState(720);
  const [rotation, setRotation] = useState<[number, number]>([-20, -25]);
  const drag = useRef<{ x: number; y: number; r: [number, number] } | null>(null);
  const userRotated = useRef(false);

  useEffect(() => {
    const element = canvas.current?.parentElement;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const box = entry.contentRect;
      setSize(Math.max(320, Math.floor(Math.min(box.width, box.height))));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (focus && !userRotated.current) setRotation([-focus[1], -focus[0]]);
  }, [focus]);

  const projection = useMemo(
    () =>
      geoOrthographic()
        .translate([size / 2, size / 2])
        .scale(size / 2 - 8)
        .clipAngle(90)
        .rotate(rotation),
    [size, rotation],
  );

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const dpr = window.devicePixelRatio || 1;
    element.width = size * dpr;
    element.height = size * dpr;
    const ctx = element.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const path = geoPath(projection, ctx);

    ctx.beginPath();
    path(SPHERE);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 1.25;
    ctx.strokeStyle = "#111111";
    ctx.stroke();

    ctx.beginPath();
    path(GRATICULE);
    ctx.lineWidth = 0.4;
    ctx.strokeStyle = "#e3e3e3";
    ctx.stroke();

    ctx.beginPath();
    path(COUNTRIES);
    ctx.fillStyle = "#fafafa";
    ctx.fill();
    ctx.lineWidth = 0.7;
    ctx.strokeStyle = "#111111";
    ctx.stroke();

    const [cx, cy] = [size / 2, size / 2];
    const radius = size / 2 - 8;
    // markers are sorted by t, so the first future marker ends the pass
    for (const marker of markers) {
      if (marker.t > now) break;
      const point = projection([marker.lon, marker.lat]);
      if (!point) continue;
      // hidden on the far hemisphere when clipAngle drops the point
      const distance = Math.hypot(point[0] - cx, point[1] - cy);
      if (distance > radius + 1) continue;
      const age = now - marker.t;
      const pop = Math.min(1, age / POP_SECONDS);
      const ease = 1 - Math.pow(1 - pop, 3);
      const base = marker.size;
      if (marker.kind === "event") {
        ctx.beginPath();
        ctx.arc(point[0], point[1], base + 4 * (1 - ease), 0, Math.PI * 2);
        ctx.strokeStyle = "#1f4fd8";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(point[0], point[1], base * 0.55, 0, Math.PI * 2);
        ctx.fillStyle = "#1f4fd8";
        ctx.fill();
        continue;
      }
      if (pop < 1) {
        ctx.beginPath();
        ctx.arc(point[0], point[1], base + 18 * ease, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(214, 30, 30, ${0.6 * (1 - ease)})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(point[0], point[1], base * (0.4 + 0.6 * ease), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(214, 30, 30, 0.85)";
      ctx.fill();
    }
  }, [markers, now, projection, size]);

  const pick = (x: number, y: number): Marker | null => {
    let best: Marker | null = null;
    let bestDistance = 12;
    for (const marker of markers) {
      if (marker.t > now) break;
      const point = projection([marker.lon, marker.lat]);
      if (!point) continue;
      const distance = Math.hypot(point[0] - x, point[1] - y);
      if (distance < bestDistance + marker.size) {
        best = marker;
        bestDistance = distance;
      }
    }
    return best;
  };

  return (
    <canvas
      ref={canvas}
      className="globe"
      style={{ width: size, height: size }}
      onPointerDown={(e) => {
        drag.current = { x: e.clientX, y: e.clientY, r: rotation };
        userRotated.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerLeave={() => onHover?.(null)}
      onPointerMove={(e) => {
        if (drag.current) {
          const k = 0.35 * (720 / size);
          const [r0, r1] = drag.current.r;
          setRotation([
            r0 + (e.clientX - drag.current.x) * k,
            Math.max(-90, Math.min(90, r1 - (e.clientY - drag.current.y) * k)),
          ]);
          return;
        }
        const box = e.currentTarget.getBoundingClientRect();
        onHover?.(pick(e.clientX - box.left, e.clientY - box.top));
      }}
    />
  );
}
