import { useEffect, useMemo, useRef, type ReactNode, type RefObject } from "react";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import {
  BufferGeometry,
  Color,
  type Group,
  Line,
  LineBasicMaterial,
  Matrix4,
  type Mesh,
  type MeshBasicMaterial,
  Quaternion,
  Vector3,
} from "three";
import type { GlobeConfig } from "./config";
import { arcPoints, countBefore, easeOut, toVector } from "./geo";
import type { CountryMarker, Hover, Origin } from "./types";

type Shared = {
  config: GlobeConfig;
  /** Current playhead in seconds; read every frame, never triggers React renders. */
  now: RefObject<number>;
  onHover: (hover: Hover | null) => void;
};

const UP = new Vector3(0, 1, 0);

/** Positions a flat child (circle/ring in XY) tangent to the sphere at lat/lon, +Z outward. */
function Surface({
  lat,
  lon,
  lift = 0.002,
  children,
}: {
  lat: number;
  lon: number;
  lift?: number;
  children: ReactNode;
}) {
  const [position, quaternion] = useMemo(() => {
    const p = toVector(lat, lon, 1 + lift);
    const outward = new Matrix4().lookAt(p.clone().multiplyScalar(2), p, UP);
    return [p, new Quaternion().setFromRotationMatrix(outward)] as const;
  }, [lat, lon, lift]);
  return (
    <group position={position} quaternion={quaternion}>
      {children}
    </group>
  );
}

/* ------------------------------------------------------------------ origin */

export function OriginMarker({ origin, config, onHover }: Omit<Shared, "now"> & { origin: Origin }) {
  const rings = useRef<Group>(null);
  const core = useRef<Mesh>(null);
  const RINGS = 3;
  const color = useMemo(() => new Color(config.colors.origin), [config.colors.origin]);
  const period = 2.6;
  const base = config.markerSize * 1.6;

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    rings.current?.children.forEach((ring, i) => {
      const phase = (((t / period + i / RINGS) % 1) + 1) % 1;
      ring.scale.setScalar(1 + phase * 6 * config.rippleStrength);
      ((ring as Mesh).material as MeshBasicMaterial).opacity =
        config.rippleStrength * 0.55 * (1 - phase) * (1 - phase);
    });
    if (core.current) core.current.scale.setScalar(1 + 0.08 * Math.sin(t * 3));
  });

  return (
    <Surface lat={origin.lat} lon={origin.lon} lift={0.004}>
      <group ref={rings}>
        {Array.from({ length: RINGS }, (_, i) => (
          <mesh key={i}>
            <ringGeometry args={[base * 0.9, base * 1.05, 48]} />
            <meshBasicMaterial color={color} transparent opacity={0} depthWrite={false} />
          </mesh>
        ))}
      </group>
      <mesh
        ref={core}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation();
          onHover({ kind: "origin", origin, x: e.clientX, y: e.clientY });
        }}
        onPointerMove={(e: ThreeEvent<PointerEvent>) =>
          onHover({ kind: "origin", origin, x: e.clientX, y: e.clientY })
        }
        onPointerOut={() => onHover(null)}
      >
        <circleGeometry args={[base, 32]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <mesh>
        <ringGeometry args={[base * 1.5, base * 1.62, 48]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} depthWrite={false} />
      </mesh>
      {/* pin so the origin reads as a beacon at glancing angles */}
      <mesh position={[0, 0, 0.03]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.0025, 0.0025, 0.06, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </Surface>
  );
}

/* --------------------------------------------------------------- countries */

export function CountryMarkers({
  markers,
  origin,
  config,
  now,
  onHover,
}: Shared & { markers: CountryMarker[]; origin: Origin | null }) {
  return (
    <group>
      {markers.map((m) => (
        <Country key={m.code} marker={m} config={config} now={now} onHover={onHover} />
      ))}
      {config.arcs && origin && markers.map((m) => (
        <Arc key={m.code} marker={m} origin={origin} config={config} now={now} />
      ))}
    </group>
  );
}

function Country({ marker, config, now, onHover }: Shared & { marker: CountryMarker }) {
  const dot = useRef<Mesh>(null);
  const ring = useRef<Mesh>(null);
  const idle = useRef<Mesh>(null);
  const active = useMemo(() => new Color(config.colors.active), [config.colors.active]);
  const ringColor = useMemo(() => new Color(config.colors.ring), [config.colors.ring]);
  const size = config.markerSize;

  useFrame(() => {
    const t = now.current ?? 0;
    const age = t - marker.t;
    const on = age >= 0;
    const pop = easeOut(age / config.animationSeconds);
    const count = on ? Math.max(1, countBefore(marker.articleTimes, t)) : 0;
    if (dot.current) {
      dot.current.visible = on;
      // grows with log(articles); popping in overshoots slightly then settles
      const grow = 1 + 0.16 * Math.log2(count);
      const overshoot = 1 + 0.35 * Math.sin(Math.PI * Math.min(1, pop));
      dot.current.scale.setScalar(on ? grow * overshoot * (0.2 + 0.8 * pop) : 0.001);
    }
    if (ring.current) {
      const phase = age / (config.animationSeconds * 1.6);
      const show = on && phase < 1 && config.rippleStrength > 0;
      ring.current.visible = show;
      if (show) {
        ring.current.scale.setScalar(1 + easeOut(phase) * 7 * config.rippleStrength);
        (ring.current.material as MeshBasicMaterial).opacity = 0.7 * (1 - phase);
      }
    }
    if (idle.current) idle.current.visible = !on;
  });

  return (
    <Surface lat={marker.lat} lon={marker.lon}>
      <mesh ref={idle}>
        <circleGeometry args={[size * 0.45, 16]} />
        <meshBasicMaterial color="#9aa0a6" transparent opacity={0.55} />
      </mesh>
      <mesh ref={ring}>
        <ringGeometry args={[size * 0.95, size * 1.1, 40]} />
        <meshBasicMaterial color={ringColor} transparent opacity={0} depthWrite={false} />
      </mesh>
      <mesh
        ref={dot}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation();
          onHover({ kind: "country", marker, x: e.clientX, y: e.clientY });
        }}
        onPointerMove={(e: ThreeEvent<PointerEvent>) =>
          onHover({ kind: "country", marker, x: e.clientX, y: e.clientY })
        }
        onPointerOut={() => onHover(null)}
      >
        <circleGeometry args={[size, 32]} />
        <meshBasicMaterial color={active} transparent opacity={0.92} />
      </mesh>
    </Surface>
  );
}

/* -------------------------------------------------------------------- arcs */

function Arc({
  marker,
  origin,
  config,
  now,
}: {
  marker: CountryMarker;
  origin: Origin;
  config: GlobeConfig;
  now: RefObject<number>;
}) {
  const line = useMemo(() => {
    const material = new LineBasicMaterial({
      color: config.colors.arc,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const object = new Line(new BufferGeometry().setFromPoints(arcPoints(origin, marker)), material);
    object.visible = false;
    return object;
  }, [origin, marker, config.colors.arc]);
  useEffect(
    () => () => {
      line.geometry.dispose();
      line.material.dispose();
    },
    [line],
  );
  const ref = useRef<Line<BufferGeometry, LineBasicMaterial>>(null);
  useFrame(() => {
    if (!ref.current) return;
    const age = (now.current ?? 0) - marker.t;
    // flashes in as the country lights up, then fades out over a few seconds
    const alpha = age < 0 ? 0 : 0.5 * easeOut(age / config.animationSeconds) * Math.exp(-age / 4);
    ref.current.material.opacity = alpha;
    ref.current.visible = alpha > 0.005;
  });
  return <primitive ref={ref} object={line} />;
}
