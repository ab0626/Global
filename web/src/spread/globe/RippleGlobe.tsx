import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { Vector3 } from "three";
import { DEFAULT_CONFIG, resolveConfig, type GlobeConfig } from "./config";
import { Earth } from "./Earth";
import { toVector } from "./geo";
import { CountryMarkers, OriginMarker } from "./Markers";
import type { CountryMarker, Hover, Origin } from "./types";

export type RippleGlobeProps = {
  markers: CountryMarker[];
  origin: Origin | null;
  /** Playhead seconds; markers with `t <= now` are lit. */
  now: number;
  onHover?: (hover: Hover | null) => void;
  config?: Partial<GlobeConfig>;
};

/** Hero 3D Earth: textured sphere with relief + atmosphere, a blue event-origin
 * beacon (where the event happened) and red publisher-country markers (where the
 * outlets covering it are based) lighting up in observed-time order. */
export function RippleGlobe({ markers, origin, now, onHover, config }: RippleGlobeProps) {
  const resolved = useMemo(() => resolveConfig(config), [config]);
  const nowRef = useRef(now);
  useEffect(() => {
    nowRef.current = now;
  }, [now]);
  const hover = onHover ?? (() => undefined);
  const dark = resolved.background === "dark";

  return (
    <Canvas
      className={`ripple-globe ${resolved.background}`}
      dpr={[1, 1.75]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ fov: resolved.camera.fov, near: 0.05, far: 50, position: [0, 0, resolved.camera.distance] }}
      onPointerMissed={() => hover(null)}
    >
      <ambientLight intensity={dark ? 0.25 : 0.7} />
      <directionalLight position={[-4, 2.5, 3]} intensity={dark ? 2.4 : 2.2} color="#ffffff" />
      <directionalLight position={[4, -1, -3]} intensity={dark ? 0.15 : 0.35} color="#dbe6ff" />
      <Suspense fallback={<Placeholder dark={dark} />}>
        <Earth config={resolved} />
      </Suspense>
      {origin && <OriginMarker origin={origin} config={resolved} onHover={hover} />}
      <CountryMarkers markers={markers} origin={origin} config={resolved} now={nowRef} onHover={hover} />
      <Camera focus={origin ?? (markers[0] ?? null)} config={resolved} />
    </Canvas>
  );
}

function Placeholder({ dark }: { dark: boolean }) {
  return (
    <mesh>
      <sphereGeometry args={[1, 48, 48]} />
      <meshStandardMaterial color={dark ? "#1b2230" : "#e9edf2"} />
    </mesh>
  );
}

/** Orbit controls plus a fly-to whenever the focus (event origin) changes; a drag
 * cancels the flight. Gentle auto-rotation after a few idle seconds. */
function Camera({ focus, config }: { focus: { lat: number; lon: number } | null; config: GlobeConfig }) {
  const controls = useRef<OrbitControlsImpl>(null);
  const camera = useThree((s) => s.camera);
  const target = useRef<Vector3 | null>(null);
  const dragging = useRef(false);
  const idleSince = useRef(0);

  useEffect(() => {
    if (!focus) return;
    target.current = toVector(focus.lat, focus.lon, camera.position.length() || config.camera.distance);
  }, [focus, camera, config.camera.distance]);

  useFrame(({ clock }, dt) => {
    const c = controls.current;
    if (!c) return;
    if (dragging.current) idleSince.current = clock.elapsedTime;
    if (target.current) {
      const radius = target.current.length();
      camera.position.lerp(target.current, Math.min(1, dt * 3)).setLength(radius);
      camera.lookAt(0, 0, 0);
      if (camera.position.distanceTo(target.current) < 0.002) target.current = null;
      idleSince.current = clock.elapsedTime;
    } else if (config.autoRotate > 0 && clock.elapsedTime - idleSince.current > 4) {
      c.autoRotate = true;
      c.autoRotateSpeed = config.autoRotate;
    } else {
      c.autoRotate = false;
    }
    c.update();
  });

  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      enableDamping
      dampingFactor={0.08}
      rotateSpeed={0.55}
      zoomSpeed={0.6}
      minDistance={config.camera.minDistance}
      maxDistance={config.camera.maxDistance}
      onStart={() => {
        dragging.current = true;
        target.current = null;
      }}
      onEnd={() => {
        dragging.current = false;
      }}
    />
  );
}

export { DEFAULT_CONFIG };
export type { CountryMarker, GlobeConfig, Hover, Origin };
