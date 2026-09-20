import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Stars } from "@react-three/drei";
import { AdditiveBlending, Color, DirectionalLight, Group, Vector3 } from "three";
import { moonPoint, subsolarPoint, toVector } from "./geo";

const SUN_DISTANCE = 34;
const MOON_DISTANCE = 26;

type SkyProps = {
  /** Absolute UTC epoch (ms) the globe should be lit for; null keeps the default studio light. */
  time: React.RefObject<number | null>;
  dark: boolean;
  /** Fallback light direction when `time` is null. */
  fallback: [number, number, number];
};

/** Key light follows the real sub-solar point of the playhead's UTC time so the
 * lit hemisphere matches the clock; in dark mode a sun disc, a moon and a starfield
 * are drawn far behind the Earth, so whichever body the camera faces is visible. */
export function Sky({ time, dark, fallback }: SkyProps) {
  const light = useRef<DirectionalLight>(null);
  const sun = useRef<Group>(null);
  const moon = useRef<Group>(null);
  const target = useMemo(() => new Vector3(), []);
  const scratch = useMemo(() => new Vector3(), []);
  const fallbackVec = useMemo(() => new Vector3(...fallback).normalize(), [fallback]);

  useFrame((_, dt) => {
    const t = time.current;
    if (t == null) {
      target.copy(fallbackVec);
    } else {
      const s = subsolarPoint(t);
      toVector(s.lat, s.lon, 1, target);
    }
    const l = light.current;
    if (l) {
      l.position.lerp(scratch.copy(target).multiplyScalar(10), Math.min(1, dt * 4));
    }
    if (sun.current) sun.current.position.copy(l ? l.position : target).setLength(SUN_DISTANCE);
    if (moon.current && t != null) {
      const m = moonPoint(t);
      moon.current.position.lerp(toVector(m.lat, m.lon, MOON_DISTANCE, scratch), Math.min(1, dt * 4));
      moon.current.lookAt(0, 0, 0);
    }
  });

  return (
    <>
      <directionalLight ref={light} position={[fallback[0] * 10, fallback[1] * 10, fallback[2] * 10]} intensity={dark ? 2.6 : 2.2} color="#fff6e8" />
      {dark && (
        <>
          <Stars radius={16} depth={6} count={3200} factor={1.4} saturation={0} fade speed={0.15} />
          <group ref={sun}>
            <SunDisc />
          </group>
          <group ref={moon}>
            <MoonDisc />
          </group>
        </>
      )}
    </>
  );
}

function SunDisc() {
  return (
    <>
      <mesh>
        <sphereGeometry args={[1.0, 24, 24]} />
        <meshBasicMaterial color={new Color("#fff4d6")} toneMapped={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[2.0, 24, 24]} />
        <meshBasicMaterial color={new Color("#ffd27a")} transparent opacity={0.16} blending={AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[3.4, 24, 24]} />
        <meshBasicMaterial color={new Color("#ffb347")} transparent opacity={0.05} blending={AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
    </>
  );
}

function MoonDisc() {
  return (
    <mesh>
      <sphereGeometry args={[0.6, 24, 24]} />
      <meshStandardMaterial color={new Color("#c9ccd3")} roughness={1} metalness={0} />
    </mesh>
  );
}
