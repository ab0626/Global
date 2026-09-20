import { useMemo, useRef } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import {
  AdditiveBlending,
  BackSide,
  Color,
  type Mesh,
  NormalBlending,
  SRGBColorSpace,
  ShaderMaterial,
  TextureLoader,
  Vector2,
} from "three";
import type { GlobeConfig } from "./config";

const ATMOSPHERE_VERTEX = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;

// rim glow on a slightly larger back-facing shell: strongest at the limb, fading inward
const ATMOSPHERE_FRAGMENT = /* glsl */ `
  uniform vec3 color;
  uniform float intensity;
  uniform float falloff;
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    float rim = 1.0 - abs(dot(vNormal, vView));
    float glow = pow(rim, falloff) * intensity;
    gl_FragColor = vec4(color, glow);
  }
`;

type Props = { config: GlobeConfig };

export function Earth({ config }: Props) {
  const q = config.textureQuality;
  const [albedo, normal, specular] = useLoader(TextureLoader, [
    `/textures/earth_atmos_${q}.jpg`,
    `/textures/earth_normal_${q}.jpg`,
    `/textures/earth_specular_${q}.jpg`,
  ]);
  const normalScale = useMemo(
    () => new Vector2(config.terrainExaggeration, config.terrainExaggeration),
    [config.terrainExaggeration],
  );
  const dark = config.background === "dark";

  const atmosphere = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: ATMOSPHERE_VERTEX,
        fragmentShader: ATMOSPHERE_FRAGMENT,
        uniforms: {
          color: { value: new Color(config.colors.atmosphere) },
          intensity: { value: config.atmosphereIntensity * (dark ? 1.4 : 0.7) },
          falloff: { value: dark ? 3.0 : 3.4 },
        },
        transparent: true,
        depthWrite: false,
        side: BackSide,
        blending: dark ? AdditiveBlending : NormalBlending,
      }),
    [config.colors.atmosphere, config.atmosphereIntensity, dark],
  );

  return (
    <group>
      <mesh>
        <sphereGeometry args={[1, 96, 96]} />
        <meshPhongMaterial
          map={albedo}
          map-colorSpace={SRGBColorSpace}
          map-anisotropy={8}
          normalMap={normal}
          normalScale={normalScale}
          specularMap={specular}
          specular={new Color(dark ? "#405a80" : "#5f7ba3")}
          shininess={14}
          emissive={new Color(dark ? "#000000" : "#e8ecf2")}
          emissiveIntensity={dark ? 0 : 0.03}
        />
      </mesh>
      {config.clouds && <Clouds />}
      <mesh scale={dark ? 1.12 : 1.07} material={atmosphere}>
        <sphereGeometry args={[1, 64, 64]} />
      </mesh>
    </group>
  );
}

function Clouds() {
  const clouds = useLoader(TextureLoader, "/textures/earth_clouds_1024.png");
  const mesh = useRef<Mesh>(null);
  useFrame((_, dt) => {
    if (mesh.current) mesh.current.rotation.y += dt * 0.006;
  });
  return (
    <mesh ref={mesh} scale={1.008}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshLambertMaterial map={clouds} map-colorSpace={SRGBColorSpace} transparent opacity={0.35} depthWrite={false} />
    </mesh>
  );
}
