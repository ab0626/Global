/** Tunables for the 3D globe. Every knob has a sensible default so `<RippleGlobe />`
 * renders well with just data props; override any subset via the `config` prop. */
export type GlobeConfig = {
  /** "white" for demo/branding, "dark" for a space-like backdrop. */
  background: "white" | "dark";
  /** Texture resolution tag: files are looked up as `/textures/earth_<kind>_<quality>.jpg`. */
  textureQuality: 1024 | 2048;
  /** Normal-map strength; 0 = flat, 1 = pronounced relief. */
  terrainExaggeration: number;
  /** 0 = no halo, 1 = strong rim glow. */
  atmosphereIntensity: number;
  /** Render the faint cloud layer (a second textured sphere). */
  clouds: boolean;
  /** Base radius (world units, globe radius = 1) of a country marker at one article. */
  markerSize: number;
  /** Draw arcs from event origin to publisher countries once they light up. */
  arcs: boolean;
  /** 0 = no expanding rings, 1 = full-strength rings. */
  rippleStrength: number;
  /** Seconds a marker takes to pop in / a ring takes to expand. */
  animationSeconds: number;
  /** Degrees per second of idle auto-rotation; 0 disables. */
  autoRotate: number;
  /** Light the Earth from the real sub-solar point of the playhead's UTC time
   * (dark mode also draws the sun, moon and stars); false = fixed studio light. */
  realSun: boolean;
  camera: { distance: number; minDistance: number; maxDistance: number; fov: number };
  /** Palette; defaults follow the Ripple white/grey/red/blue/black scheme. */
  colors: { origin: string; active: string; arc: string; atmosphere: string; ring: string };
};

export const DEFAULT_CONFIG: GlobeConfig = {
  background: "white",
  textureQuality: 2048,
  terrainExaggeration: 0.55,
  atmosphereIntensity: 0.6,
  clouds: true,
  markerSize: 0.009,
  arcs: true,
  rippleStrength: 0.8,
  animationSeconds: 1.4,
  autoRotate: 0.6,
  realSun: true,
  camera: { distance: 3.6, minDistance: 1.6, maxDistance: 6, fov: 40 },
  colors: {
    origin: "#1f4fd8",
    active: "#d61e1e",
    arc: "#d61e1e",
    atmosphere: "#6e9ee8",
    ring: "#d61e1e",
  },
};

export function resolveConfig(partial?: Partial<GlobeConfig>): GlobeConfig {
  return {
    ...DEFAULT_CONFIG,
    ...partial,
    camera: { ...DEFAULT_CONFIG.camera, ...partial?.camera },
    colors: { ...DEFAULT_CONFIG.colors, ...partial?.colors },
  };
}
