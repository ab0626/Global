import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import type { GeoFeature } from "../api";

function radius(count: number, max: number): number {
  return 5 + 20 * Math.sqrt(count / Math.max(max, 1));
}

function colour(tone: number): string {
  if (tone <= -5) return "#ef4444";
  if (tone < 0) return "#fb923c";
  if (tone < 3) return "#38bdf8";
  return "#4ade80";
}

export function CoverageMap({
  features,
  onPick,
}: {
  features: GeoFeature[];
  onPick: (place: string) => void;
}) {
  const max = features.reduce((best, feature) => Math.max(best, feature.properties.count), 0);
  return (
    <MapContainer center={[20, 5]} zoom={2} minZoom={2} worldCopyJump className="map">
      <TileLayer
        attribution="&copy; OpenStreetMap"
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {features.map((feature) => {
        const [lon, lat] = feature.geometry.coordinates;
        const { name, count, tone, theme } = feature.properties;
        return (
          <CircleMarker
            key={`${name}-${lat}-${lon}`}
            center={[lat, lon]}
            radius={radius(count, max)}
            pathOptions={{ color: colour(tone), fillColor: colour(tone), fillOpacity: 0.35 }}
            eventHandlers={{ click: () => onPick(name) }}
          >
            <Popup>
              <strong>{name}</strong>
              <br />
              {count.toLocaleString()} articles · mean tone {tone?.toFixed(1)}
              <br />
              top CAMEO action: {theme}
              <br />
              <em>click the circle to filter on this location</em>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
