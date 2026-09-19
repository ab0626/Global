import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Facets, GdeltEvent, TimelineSeries, ToneBin } from "../api";
import { formatStamp, parseStamp } from "../api";

const AXIS = { stroke: "#64748b", fontSize: 11 };
const TOOLTIP_STYLE = {
  background: "#0b1220",
  border: "1px solid #1e293b",
  borderRadius: 8,
  fontSize: 12,
  color: "#e2e8f0",
};

export function Panel({
  title,
  subtitle,
  children,
  className,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className ?? ""}`}>
      <header>
        <h2>{title}</h2>
        {subtitle && <span>{subtitle}</span>}
      </header>
      {children}
    </section>
  );
}

type TimelinePoint = { date: string; label: string; [series: string]: string | number };

function mergeSeries(timeline: TimelineSeries[]): TimelinePoint[] {
  const rows = new Map<string, TimelinePoint>();
  for (const series of timeline) {
    for (const point of series.data) {
      const existing = rows.get(point.date) ?? {
        date: point.date,
        label: parseStamp(point.date).toISOString().slice(5, 16).replace("T", " "),
      };
      existing[series.series] = point.value;
      rows.set(point.date, existing);
    }
  }
  return [...rows.values()].sort((a, b) => a.date.localeCompare(b.date));
}

export function VolumeTimeline({
  volume,
  raw,
  onBrush,
}: {
  volume: TimelineSeries[];
  raw: TimelineSeries[];
  onBrush: (from: string, to: string) => void;
}) {
  const points = mergeSeries([...volume, ...raw]);
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart
        data={points}
        onClick={(state) => {
          const index = state?.activeTooltipIndex;
          if (index === undefined || index === null) return;
          const from = points[Math.max(0, Number(index) - 2)];
          const to = points[Math.min(points.length - 1, Number(index) + 2)];
          if (from && to) onBrush(from.date, to.date);
        }}
      >
        <defs>
          <linearGradient id="vol" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.7} />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#1e293b" vertical={false} />
        <XAxis dataKey="label" tick={AXIS} minTickGap={40} />
        <YAxis yAxisId="left" tick={AXIS} width={44} />
        <YAxis yAxisId="right" orientation="right" tick={AXIS} width={52} />
        <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={{ color: "#94a3b8" }} />
        <Area
          yAxisId="left"
          type="monotone"
          dataKey="Volume Intensity"
          name="% of monitored articles"
          stroke="#38bdf8"
          fill="url(#vol)"
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="Article Count"
          stroke="#f472b6"
          dot={false}
          strokeWidth={1.5}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function ToneTimeline({ timeline }: { timeline: TimelineSeries[] }) {
  const points = mergeSeries(timeline);
  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={points}>
        <CartesianGrid stroke="#1e293b" vertical={false} />
        <XAxis dataKey="label" tick={AXIS} minTickGap={40} />
        <YAxis tick={AXIS} width={44} />
        <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={{ color: "#94a3b8" }} />
        <Area type="monotone" dataKey="Average Tone" stroke="#a78bfa" fill="#a78bfa33" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function ToneHistogram({
  bins,
  onSelect,
}: {
  bins: ToneBin[];
  onSelect: (bin: ToneBin) => void;
}) {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={bins}>
        <CartesianGrid stroke="#1e293b" vertical={false} />
        <XAxis dataKey="bin" tick={AXIS} />
        <YAxis tick={AXIS} width={44} />
        <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#1e293b55" }} />
        <Bar dataKey="count" onClick={(_, index) => onSelect(bins[index])}>
          {bins.map((bin) => (
            <Cell key={bin.bin} fill={bin.bin < -2 ? "#f87171" : bin.bin > 2 ? "#4ade80" : "#64748b"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function FacetList({
  facets,
  onPick,
}: {
  facets: Facets;
  onPick: (operator: string, value: string) => void;
}) {
  const groups: [string, string, { value: string; count: number }[]][] = [
    ["CAMEO themes", "theme", facets.themes],
    ["Locations", "location", facets.locations],
    ["Actors", "actor", facets.actors],
    ["Domains", "domainis", facets.domains],
    ["Publisher country (ccTLD)", "sourcecountry", facets.sourcecountries],
    ["Source language", "sourcelang", facets.languages],
  ];
  return (
    <div className="facets">
      {groups.map(([label, operator, values]) => (
        <div key={label}>
          <h3>{label}</h3>
          <ul>
            {values.slice(0, 8).map((entry) => (
              <li key={entry.value}>
                <button onClick={() => onPick(operator, entry.value)}>
                  <span className="facet-value">{entry.value}</span>
                  <span className="facet-count">{entry.count.toLocaleString()}</span>
                </button>
              </li>
            ))}
            {values.length === 0 && <li className="empty">no matches</li>}
          </ul>
        </div>
      ))}
    </div>
  );
}

export function EventTable({
  events,
  onSelect,
}: {
  events: GdeltEvent[];
  onSelect: (event: GdeltEvent) => void;
}) {
  return (
    <table className="grid">
      <thead>
        <tr>
          <th>CAMEO action</th>
          <th>Actors</th>
          <th>Location</th>
          <th className="numeric">Articles</th>
          <th className="numeric">Goldstein</th>
          <th className="numeric">Tone</th>
        </tr>
      </thead>
      <tbody>
        {events.map((event) => (
          <tr key={event.globaleventid} onClick={() => onSelect(event)}>
            <td>
              <span className={`quad quad-${event.quadclass}`}>{event.quadlabel}</span>
              <div className="muted">
                {event.rootlabel} ({event.eventcode})
              </div>
            </td>
            <td>
              {event.actor1 || "—"}
              {event.actor2 ? ` → ${event.actor2}` : ""}
            </td>
            <td>{event.location || "—"}</td>
            <td className="numeric">{event.numarticles.toLocaleString()}</td>
            <td className="numeric">{event.goldstein?.toFixed(1) ?? "—"}</td>
            <td className="numeric">{event.avgtone?.toFixed(1) ?? "—"}</td>
          </tr>
        ))}
        {events.length === 0 && (
          <tr>
            <td colSpan={6} className="empty">
              no events in this window
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

export function ArticleTable({
  articles,
  onEvent,
}: {
  articles: { url: string; title: string; domain: string; seendate: string; tone: number | null; language: string; sourcecountry: string; themes: string[]; eventids: number[] }[];
  onEvent: (id: number) => void;
}) {
  return (
    <table className="grid">
      <thead>
        <tr>
          <th>Article (title derived from URL slug)</th>
          <th>Domain</th>
          <th>Seen (UTC)</th>
          <th className="numeric">Tone</th>
          <th>Events</th>
        </tr>
      </thead>
      <tbody>
        {articles.map((article) => (
          <tr key={article.url}>
            <td>
              <a href={article.url} target="_blank" rel="noreferrer">
                {article.title}
              </a>
              <div className="muted">{article.themes.slice(0, 3).join(" · ")}</div>
            </td>
            <td>
              {article.domain}
              <div className="muted">
                {[article.language, article.sourcecountry].filter(Boolean).join(" · ")}
              </div>
            </td>
            <td>{formatStamp(article.seendate)}</td>
            <td className={`numeric ${(article.tone ?? 0) < 0 ? "negative" : "positive"}`}>
              {article.tone?.toFixed(1) ?? "—"}
            </td>
            <td>
              <button className="link" onClick={() => onEvent(article.eventids[0])} disabled={!article.eventids.length}>
                {article.eventids.length} linked
              </button>
            </td>
          </tr>
        ))}
        {articles.length === 0 && (
          <tr>
            <td colSpan={5} className="empty">
              no articles match this query
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
