import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import {
  analyticsCountry,
  analyticsEvents,
  analyticsMatrix,
  analyticsPair,
  eventTypes,
  type AnalyticsQuery,
  type CountryBaseline,
  type CountryOverview,
  type EventsResponse,
  type MagnitudeSummary,
  type MatrixCell,
  type MatrixResponse,
  type PairResponse,
  type ResponseEvent,
  type ResponseSummary,
  type TypeSummary,
} from "./api";

/** What the globe should show for the current analytics selection. `focus` is the blue
 * beacon, `related` the red markers; arcs between them are observed media-attention
 * relationships, never a claim of causal transmission. */
export type StatsScene = {
  focus: { code: string; label: string; lines: [string, string][] } | null;
  related: { code: string; label: string; lines: [string, string][] }[];
  legend: { focus: string | null; related: string };
};

export type StatsMode = "overview" | "pair" | "compare";
type Metric = "median" | "mean" | "coverage" | "count";
/** Outcome of the last finished request, tagged with the selection key it answered; a
 * selection whose key differs is still loading. */
type Status = { key: string; kind: "ready" } | { key: string; kind: "error"; message: string };

const METRICS: { key: Metric; label: string }[] = [
  { key: "median", label: "median response" },
  { key: "mean", label: "average response" },
  { key: "coverage", label: "coverage rate" },
  { key: "count", label: "covered events" },
];
const DEFAULT_COMPARE = ["GM", "FR", "JA", "BR", "US"];
const MAX_COMPARE = 12;
const GLOBE_RELATED = 12;

type Filters = {
  level: "family" | "incident";
  eventTypes: string[];
  minEventReports: number;
  start: string;
  end: string;
  reference: AnalyticsQuery["reference"];
  minCovered: number;
  minEffective: number;
};

const DEFAULT_FILTERS: Filters = {
  level: "family",
  eventTypes: [],
  minEventReports: 0,
  start: "",
  end: "",
  reference: "origin_preferred",
  minCovered: 5,
  minEffective: 15,
};

function toQuery(f: Filters): AnalyticsQuery {
  const iso = (v: string) => (v ? `${v}${v.length === 16 ? ":00" : ""}Z` : undefined);
  return {
    level: f.level,
    event_type: f.eventTypes.length ? f.eventTypes : undefined,
    min_event_effective_reports: f.minEventReports || undefined,
    start: iso(f.start),
    end: iso(f.end),
    reference: f.reference,
    min_covered_events: f.minCovered,
    min_effective_reports: f.minEffective,
  };
}

const hours = (v: number | null | undefined, digits = 1) =>
  v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)} h`;
const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);
const ci = (low: number | null, high: number | null, unit: string) =>
  low == null || high == null ? "—" : `${low.toFixed(1)}–${high.toFixed(1)}${unit}`;
const ciPct = (low: number | null, high: number | null) =>
  low == null || high == null ? "—" : `${(low * 100).toFixed(0)}–${(high * 100).toFixed(0)}%`;

/** Marker/tooltip lines for a summary; latency is omitted when unsupported. */
function summaryLines(s: ResponseSummary): [string, string][] {
  const lines: [string, string][] = [
    ["coverage", `${pct(s.coverage_rate)} (${s.covered_events}/${s.eligible_events})`],
  ];
  if (s.support.status === "ok") {
    lines.unshift(["median response", hours(s.median_response_hours)], ["average", hours(s.mean_response_hours)]);
  } else lines.push(["latency", `insufficient support (N=${s.covered_events})`]);
  return lines;
}

type Props = {
  baseline: Map<string, CountryBaseline>;
  onScene: (scene: StatsScene | null) => void;
  /** Return to the event view: story family, optionally one incident of it. */
  onOpenEvent: (familyId: number, incidentId: number | null) => void;
  utc: (iso: string) => string;
};

export function CountryStats({ baseline, onScene, onOpenEvent, utc }: Props) {
  const [mode, setMode] = useState<StatsMode>("overview");
  const [country, setCountry] = useState("GM");
  const [origin, setOrigin] = useState("FR");
  const [destination, setDestination] = useState("GM");
  const [compare, setCompare] = useState<string[]>(DEFAULT_COMPARE);
  const [metric, setMetric] = useState<Metric>("median");
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [types, setTypes] = useState<string[]>([]);
  const [status, setStatus] = useState<Status>({ key: "", kind: "ready" });
  const [overview, setOverview] = useState<CountryOverview | null>(null);
  const [pair, setPair] = useState<PairResponse | null>(null);
  const [matrix, setMatrix] = useState<MatrixResponse | null>(null);
  const [drill, setDrill] = useState<{ cell: MatrixCell; events: EventsResponse } | null>(null);
  const [breakdown, setBreakdown] = useState<Breakdown | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "median", desc: false });
  const request = useRef(0);
  const query = useMemo(() => toQuery(filters), [filters]);
  const name = (code: string) => baseline.get(code)?.country_name ?? code;

  useEffect(() => {
    eventTypes()
      .then((r) => setTypes(r.types.map((t) => t.type)))
      .catch(() => setTypes([]));
  }, []);

  const selectionKey = JSON.stringify([mode, country, origin, destination, compare, query]);
  const loading = status.key !== selectionKey;

  useEffect(() => {
    const id = ++request.current;
    const stale = () => id !== request.current;
    const run = async () => {
      if (mode === "overview") {
        const r = await analyticsCountry(country, query);
        if (stale()) return;
        setOverview(r);
      } else if (mode === "pair") {
        const r = await analyticsPair(origin, destination, query);
        if (stale()) return;
        setPair(r);
      } else {
        const r = compare.length ? await analyticsMatrix(compare, compare, query) : null;
        if (stale()) return;
        setMatrix(r);
      }
      setDrill(null);
      setBreakdown(null);
    };
    run()
      .then(() => {
        if (!stale()) setStatus({ key: selectionKey, kind: "ready" });
      })
      .catch((e: Error) => {
        if (!stale()) setStatus({ key: selectionKey, kind: "error", message: e.message });
      });
  }, [mode, country, origin, destination, compare, query, selectionKey]);

  useEffect(() => {
    const label = (code: string) => baseline.get(code)?.country_name ?? code;
    onScene(buildScene(mode, { overview, pair, matrix, compare, name: label }));
  }, [mode, overview, pair, matrix, compare, baseline, onScene]);

  const codes = useMemo(
    () =>
      [...baseline.values()]
        .map((c) => ({ code: c.publisher_country, name: c.country_name ?? c.publisher_country }))
        .sort((a, b) => a.name.localeCompare(b.name)),
    [baseline],
  );

  const showPair = (o: string, d: string) => {
    setOrigin(o);
    setDestination(d);
    setMode("pair");
  };

  async function openCell(cell: MatrixCell) {
    if (drill?.cell === cell) {
      setDrill(null);
      return;
    }
    try {
      const events = await analyticsEvents(
        cell.origin_country,
        cell.destination_country,
        cell.kind === "domestic" ? "domestic" : "foreign",
        query,
      );
      setDrill({ cell, events });
    } catch (e) {
      setStatus({ key: selectionKey, kind: "error", message: (e as Error).message });
    }
  }

  async function openBreakdown(title: string, args: BreakdownArgs) {
    if (breakdown?.title === title) {
      setBreakdown(null);
      return;
    }
    try {
      const events = await analyticsEvents(args.origin, args.destination, args.kind, { ...query, ...args.narrow });
      setBreakdown({ title, events });
    } catch (e) {
      setStatus({ key: selectionKey, kind: "error", message: (e as Error).message });
    }
  }

  return (
    <div className="country-stats">
      <nav className="stats-modes" aria-label="country stats mode">
        {(
          [
            ["overview", "Country overview"],
            ["pair", "Country ↔ Country"],
            ["compare", "Compare countries"],
          ] as [StatsMode, string][]
        ).map(([m, label]) => (
          <button key={m} className={`tab ${mode === m ? "active" : ""}`} onClick={() => setMode(m)}>
            {label}
          </button>
        ))}
      </nav>

      {mode === "overview" && (
        <div className="picker-row">
          <CountryPicker label="Publisher country" value={country} codes={codes} onChange={setCountry} />
        </div>
      )}
      {mode === "pair" && (
        <div className="picker-row">
          <CountryPicker label="Events originating in" value={origin} codes={codes} onChange={setOrigin} />
          <span className="arrow">→</span>
          <CountryPicker label="Publisher country" value={destination} codes={codes} onChange={setDestination} />
          <button className="link" onClick={() => showPair(destination, origin)}>
            View {name(destination)} → {name(origin)}
          </button>
        </div>
      )}
      {mode === "compare" && (
        <div className="picker-row wrap">
          {compare.map((c) => (
            <span key={c} className="chip selectable">
              {name(c)}
              <button aria-label={`remove ${name(c)}`} onClick={() => setCompare(compare.filter((x) => x !== c))}>
                ×
              </button>
            </span>
          ))}
          {compare.length < MAX_COMPARE && (
            <CountryPicker
              label="Add country"
              value=""
              codes={codes.filter((c) => !compare.includes(c.code))}
              onChange={(code) => code && setCompare([...compare, code])}
            />
          )}
        </div>
      )}

      <details className="filters">
        <summary>Filters · {filterSummary(filters)}</summary>
        <div className="filter-grid">
          <label>
            level
            <select
              value={filters.level}
              onChange={(e) => setFilters({ ...filters, level: e.target.value as Filters["level"] })}
            >
              <option value="family">story family (one observation per story)</option>
              <option value="incident">incident (drilldown; correlated)</option>
            </select>
          </label>
          <label>
            timing reference
            <select
              value={filters.reference}
              onChange={(e) => setFilters({ ...filters, reference: e.target.value as Filters["reference"] })}
            >
              <option value="origin_preferred">origin onset, world onset as fallback</option>
              <option value="origin_only">origin onset only</option>
              <option value="world">world onset for every event</option>
            </select>
          </label>
          <label>
            event types (any of)
            <select
              multiple
              size={5}
              value={filters.eventTypes}
              onChange={(e) =>
                setFilters({ ...filters, eventTypes: [...e.target.selectedOptions].map((o) => o.value) })
              }
            >
              {types.map((t) => (
                <option key={t} value={t}>
                  {t.replaceAll("_", " ")}
                </option>
              ))}
            </select>
          </label>
          <label>
            min. event size (effective reports)
            <input
              type="number"
              min={0}
              value={filters.minEventReports}
              onChange={(e) => setFilters({ ...filters, minEventReports: Number(e.target.value) })}
            />
          </label>
          <label>
            events from (UTC)
            <input
              type="datetime-local"
              value={filters.start}
              onChange={(e) => setFilters({ ...filters, start: e.target.value })}
            />
          </label>
          <label>
            events before (UTC)
            <input type="datetime-local" value={filters.end} onChange={(e) => setFilters({ ...filters, end: e.target.value })} />
          </label>
          <label>
            support: min. covered events
            <input
              type="number"
              min={1}
              value={filters.minCovered}
              onChange={(e) => setFilters({ ...filters, minCovered: Math.max(1, Number(e.target.value)) })}
            />
          </label>
          <label>
            support: min. effective reports
            <input
              type="number"
              min={0}
              value={filters.minEffective}
              onChange={(e) => setFilters({ ...filters, minEffective: Math.max(0, Number(e.target.value)) })}
            />
          </label>
        </div>
        <button className="link small" onClick={() => setFilters(DEFAULT_FILTERS)}>
          reset filters
        </button>
      </details>

      {status.kind === "error" && <div className="notice error">{status.message}</div>}
      {loading && <div className="notice">computing…</div>}

      {mode === "overview" && overview && (
        <OverviewPanel
          data={overview}
          onOrigin={(o) => showPair(o, overview.publisher_country)}
          breakdown={breakdown}
          onBreakdown={openBreakdown}
          utc={utc}
          onOpenEvent={onOpenEvent}
        />
      )}
      {mode === "pair" && pair && (
        <PairPanel data={pair} utc={utc} onOpenEvent={onOpenEvent} onReverse={() => showPair(destination, origin)} />
      )}
      {mode === "compare" && compare.length === 0 && (
        <div className="notice">Add at least one country to compare.</div>
      )}
      {mode === "compare" && matrix && compare.length > 0 && (
        <ComparePanel
          data={matrix}
          metric={metric}
          onMetric={setMetric}
          sort={sort}
          onSort={setSort}
          drill={drill}
          onCell={openCell}
          onPair={showPair}
          onOpenEvent={onOpenEvent}
          utc={utc}
        />
      )}

      {(overview || pair || matrix) && (
        <p className="muted small caveats">
          Observed media response in GDELT: onset = 3rd outlet and 10th-percentile document of a
          publisher country; response = destination onset − origin-country onset (or − world onset when
          flagged <em>world fallback</em>); domestic = onset − observed event start. Publisher country is
          where outlets are based, not the audience or the event location. Negative values mean the
          destination reached onset first. Latency is conditional on coverage; coverage is reported
          separately. Not a measure of when anyone learned about an event.
        </p>
      )}
    </div>
  );
}

function filterSummary(f: Filters) {
  const parts: string[] = [f.level];
  if (f.eventTypes.length) parts.push(`${f.eventTypes.length} type${f.eventTypes.length > 1 ? "s" : ""}`);
  if (f.minEventReports) parts.push(`≥${f.minEventReports} reports`);
  if (f.start || f.end) parts.push("date range");
  if (f.reference !== "origin_preferred") parts.push(f.reference.replace("_", " "));
  parts.push(`support ≥${f.minCovered} events`);
  return parts.join(" · ");
}

/* ------------------------------------------------------------------ pickers */

function CountryPicker({
  label,
  value,
  codes,
  onChange,
}: {
  label: string;
  value: string;
  codes: { code: string; name: string }[];
  onChange: (code: string) => void;
}) {
  const byName = useMemo(() => new Map(codes.map((c) => [c.name.toLowerCase(), c.code])), [codes]);
  const byCode = useMemo(() => new Map(codes.map((c) => [c.code, c.name])), [codes]);
  /** Text being typed; null shows the selected country's name. */
  const [draft, setDraft] = useState<string | null>(null);
  const text = draft ?? (value ? (byCode.get(value) ?? value) : "");
  const listId = `countries-${label.replaceAll(/\W+/g, "-").toLowerCase()}`;
  const commit = (raw: string) => {
    const t = raw.trim();
    if (!t) {
      setDraft(null);
      return;
    }
    const code = byName.get(t.toLowerCase()) ?? (byCode.has(t.toUpperCase()) ? t.toUpperCase() : null);
    if (code) {
      onChange(code);
      setDraft(null);
    }
  };
  return (
    <label className="picker">
      <span>{label}</span>
      <input
        list={listId}
        value={text}
        placeholder="country name or FIPS code"
        onChange={(e) => {
          setDraft(e.target.value);
          if (byName.has(e.target.value.trim().toLowerCase())) commit(e.target.value);
        }}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit((e.target as HTMLInputElement).value);
          }
        }}
      />
      <datalist id={listId}>
        {codes.map((c) => (
          <option key={c.code} value={c.name}>
            {c.code}
          </option>
        ))}
      </datalist>
    </label>
  );
}

/* ------------------------------------------------------------------ summary */

function SummaryCard({
  title,
  kicker,
  summary,
  domestic = false,
}: {
  title: ReactNode;
  kicker?: string;
  summary: ResponseSummary;
  domestic?: boolean;
}) {
  const s = summary;
  const ok = s.support.status === "ok";
  return (
    <section className="summary-card">
      {kicker && <span className="result-kind">{kicker}</span>}
      <h3>{title}</h3>
      {!ok && (
        <p className="insufficient">
          Insufficient support — N = {s.covered_events} covered of {s.eligible_events} eligible
          {s.covered_events > 0 && ` · ${s.effective_reports} effective reports`} (needs ≥{" "}
          {s.support.min_covered_events} events, ≥ {s.support.min_effective_reports} reports)
        </p>
      )}
      <dl className="stats">
        <div>
          <dt>{domestic ? "median observed domestic response" : "median response"}</dt>
          <dd>{ok ? hours(s.median_response_hours) : "—"}</dd>
        </div>
        <div>
          <dt>average</dt>
          <dd>{ok ? hours(s.mean_response_hours) : "—"}</dd>
        </div>
        <div>
          <dt>coverage</dt>
          <dd>{pct(s.coverage_rate)}</dd>
        </div>
        <div>
          <dt>eligible / covered</dt>
          <dd className="time">
            {s.eligible_events} / {s.covered_events}
          </dd>
        </div>
        <div>
          <dt>p25 – p75</dt>
          <dd className="time">{ok ? ci(s.p25_response_hours, s.p75_response_hours, " h") : "—"}</dd>
        </div>
        <div>
          <dt>median 95% CI</dt>
          <dd className="time">{ok ? ci(s.latency_ci_low, s.latency_ci_high, " h") : "—"}</dd>
        </div>
        <div>
          <dt>coverage 95% CI</dt>
          <dd className="time">{ciPct(s.coverage_ci_low, s.coverage_ci_high)}</dd>
        </div>
        <div>
          <dt>fastest / slowest</dt>
          <dd className="time">
            {ok ? `${hours(s.fastest_response_hours)} / ${hours(s.slowest_response_hours)}` : "—"}
          </dd>
        </div>
        <div>
          <dt>{domestic ? "effective reports" : "world-fallback events"}</dt>
          <dd className="time">{domestic ? s.effective_reports : s.world_fallback_events}</dd>
        </div>
      </dl>
    </section>
  );
}

function BreakdownTable<T extends ResponseSummary>({
  title,
  rows,
  label,
  onRow,
  isActive,
}: {
  title: string;
  rows: T[];
  label: (row: T) => string;
  onRow?: (row: T) => void;
  isActive?: (row: T) => boolean;
}) {
  if (rows.length === 0) return null;
  return (
    <>
      <h3>{title}</h3>
      <table className="countries analytics">
        <thead>
          <tr>
            <th></th>
            <th>median</th>
            <th>avg</th>
            <th>coverage</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const ok = r.support.status === "ok";
            return (
              <tr
                key={i}
                className={`${ok ? "" : "pending"} ${isActive?.(r) ? "active" : ""}`}
                onClick={onRow ? () => onRow(r) : undefined}
              >
                <td>{onRow ? <button className="link">{label(r)}</button> : label(r)}</td>
                <td>{ok ? hours(r.median_response_hours) : "n/s"}</td>
                <td>{ok ? hours(r.mean_response_hours) : "n/s"}</td>
                <td>{pct(r.coverage_rate)}</td>
                <td>
                  {r.covered_events}/{r.eligible_events}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

/* ----------------------------------------------------------------- overview */

/** Events behind one breakdown row (event type or event-size bin). */
type Breakdown = { title: string; events: EventsResponse };

type BreakdownArgs = {
  origin?: string;
  destination?: string;
  kind: "foreign" | "domestic";
  narrow: Partial<AnalyticsQuery>;
};

function OverviewPanel({
  data,
  onOrigin,
  breakdown,
  onBreakdown,
  utc,
  onOpenEvent,
}: {
  data: CountryOverview;
  onOrigin: (code: string) => void;
  breakdown: Breakdown | null;
  onBreakdown: (title: string, args: BreakdownArgs) => void;
  utc: (iso: string) => string;
  onOpenEvent: Props["onOpenEvent"];
}) {
  const me = data.publisher_country;
  const typeLabel = (r: TypeSummary) => r.event_types.replaceAll("_", " ");
  const byType = (kind: "foreign" | "domestic") => (r: TypeSummary) =>
    onBreakdown(`${data.publisher_country_name} · ${kind} · ${typeLabel(r)}`, {
      destination: me,
      kind,
      narrow: { event_type: [r.event_types] },
    });
  const bySize = (r: MagnitudeSummary) =>
    onBreakdown(`${data.publisher_country_name} · foreign · ${r.magnitude}`, {
      destination: me,
      kind: "foreign",
      narrow: {
        min_event_effective_reports: Math.max(r.min_event_effective_reports, data.filters.min_event_effective_reports),
        max_event_effective_reports: r.max_event_effective_reports ?? undefined,
      },
    });
  const active = (title: string) => breakdown?.title === title;
  return (
    <section className="detail">
      <h2>{data.publisher_country_name}</h2>
      <SummaryCard title="Response to foreign news" kicker="publisher country" summary={data.foreign} />
      <SummaryCard title="Observed domestic response" kicker="events located in the same country" summary={data.domestic} domestic />
      <BreakdownTable
        title="Foreign response by origin country"
        rows={data.by_origin}
        label={(r) => r.origin_country_name ?? r.origin_country}
        onRow={(r) => onOrigin(r.origin_country)}
      />
      <BreakdownTable
        title="Foreign response by event type"
        rows={data.by_event_type}
        label={typeLabel}
        onRow={byType("foreign")}
        isActive={(r) => active(`${data.publisher_country_name} · foreign · ${typeLabel(r)}`)}
      />
      <BreakdownTable
        title="Foreign response by event size"
        rows={data.by_magnitude}
        label={(r) => r.magnitude}
        onRow={bySize}
        isActive={(r) => active(`${data.publisher_country_name} · foreign · ${r.magnitude}`)}
      />
      <BreakdownTable
        title="Domestic response by event type"
        rows={data.domestic_by_event_type}
        label={typeLabel}
        onRow={byType("domestic")}
        isActive={(r) => active(`${data.publisher_country_name} · domestic · ${typeLabel(r)}`)}
      />
      <p className="muted small">
        n/s = insufficient support under the current gate; coverage is still shown. Origin rows open the
        country pair; type and size rows list the events behind them.
      </p>
      {breakdown && (
        <div className="drill">
          <h3>{breakdown.title}</h3>
          <EventsTable
            title={`Events behind this row (${breakdown.events.total})`}
            events={breakdown.events.events}
            utc={utc}
            onOpenEvent={onOpenEvent}
            compact
          />
        </div>
      )}
    </section>
  );
}

/* --------------------------------------------------------------------- pair */

function PairPanel({
  data,
  utc,
  onOpenEvent,
  onReverse,
}: {
  data: PairResponse;
  utc: (iso: string) => string;
  onOpenEvent: Props["onOpenEvent"];
  onReverse: () => void;
}) {
  const f = data.forward;
  const r = data.reverse;
  const arrow = (b: { origin_country_name: string; destination_country_name: string }) =>
    `${b.origin_country_name} → ${b.destination_country_name}`;
  return (
    <section className="detail">
      <h2>
        {f.origin_country_name} {r ? "↔" : "→"} {f.destination_country_name}
      </h2>
      <div className={`pair-grid ${r ? "" : "single"}`}>
        <SummaryCard title={arrow(f)} kicker={f.kind === "domestic" ? "observed domestic response" : "events in origin → publishers in destination"} summary={f.summary} domestic={f.kind === "domestic"} />
        {r && (
          <SummaryCard title={arrow(r)} kicker="reverse direction (measured separately)" summary={r.summary} />
        )}
      </div>
      {r && (
        <p className="scope">
          <button className="link" onClick={onReverse}>
            View {arrow(r)} in detail →
          </button>
        </p>
      )}
      <BreakdownTable title={`${arrow(f)} by event type`} rows={f.by_event_type} label={(x) => x.event_types.replaceAll("_", " ")} />
      <EventsTable title={`Contributing events (${f.events.length})`} events={f.events} utc={utc} onOpenEvent={onOpenEvent} />
    </section>
  );
}

/* ------------------------------------------------------------------ compare */

type SortKey = "name" | "median" | "mean" | "coverage" | "covered" | "domestic";

function metricValue(s: ResponseSummary, metric: Metric): number | null {
  if (metric === "coverage") return s.coverage_rate;
  if (metric === "count") return s.covered_events;
  if (s.support.status !== "ok") return null;
  return metric === "median" ? s.median_response_hours : s.mean_response_hours;
}

function ComparePanel({
  data,
  metric,
  onMetric,
  sort,
  onSort,
  drill,
  onCell,
  onPair,
  onOpenEvent,
  utc,
}: {
  data: MatrixResponse;
  metric: Metric;
  onMetric: (m: Metric) => void;
  sort: { key: SortKey; desc: boolean };
  onSort: (s: { key: SortKey; desc: boolean }) => void;
  drill: { cell: MatrixCell; events: EventsResponse } | null;
  onCell: (cell: MatrixCell) => void;
  onPair: (o: string, d: string) => void;
  onOpenEvent: Props["onOpenEvent"];
  utc: (iso: string) => string;
}) {
  const cells = new Map(data.cells.map((c) => [`${c.origin_country}>${c.destination_country}`, c]));
  const values = data.cells.map((c) => metricValue(c, metric)).filter((v): v is number => v != null);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const shade = (v: number | null) => {
    if (v == null || values.length === 0) return 0;
    if (hi === lo) return 0.5;
    const x = (v - lo) / (hi - lo);
    return metric === "median" || metric === "mean" ? 1 - x : x; // faster / more = stronger
  };
  const fmt = (v: number | null) =>
    v == null ? "—" : metric === "coverage" ? pct(v) : metric === "count" ? String(v) : hours(v);

  const rows = [...data.destination_rows].sort((a, b) => {
    const val = (r: typeof a): number | string => {
      if (sort.key === "name") return r.publisher_country_name;
      if (sort.key === "coverage") return r.foreign.coverage_rate ?? -1;
      if (sort.key === "covered") return r.foreign.covered_events;
      if (sort.key === "domestic") return r.domestic.median_response_hours ?? Number.POSITIVE_INFINITY;
      const v = sort.key === "median" ? r.foreign.median_response_hours : r.foreign.mean_response_hours;
      return v ?? Number.POSITIVE_INFINITY;
    };
    const x = val(a);
    const y = val(b);
    const c = typeof x === "string" && typeof y === "string" ? x.localeCompare(y) : Number(x) - Number(y);
    return sort.desc ? -c : c;
  });
  const header = (key: SortKey, label: string) => (
    <th>
      <button className="link sort" onClick={() => onSort({ key, desc: sort.key === key ? !sort.desc : key !== "name" && key !== "median" && key !== "mean" && key !== "domestic" })}>
        {label}
        {sort.key === key ? (sort.desc ? " ↓" : " ↑") : ""}
      </button>
    </th>
  );

  return (
    <section className="detail">
      <h2>Response matrix</h2>
      <p className="muted small">
        Rows: where events happened. Columns: publisher country. Diagonal: observed domestic response.
        Muted cells fail the support gate; click any cell for the events behind it.
      </p>
      <div className="metric-switch">
        {METRICS.map((m) => (
          <button key={m.key} className={`tab ${metric === m.key ? "active" : ""}`} onClick={() => onMetric(m.key)}>
            {m.label}
          </button>
        ))}
      </div>
      <div className="matrix-scroll">
        <table className="matrix">
          <thead>
            <tr>
              <th className="corner">origin ↓ · publisher →</th>
              {data.destinations.map((d) => (
                <th key={d.code} title={d.name}>
                  {d.code}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.origins.map((o) => (
              <tr key={o.code}>
                <th title={o.name}>{o.name}</th>
                {data.destinations.map((d) => {
                  const cell = cells.get(`${o.code}>${d.code}`);
                  if (!cell) return <td key={d.code}>—</td>;
                  const v = metricValue(cell, metric);
                  const ok = cell.support.status === "ok";
                  const active = drill?.cell === cell;
                  return (
                    <td
                      key={d.code}
                      className={`${cell.kind} ${ok ? "" : "unsupported"} ${active ? "active" : ""}`}
                      style={{ "--heat": shade(v) } as CSSProperties}
                      title={cellTitle(cell)}
                      onClick={() => void onCell(cell)}
                    >
                      {ok || metric === "coverage" || metric === "count" ? fmt(v) : "n/s"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {drill && (
        <>
          <h3>
            {drill.cell.kind === "domestic"
              ? `${drill.cell.origin_country_name} · observed domestic response`
              : `${drill.cell.origin_country_name} → ${drill.cell.destination_country_name}`}
            {" · "}
            <button className="link" onClick={() => onPair(drill.cell.origin_country, drill.cell.destination_country)}>
              open as pair →
            </button>
          </h3>
          <EventsTable title={`Events behind this cell (${drill.events.total})`} events={drill.events.events} utc={utc} onOpenEvent={onOpenEvent} compact />
        </>
      )}

      <h3>Destination comparison (response to foreign events)</h3>
      <table className="countries analytics">
        <thead>
          <tr>
            {header("name", "country")}
            {header("median", "median")}
            {header("mean", "avg")}
            {header("coverage", "coverage")}
            {header("covered", "N")}
            {header("domestic", "domestic median")}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const ok = r.foreign.support.status === "ok";
            const dom = r.domestic.support.status === "ok";
            return (
              <tr key={r.publisher_country} className={ok ? "" : "pending"}>
                <td>{r.publisher_country_name}</td>
                <td>{ok ? hours(r.foreign.median_response_hours) : "n/s"}</td>
                <td>{ok ? hours(r.foreign.mean_response_hours) : "n/s"}</td>
                <td title={`95% CI ${ciPct(r.foreign.coverage_ci_low, r.foreign.coverage_ci_high)}`}>{pct(r.foreign.coverage_rate)}</td>
                <td>
                  {r.foreign.covered_events}/{r.foreign.eligible_events}
                </td>
                <td>{dom ? hours(r.domestic.median_response_hours) : `n/s (${r.domestic.covered_events})`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        N = covered / eligible foreign story families. Domestic median = onset − observed event start for
        events located in the country itself.
      </p>
    </section>
  );
}

function cellTitle(c: MatrixCell) {
  const head =
    c.kind === "domestic"
      ? `${c.origin_country_name} → ${c.destination_country_name}\nObserved domestic response`
      : `${c.origin_country_name} → ${c.destination_country_name}`;
  const ok = c.support.status === "ok";
  return [
    head,
    `Average response   ${ok ? hours(c.mean_response_hours) : "insufficient support"}`,
    `Median response    ${ok ? hours(c.median_response_hours) : "insufficient support"}`,
    `Coverage rate      ${pct(c.coverage_rate)}`,
    `Covered / eligible ${c.covered_events} / ${c.eligible_events}`,
    `Median 95% CI      ${ok ? ci(c.latency_ci_low, c.latency_ci_high, " h") : "—"}`,
    `Coverage 95% CI    ${ciPct(c.coverage_ci_low, c.coverage_ci_high)}`,
  ].join("\n");
}

/* ------------------------------------------------------------------- events */

function EventsTable({
  title,
  events,
  utc,
  onOpenEvent,
  compact = false,
}: {
  title: string;
  events: ResponseEvent[];
  utc: (iso: string) => string;
  onOpenEvent: Props["onOpenEvent"];
  compact?: boolean;
}) {
  if (events.length === 0) return <p className="muted small">No eligible events under the current filters.</p>;
  return (
    <>
      <h3>{title}</h3>
      <table className={`countries analytics events ${compact ? "compact" : ""}`}>
        <thead>
          <tr>
            <th>event</th>
            <th>origin onset</th>
            <th>dest. onset</th>
            <th>response</th>
            <th>eff.</th>
            <th>ratio</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr key={`${e.level}-${e.event_id}`} className={e.covered ? "seen" : "pending"}>
              <td>
                <button
                  className="link event"
                  title="open this story on the globe"
                  onClick={() => onOpenEvent(e.family_id, e.level === "incident" ? e.event_id : null)}
                >
                  {e.title ?? `#${e.event_id}`}
                </button>
                <span className="result-meta">
                  {utc(e.event_start)}
                  {e.event_types.length > 0 && ` · ${e.event_types.slice(0, 2).join(", ").replaceAll("_", " ")}`}
                  {` · ${e.event_effective_reports} reports`}
                  {e.response_reference === "world_fallback" && " · world fallback"}
                </span>
              </td>
              <td className="time">{e.response_reference === "event_start" ? "event start" : e.origin_onset ? utc(e.origin_onset) : e.world_onset ? `world ${utc(e.world_onset)}` : "—"}</td>
              <td className="time">{e.destination_onset ? utc(e.destination_onset) : e.has_documents ? "no onset" : "no coverage"}</td>
              <td>{e.covered ? hours(e.response_hours) : e.censor_hours != null ? `>${e.censor_hours.toFixed(0)} h*` : "—"}</td>
              <td>{e.effective_reports ?? "—"}</td>
              <td>{e.attention_ratio == null ? "—" : e.attention_ratio.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Covered events first (fastest response first), then uncovered ones. * right-censored: no onset before
        the window ended. Click an event to open it on the globe.
      </p>
    </>
  );
}

/* -------------------------------------------------------------------- scene */

function buildScene(
  mode: StatsMode,
  ctx: {
    overview: CountryOverview | null;
    pair: PairResponse | null;
    matrix: MatrixResponse | null;
    compare: string[];
    name: (code: string) => string;
  },
): StatsScene | null {
  if (mode === "overview" && ctx.overview) {
    const o = ctx.overview;
    return {
      focus: { code: o.publisher_country, label: `${o.publisher_country_name} · foreign response`, lines: summaryLines(o.foreign) },
      related: o.by_origin
        .filter((r) => r.support.status === "ok" && r.origin_country !== o.publisher_country)
        .slice(0, GLOBE_RELATED)
        .map((r) => ({
          code: r.origin_country,
          label: `events in ${r.origin_country_name ?? r.origin_country} → ${o.publisher_country_name}`,
          lines: summaryLines(r),
        })),
      legend: {
        focus: "selected publisher country",
        related: "origin countries with supported response estimates",
      },
    };
  }
  if (mode === "pair" && ctx.pair) {
    const f = ctx.pair.forward;
    if (f.kind === "domestic") {
      return {
        focus: { code: f.origin_country, label: `${f.origin_country_name} · observed domestic response`, lines: summaryLines(f.summary) },
        related: [],
        legend: { focus: "events and publishers in the same country", related: "" },
      };
    }
    return {
      focus: { code: f.origin_country, label: `events located in ${f.origin_country_name}`, lines: summaryLines(f.summary) },
      related: [
        {
          code: f.destination_country,
          label: `${f.origin_country_name} → ${f.destination_country_name}`,
          lines: summaryLines(f.summary),
        },
      ],
      legend: { focus: "where the events happened (origin)", related: "publisher country (destination)" },
    };
  }
  if (mode === "compare" && ctx.matrix) {
    const rows = new Map(ctx.matrix.destination_rows.map((r) => [r.publisher_country, r]));
    return {
      focus: null,
      related: ctx.compare.map((code) => {
        const r = rows.get(code);
        return {
          code,
          label: `${ctx.name(code)} · response to foreign events`,
          lines: r ? summaryLines(r.foreign) : [],
        };
      }),
      legend: { focus: null, related: "selected countries (as publishers)" },
    };
  }
  return null;
}
