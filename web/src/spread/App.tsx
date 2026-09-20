import { useEffect, useMemo, useRef, useState } from "react";
import { Globe, type Marker } from "./Globe";
import {
  countries,
  eventCountries,
  familyCountries,
  familyDetail,
  fullSpread,
  search,
  type CountryAttention,
  type CountryBaseline,
  type Family,
  type MacroEvent,
  type Spread,
  type SpreadDocument,
} from "./api";
import "./spread.css";

const EXAMPLES = ["Turkey Syria earthquake", "Chinese balloon", "Grammy", "Erdbeben", "地震"];
const MIN_CONFIDENCE = 0.5;
/** Animation compresses the observed window into this many seconds. */
const PLAY_SECONDS = 40;
/** A story family (what a user means by "the event") or one of its incidents
 * (a single Leiden cluster). */
type Selection =
  | { kind: "family"; family: Family }
  | { kind: "incident"; event: MacroEvent; family: Family };

type Status =
  | { kind: "idle" }
  | { kind: "loading"; what: string }
  | { kind: "error"; message: string }
  | { kind: "ready" };

type Loaded = {
  selection: Selection;
  /** incidents of the family, largest first; the first one supplies the event location */
  incidents: MacroEvent[];
  spread: Spread;
  attention: CountryAttention[];
  worldOnset: string | null;
};

const fmt = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  month: "short",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});
const utc = (iso: string) => `${fmt.format(new Date(iso))} UTC`;

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Family[] | null>(null);
  const [expanded, setExpanded] = useState<Map<number, MacroEvent[]>>(new Map());
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [baseline, setBaseline] = useState<Map<string, CountryBaseline>>(new Map());
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [hover, setHover] = useState<Marker | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    countries()
      .then((r) =>
        setBaseline(new Map(r.publisher_countries.map((c) => [c.publisher_country, c]))),
      )
      .catch((e: Error) => setStatus({ kind: "error", message: `country table: ${e.message}` }));
  }, []);

  async function runSearch(q: string) {
    const text = q.trim();
    if (!text) return;
    setQuery(text);
    setStatus({ kind: "loading", what: "searching" });
    setResults(null);
    setExpanded(new Map());
    try {
      const r = await search(text);
      setResults(r.families);
      setStatus({ kind: "ready" });
      if (r.families.length > 0) void select({ kind: "family", family: r.families[0] });
    } catch (e) {
      setStatus({ kind: "error", message: (e as Error).message });
    }
  }

  async function incidentsOf(familyId: number): Promise<MacroEvent[]> {
    const known = expanded.get(familyId);
    if (known) return known;
    const detail = await familyDetail(familyId);
    setExpanded((m) => new Map(m).set(familyId, detail.events));
    return detail.events;
  }

  async function select(selection: Selection) {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setPlaying(false);
    setPlayhead(0);
    setLoaded(null);
    const familyId = selection.family.family_id;
    const target =
      selection.kind === "family"
        ? { kind: "family" as const, id: familyId }
        : { kind: "incident" as const, id: selection.event.macro_event_id };
    setStatus({
      kind: "loading",
      what: `loading spread of ${target.kind === "family" ? "story" : "incident"} #${target.id}`,
    });
    try {
      const [spread, att, incidents] = await Promise.all([
        fullSpread(target, MIN_CONFIDENCE, controller.signal),
        target.kind === "family" ? familyCountries(familyId) : eventCountries(target.id),
        incidentsOf(familyId),
      ]);
      if (controller.signal.aborted) return;
      setLoaded({
        selection,
        incidents,
        spread,
        attention: att.publisher_countries,
        worldOnset: att.world_onset,
      });
      setStatus({ kind: "ready" });
      setPlaying(true);
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setStatus({ kind: "error", message: (e as Error).message });
    }
  }

  const timeline = useMemo(() => buildTimeline(loaded, baseline), [loaded, baseline]);

  useEffect(() => {
    if (!playing || !timeline) return;
    let frame = 0;
    let last = performance.now();
    const tick = (ts: number) => {
      const dt = ((ts - last) / 1000) * speed;
      last = ts;
      setPlayhead((p) => {
        const next = p + dt;
        if (next >= timeline.duration + 1.5) {
          setPlaying(false);
          return timeline.duration + 1.5;
        }
        return next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, speed, timeline]);

  const shown = timeline ? timeline.docs.filter((d) => d.t <= playhead) : [];
  const clock =
    timeline && timeline.docs.length > 0
      ? new Date(timeline.start + (Math.min(playhead, timeline.duration) / timeline.duration) * timeline.span)
      : null;
  const visibleCountries = new Set(shown.map((d) => d.doc.publisher_country));

  return (
    <div className="spread-app">
      <header className="spread-header">
        <h1>
          Show the spread of{" "}
          <form
            className="inline-form"
            onSubmit={(e) => {
              e.preventDefault();
              void runSearch(query);
            }}
          >
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="an event, e.g. Turkey earthquake"
              aria-label="event query"
              autoFocus
            />
            <button type="submit">Go</button>
          </form>
        </h1>
        <p className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="link" onClick={() => void runSearch(ex)}>
              {ex}
            </button>
          ))}
        </p>
      </header>

      <main className="spread-main">
        <aside className="spread-side">
          {status.kind === "error" && <div className="notice error">{status.message}</div>}
          {status.kind === "loading" && <div className="notice">{status.what}…</div>}
          {results && results.length === 0 && (
            <div className="notice">
              No story matches “{query}” in this window
              {loadedMeta(loaded)}. Try a different wording or language.
            </div>
          )}
          {results && results.length > 0 && (
            <section>
              <h2>Matching stories</h2>
              <ol className="results">
                {results.map((fam) => {
                  const active =
                    loaded?.selection.kind === "family" &&
                    loaded.selection.family.family_id === fam.family_id;
                  const incidents = expanded.get(fam.family_id);
                  return (
                    <li key={fam.family_id} className={active ? "active" : ""}>
                      <button
                        className="result"
                        onClick={() => void select({ kind: "family", family: fam })}
                      >
                        <span className="result-kind">story family</span>
                        <span className="result-title">{fam.title ?? fam.label ?? "(untitled)"}</span>
                        <span className="result-meta">
                          {fam.publisher_country_count} countries · {fam.raw_documents.toLocaleString()}{" "}
                          articles · {fam.incident_count} incident{fam.incident_count === 1 ? "" : "s"} ·{" "}
                          {fam.language_count} languages
                        </span>
                      </button>
                      {fam.incident_count > 1 && !incidents && (
                        <button
                          className="link small"
                          onClick={() => void incidentsOf(fam.family_id).catch(() => undefined)}
                        >
                          Explore incidents →
                        </button>
                      )}
                      {incidents && (
                        <ol className="incidents">
                          {incidents.map((ev) => {
                            const on =
                              loaded?.selection.kind === "incident" &&
                              loaded.selection.event.macro_event_id === ev.macro_event_id;
                            return (
                              <li key={ev.macro_event_id} className={on ? "active" : ""}>
                                <button
                                  className="result"
                                  onClick={() =>
                                    void select({ kind: "incident", event: ev, family: fam })
                                  }
                                >
                                  <span className="result-title">
                                    {ev.title ?? ev.label ?? "(untitled)"}
                                  </span>
                                  <span className="result-meta">
                                    {ev.publisher_country_count} countries · {ev.raw_documents}{" "}
                                    articles
                                  </span>
                                </button>
                              </li>
                            );
                          })}
                        </ol>
                      )}
                    </li>
                  );
                })}
              </ol>
            </section>
          )}

          {loaded && timeline && (
            <section className="detail">
              {loaded.selection.kind === "family" ? (
                <>
                  <h2>{loaded.selection.family.title ?? loaded.selection.family.label}</h2>
                  <p className="muted">
                    Story family #{loaded.selection.family.family_id} ·{" "}
                    {utc(loaded.selection.family.start_time)} → {utc(loaded.selection.family.end_time)}
                    {loaded.incidents[0]?.event_country && (
                      <> · event in {name(loaded.incidents[0].event_country, baseline)}</>
                    )}
                  </p>
                </>
              ) : (
                <>
                  <h2>{loaded.selection.event.title ?? loaded.selection.event.label}</h2>
                  <p className="muted">
                    Incident #{loaded.selection.event.macro_event_id} ·{" "}
                    {utc(loaded.selection.event.start_time)} → {utc(loaded.selection.event.end_time)}
                    {loaded.selection.event.event_country && (
                      <> · event in {name(loaded.selection.event.event_country, baseline)}</>
                    )}
                  </p>
                  <p className="scope">
                    <button
                      className="link"
                      onClick={() => {
                        if (loaded.selection.kind === "incident")
                          void select({ kind: "family", family: loaded.selection.family });
                      }}
                    >
                      ← whole story (family #{loaded.selection.family.family_id})
                    </button>
                  </p>
                </>
              )}
              <dl className="stats">
                <div>
                  <dt>articles</dt>
                  <dd>{loaded.spread.total}</dd>
                </div>
                <div>
                  <dt>outlets</dt>
                  <dd>{new Set(loaded.spread.documents.map((d) => d.source_domain)).size}</dd>
                </div>
                <div>
                  <dt>incidents</dt>
                  <dd>{new Set(loaded.spread.documents.map((d) => d.macro_event_id)).size}</dd>
                </div>
                <div>
                  <dt>publisher countries</dt>
                  <dd>{loaded.spread.countries.length}</dd>
                </div>
              </dl>
              {timeline.unplaced.length > 0 && (
                <p className="muted small">
                  Not drawn (no centroid): {timeline.unplaced.join(", ")}
                </p>
              )}
              {loaded.spread.excluded_documents > 0 && (
                <p className="muted small">
                  {loaded.spread.excluded_documents} articles hidden: publisher country
                  unresolved or below confidence {MIN_CONFIDENCE}.
                </p>
              )}

              <h3>Publisher countries by first observation</h3>
              <table className="countries">
                <thead>
                  <tr>
                    <th>country</th>
                    <th>first seen</th>
                    <th>lag</th>
                    <th>articles</th>
                    <th>eff.</th>
                    <th>ratio</th>
                  </tr>
                </thead>
                <tbody>
                  {loaded.spread.countries.map((c) => {
                    const a = loaded.attention.find((x) => x.publisher_country === c.publisher_country);
                    const seen = visibleCountries.has(c.publisher_country);
                    return (
                      <tr key={c.publisher_country} className={seen ? "seen" : "pending"}>
                        <td>{name(c.publisher_country, baseline)}</td>
                        <td>{utc(c.first_seen)}</td>
                        <td>{a?.lag_hours == null ? "—" : `${a.lag_hours >= 0 ? "+" : ""}${a.lag_hours.toFixed(1)} h`}</td>
                        <td>{c.raw_documents}</td>
                        <td>{c.effective_reports}</td>
                        <td>{a?.attention_ratio == null ? "—" : a.attention_ratio.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="muted small">
                Times are when GDELT first observed each URL (15-minute batches), not publication
                times. Lag is country onset (3rd outlet or 10th percentile) minus world onset
                {loaded.worldOnset ? ` (${utc(loaded.worldOnset)})` : ""}; “—” = fewer than 3
                outlets. Ratio = country's share of its own output vs world share, computed over the{" "}
                {loaded.selection.kind === "family" ? "whole story" : "incident"}.
              </p>
            </section>
          )}
        </aside>

        <section className="spread-stage">
          <div className="globe-wrap">
            <Globe
              markers={timeline?.markers ?? []}
              now={playhead}
              focus={timeline?.focus ?? null}
              onHover={setHover}
            />
            {hover && <div className="tooltip">{hover.label}</div>}
            {!loaded && status.kind !== "loading" && (
              <div className="stage-hint">Type an event above to animate where its coverage appeared.</div>
            )}
          </div>
          {timeline && (
            <div className="controls">
              <button
                onClick={() => {
                  if (!playing && playhead >= timeline.duration) setPlayhead(0);
                  setPlaying((p) => !p);
                }}
                disabled={timeline.docs.length === 0}
              >
                {playing ? "Pause" : playhead >= timeline.duration ? "Replay" : "Play"}
              </button>
              <input
                type="range"
                min={0}
                max={timeline.duration + 1.5}
                step={0.05}
                value={playhead}
                onChange={(e) => {
                  setPlaying(false);
                  setPlayhead(Number(e.target.value));
                }}
                aria-label="playhead"
              />
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="speed">
                {[0.5, 1, 2, 4].map((s) => (
                  <option key={s} value={s}>
                    {s}×
                  </option>
                ))}
              </select>
              <span className="clock">{clock ? utc(clock.toISOString()) : "—"}</span>
              <span className="muted">
                {shown.length}/{timeline.docs.length} articles · {visibleCountries.size} countries
              </span>
            </div>
          )}
          {loaded && timeline && timeline.docs.length === 0 && (
            <div className="notice">
              No articles with a resolved publisher country to animate.
            </div>
          )}
        </section>
      </main>
      <footer className="spread-footer">
        {loaded?.spread.meta && (
          <>
            window {utc(loaded.spread.meta.window_start)} → {utc(loaded.spread.meta.window_end)} ·{" "}
            {loaded.spread.meta.documents_total.toLocaleString()} documents ·{" "}
            {loaded.spread.meta.macro_events.toLocaleString()} incidents · model{" "}
            {loaded.spread.meta.resolution_model} · publisher country = outlet location, not event
            location
          </>
        )}
      </footer>
    </div>
  );
}

function loadedMeta(loaded: Loaded | null) {
  const m = loaded?.spread.meta;
  return m ? ` (${utc(m.window_start)} → ${utc(m.window_end)})` : "";
}

function name(code: string, baseline: Map<string, CountryBaseline>) {
  return baseline.get(code)?.country_name ?? code;
}

type Timeline = {
  docs: { doc: SpreadDocument; t: number }[];
  markers: Marker[];
  unplaced: string[];
  duration: number;
  start: number;
  span: number;
  focus: [number, number] | null;
};

/** Map observed_time onto [0, PLAY_SECONDS]; one red dot per article, jittered around
 * its publisher-country centroid. */
function buildTimeline(loaded: Loaded | null, baseline: Map<string, CountryBaseline>): Timeline | null {
  if (!loaded) return null;
  const docs = [...loaded.spread.documents].sort((a, b) =>
    a.observed_time === b.observed_time
      ? a.document_id - b.document_id
      : a.observed_time < b.observed_time
        ? -1
        : 1,
  );
  if (docs.length === 0) {
    return { docs: [], markers: [], unplaced: [], duration: 0, start: 0, span: 1, focus: null };
  }
  const start = Date.parse(docs[0].observed_time);
  const end = Date.parse(docs[docs.length - 1].observed_time);
  const span = Math.max(end - start, 15 * 60 * 1000);
  const duration = PLAY_SECONDS;
  const timed = docs.map((doc) => ({
    doc,
    t: ((Date.parse(doc.observed_time) - start) / span) * duration,
  }));
  const markers: Marker[] = [];
  const unplaced = new Set<string>();
  const perCountry = new Map<string, number>();
  for (const { doc, t } of timed) {
    const code = doc.publisher_country ?? "?";
    const row = baseline.get(code);
    if (!row || row.lat == null || row.lon == null) {
      unplaced.add(code);
      continue;
    }
    const n = perCountry.get(code) ?? 0;
    perCountry.set(code, n + 1);
    // deterministic sunflower jitter so stacked articles in one country stay legible
    const angle = n * 2.399963;
    const radius = Math.min(4, 0.9 * Math.sqrt(n));
    markers.push({
      key: String(doc.document_id),
      lat: row.lat + radius * Math.sin(angle),
      lon: row.lon + radius * Math.cos(angle),
      t,
      size: 3.2,
      label: `${doc.source_domain} · ${utc(doc.observed_time)}${doc.title ? ` — ${doc.title}` : ""}`,
      kind: "publisher",
    });
  }
  const ev = loaded.selection.kind === "incident" ? loaded.selection.event : loaded.incidents[0];
  let focus: [number, number] | null = null;
  if (ev && ev.lat != null && ev.lon != null) {
    focus = [ev.lat, ev.lon];
    markers.unshift({
      key: "event",
      lat: ev.lat,
      lon: ev.lon,
      t: 0,
      size: 6,
      label: `event location (GDELT geo of the incident)`,
      kind: "event",
    });
  } else if (markers.length > 0) {
    focus = [markers[0].lat, markers[0].lon];
  }
  return { docs: timed, markers, unplaced: [...unplaced], duration, start, span, focus };
}
