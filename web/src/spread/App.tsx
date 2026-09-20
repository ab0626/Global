import { useEffect, useMemo, useRef, useState } from "react";
import { RippleGlobe } from "./globe/RippleGlobe";
import type { GlobeConfig } from "./globe/config";
import type { CountryMarker, Hover, Origin } from "./globe/types";
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
  const [hover, setHover] = useState<Hover | null>(null);
  const [background, setBackground] = useState<GlobeConfig["background"]>("white");
  const [arcs, setArcs] = useState(true);
  const globeConfig = useMemo<Partial<GlobeConfig>>(() => ({ background, arcs }), [background, arcs]);
  const abort = useRef<AbortController | null>(null);
  const side = useRef<HTMLElement | null>(null);

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
    abort.current?.abort();
    setPlaying(false);
    setPlayhead(0);
    setLoaded(null);
    side.current?.scrollTo({ top: 0 });
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

  const shownCount = timeline ? countAtOrBefore(timeline.docTimes, playhead) : 0;
  const clock =
    timeline && timeline.docTimes.length > 0
      ? new Date(timeline.start + (Math.min(playhead, timeline.duration) / timeline.duration) * timeline.span)
      : null;
  const visibleCountries = useMemo(
    () => new Set(timeline?.markers.filter((m) => m.t <= playhead).map((m) => m.code) ?? []),
    [timeline, playhead],
  );

  return (
    <div className={`spread-app ${background}`}>
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
        <aside className="spread-side" ref={side}>
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
              {eventTypes(loaded).length > 0 && (
                <p className="types">
                  {eventTypes(loaded).map((t) => (
                    <span key={t} className="chip">
                      {t.replaceAll("_", " ")}
                    </span>
                  ))}
                </p>
              )}
              <dl className="stats">
                <div>
                  <dt>articles</dt>
                  <dd>{loaded.spread.total.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>outlets</dt>
                  <dd>{new Set(loaded.spread.documents.map((d) => d.source_domain)).size.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>incidents</dt>
                  <dd>{new Set(loaded.spread.documents.map((d) => d.macro_event_id)).size}</dd>
                </div>
                <div>
                  <dt>publisher countries</dt>
                  <dd>{loaded.spread.countries.length}</dd>
                </div>
                <div>
                  <dt>first observed</dt>
                  <dd className="time">{timeline.docTimes.length ? utc(new Date(timeline.start).toISOString()) : "—"}</dd>
                </div>
                <div>
                  <dt>global onset</dt>
                  <dd className="time">{loaded.worldOnset ? utc(loaded.worldOnset) : "—"}</dd>
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
            <RippleGlobe
              markers={timeline?.markers ?? []}
              origin={timeline?.origin ?? null}
              now={playhead}
              time={clock?.getTime() ?? null}
              onHover={setHover}
              config={globeConfig}
            />
            {hover && <Tooltip hover={hover} />}
            {!loaded && status.kind !== "loading" && (
              <div className="stage-hint">Type an event above to animate where its coverage appeared.</div>
            )}
            <div className="stage-options">
              <button className="link" onClick={() => setBackground((b) => (b === "white" ? "dark" : "white"))}>
                {background === "white" ? "dark mode" : "light mode"}
              </button>
              <button className="link" onClick={() => setArcs((a) => !a)}>
                {arcs ? "hide arcs" : "show arcs"}
              </button>
            </div>
            <div className="legend">
              <span><i className="swatch origin" /> event location</span>
              <span><i className="swatch active" /> publisher country (outlet base)</span>
              {arcs && <span className="muted">arcs show attention order, not transmission</span>}
              {clock && <span className="muted">daylight follows the clock (UTC)</span>}
            </div>
          </div>
          {timeline && (
            <div className="controls">
              <button
                onClick={() => {
                  if (!playing && playhead >= timeline.duration) setPlayhead(0);
                  setPlaying((p) => !p);
                }}
                disabled={timeline.docTimes.length === 0}
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
                {shownCount.toLocaleString()}/{timeline.docTimes.length.toLocaleString()} articles ·{" "}
                {visibleCountries.size}/{timeline.markers.length} countries
              </span>
            </div>
          )}
          {loaded && timeline && timeline.docTimes.length === 0 && (
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

function eventTypes(loaded: Loaded): string[] {
  const events = loaded.selection.kind === "incident" ? [loaded.selection.event] : loaded.incidents;
  const counts = new Map<string, number>();
  for (const ev of events) for (const t of ev.event_types) counts.set(t, (counts.get(t) ?? 0) + 1);
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([t]) => t);
}

/** Number of ascending `times` that are <= `value`. */
function countAtOrBefore(times: number[], value: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function Tooltip({ hover }: { hover: Hover }) {
  const style = { left: hover.x + 14, top: hover.y + 14 };
  if (hover.kind === "origin") {
    const o = hover.origin;
    return (
      <div className="tooltip" style={style}>
        <strong>{o.title}</strong>
        <span className="tooltip-kind origin">event location</span>
        <div>{o.location}</div>
        <div>global onset {o.globalOnset ? utc(o.globalOnset) : "—"}</div>
      </div>
    );
  }
  const m = hover.marker;
  return (
    <div className="tooltip" style={style}>
      <strong>{m.name}</strong>
      <span className="tooltip-kind">publisher country</span>
      <dl>
        <dt>first seen</dt>
        <dd>{utc(m.firstSeen)}</dd>
        <dt>onset</dt>
        <dd>{m.onset ? utc(m.onset) : "— (fewer than 3 outlets)"}</dd>
        <dt>lag vs world</dt>
        <dd>{m.lagHours == null ? "—" : `${m.lagHours >= 0 ? "+" : ""}${m.lagHours.toFixed(1)} h`}</dd>
        <dt>articles</dt>
        <dd>{m.articles.toLocaleString()}</dd>
        <dt>effective reports</dt>
        <dd>{m.effectiveReports.toLocaleString()}</dd>
        <dt>attention ratio</dt>
        <dd>{m.attentionRatio == null ? "—" : m.attentionRatio.toFixed(2)}</dd>
      </dl>
    </div>
  );
}

type Timeline = {
  /** playhead second of every drawn article, ascending */
  docTimes: number[];
  markers: CountryMarker[];
  origin: Origin | null;
  unplaced: string[];
  duration: number;
  start: number;
  span: number;
};

/** Map observed_time onto [0, PLAY_SECONDS] and fold articles into one marker per
 * publisher country (activation = first article, growth = cumulative articles). */
function buildTimeline(loaded: Loaded | null, baseline: Map<string, CountryBaseline>): Timeline | null {
  if (!loaded) return null;
  const docs = [...loaded.spread.documents].sort((a, b) =>
    a.observed_time === b.observed_time
      ? a.document_id - b.document_id
      : a.observed_time < b.observed_time
        ? -1
        : 1,
  );
  const empty: Timeline = {
    docTimes: [],
    markers: [],
    origin: null,
    unplaced: [],
    duration: 0,
    start: 0,
    span: 1,
  };
  if (docs.length === 0) return { ...empty, origin: originOf(loaded, baseline) };
  const start = Date.parse(docs[0].observed_time);
  const end = Date.parse(docs[docs.length - 1].observed_time);
  const span = Math.max(end - start, 15 * 60 * 1000);
  const duration = PLAY_SECONDS;
  const at = (doc: SpreadDocument) => ((Date.parse(doc.observed_time) - start) / span) * duration;
  const perCountry = new Map<string, number[]>();
  const unplaced = new Set<string>();
  const docTimes: number[] = [];
  for (const doc of docs) {
    const code = doc.publisher_country ?? "?";
    const row = baseline.get(code);
    if (!row || row.lat == null || row.lon == null) {
      unplaced.add(code);
      continue;
    }
    const t = at(doc);
    docTimes.push(t);
    const times = perCountry.get(code);
    if (times) times.push(t);
    else perCountry.set(code, [t]);
  }
  const attention = new Map(loaded.attention.map((a) => [a.publisher_country, a]));
  const markers: CountryMarker[] = [];
  for (const c of loaded.spread.countries) {
    const times = perCountry.get(c.publisher_country);
    const row = baseline.get(c.publisher_country);
    if (!times || !row || row.lat == null || row.lon == null) continue;
    const a = attention.get(c.publisher_country);
    markers.push({
      code: c.publisher_country,
      name: row.country_name ?? c.publisher_country,
      lat: row.lat,
      lon: row.lon,
      t: times[0],
      articleTimes: times,
      firstSeen: c.first_seen,
      onset: a?.onset ?? null,
      lagHours: a?.lag_hours ?? null,
      articles: c.raw_documents,
      effectiveReports: c.effective_reports,
      attentionRatio: a?.attention_ratio ?? null,
    });
  }
  markers.sort((a, b) => a.t - b.t);
  return {
    docTimes,
    markers,
    origin: originOf(loaded, baseline),
    unplaced: [...unplaced],
    duration,
    start,
    span,
  };
}

/** Event location = GDELT geography of the selected incident (or the family's
 * largest incident) — distinct from where the covering outlets are based. */
function originOf(loaded: Loaded, baseline: Map<string, CountryBaseline>): Origin | null {
  const ev = loaded.selection.kind === "incident" ? loaded.selection.event : loaded.incidents[0];
  if (!ev || ev.lat == null || ev.lon == null) return null;
  const title =
    loaded.selection.kind === "family"
      ? (loaded.selection.family.title ?? loaded.selection.family.label ?? "story")
      : (ev.title ?? ev.label ?? "incident");
  return {
    lat: ev.lat,
    lon: ev.lon,
    title,
    location: ev.event_country
      ? `${name(ev.event_country, baseline)} · ${ev.lat.toFixed(2)}, ${ev.lon.toFixed(2)}`
      : `${ev.lat.toFixed(2)}, ${ev.lon.toFixed(2)}`,
    globalOnset: loaded.worldOnset,
  };
}
