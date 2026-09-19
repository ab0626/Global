import { useCallback, useEffect, useMemo, useState } from "react";
import "leaflet/dist/leaflet.css";
import "./App.css";
import {
  fetchArticles,
  fetchEventDetail,
  fetchEvents,
  fetchFacets,
  fetchGeo,
  fetchMeta,
  fetchTimeline,
  fetchToneChart,
  formatStamp,
  parseStamp,
  toStamp,
} from "./api";
import type {
  Article,
  Facets,
  GdeltEvent,
  GeoFeature,
  Meta,
  TimelineSeries,
  ToneBin,
  Window as TimeWindow,
} from "./api";
import { CoverageMap } from "./components/CoverageMap";
import {
  ArticleTable,
  EventTable,
  FacetList,
  Panel,
  ToneHistogram,
  ToneTimeline,
  VolumeTimeline,
} from "./components/Panels";

const EXAMPLES = [
  { label: "Christchurch attack", query: "christchurch OR mosque" },
  { label: "737 MAX grounding", query: "boeing OR 737 -flight" },
  { label: "Cyclone Idai", query: "idai OR mozambique" },
  { label: "Brexit", query: "brexit" },
  { label: "Protests (CAMEO)", query: "theme:protest" },
  { label: "Non-English coverage", query: "-sourcelang:english" },
];

type Results = {
  articles: Article[];
  volume: TimelineSeries[];
  raw: TimelineSeries[];
  tone: TimelineSeries[];
  tonechart: ToneBin[];
  geo: GeoFeature[];
  events: GdeltEvent[];
  facets: Facets | null;
};

const EMPTY: Results = {
  articles: [],
  volume: [],
  raw: [],
  tone: [],
  tonechart: [],
  geo: [],
  events: [],
  facets: null,
};

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [draft, setDraft] = useState("christchurch OR mosque");
  const [query, setQuery] = useState("christchurch OR mosque");
  const [span, setSpan] = useState<TimeWindow | null>(null);
  const [fullSpan, setFullSpan] = useState<TimeWindow | null>(null);
  const [sort, setSort] = useState("hybridrel");
  const [quad, setQuad] = useState<number | undefined>(undefined);
  const [results, setResults] = useState<Results>(EMPTY);
  const [selected, setSelected] = useState<{ event: GdeltEvent; articles: Article[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMeta()
      .then((value) => {
        setMeta(value);
        setSpan(value.coverage);
        setFullSpan(value.coverage);
      })
      .catch((problem: Error) => setError(problem.message));
  }, []);

  useEffect(() => {
    if (!span) return;
    let live = true;
    setLoading(true);
    setError(null);
    Promise.all([
      fetchArticles(query, span, sort),
      fetchTimeline(query, span, "timelinevol"),
      fetchTimeline(query, span, "timelinevolraw"),
      fetchTimeline(query, span, "timelinetone"),
      fetchToneChart(query, span),
      fetchGeo(query, span),
      fetchEvents(query, span, quad),
      fetchFacets(query, span),
    ])
      .then(([articles, volume, raw, tone, tonechart, geo, events, facets]) => {
        if (!live) return;
        setResults({
          articles: articles.articles,
          volume: volume.timeline,
          raw: raw.timeline.filter((series) => series.series === "Article Count"),
          tone: tone.timeline,
          tonechart: tonechart.tonechart,
          geo: geo.features,
          events: events.events,
          facets,
        });
      })
      .catch((problem: Error) => live && setError(problem.message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [query, span, sort, quad]);

  const addOperator = useCallback((operator: string, value: string) => {
    const token = value.includes(" ") ? `${operator}:"${value}"` : `${operator}:${value}`;
    setDraft((current) => (current.includes(token) ? current : `${current} ${token}`.trim()));
    setQuery((current) => (current.includes(token) ? current : `${current} ${token}`.trim()));
  }, []);

  const openEvent = useCallback((id: number) => {
    fetchEventDetail(id)
      .then(setSelected)
      .catch((problem: Error) => setError(problem.message));
  }, []);

  const matched = results.facets?.matched_articles ?? 0;
  const share = useMemo(() => {
    const volume = results.volume[0]?.data ?? [];
    if (!volume.length) return 0;
    return volume.reduce((total, point) => total + point.value, 0) / volume.length;
  }, [results.volume]);
  const peak = useMemo(() => {
    const counts = results.raw[0]?.data ?? [];
    return counts.reduce(
      (best, point) => (point.value > best.value ? point : best),
      { date: "", value: 0 },
    );
  }, [results.raw]);
  const meanTone = useMemo(() => {
    const values = results.articles.map((article) => article.tone ?? 0);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  }, [results.articles]);

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>GDELT Explorer</h1>
          <p>
            {meta
              ? `${meta.articles.toLocaleString()} articles · ${meta.events.toLocaleString()} CAMEO events · ${formatStamp(
                  meta.coverage.start,
                )} → ${formatStamp(meta.coverage.end)}`
              : "loading dataset metadata…"}
          </p>
        </div>
        <form
          className="search"
          onSubmit={(submit) => {
            submit.preventDefault();
            setQuery(draft.trim());
          }}
        >
          <input
            value={draft}
            spellCheck={false}
            onChange={(change) => setDraft(change.target.value)}
            placeholder='e.g. mosque OR christchurch -domain:reddit.com theme:fight'
          />
          <select value={sort} onChange={(change) => setSort(change.target.value)}>
            <option value="hybridrel">most events</option>
            <option value="datedesc">newest</option>
            <option value="dateasc">oldest</option>
            <option value="toneasc">most negative</option>
            <option value="tonedesc">most positive</option>
          </select>
          <button type="submit">Search</button>
        </form>
      </header>

      <nav className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example.label}
            className={query === example.query ? "chip active" : "chip"}
            onClick={() => {
              setDraft(example.query);
              setQuery(example.query);
            }}
          >
            {example.label}
          </button>
        ))}
        <span className="spacer" />
        {[1, 2, 3, 4].map((value) => (
          <button
            key={value}
            className={quad === value ? "chip active" : "chip"}
            onClick={() => setQuad(quad === value ? undefined : value)}
          >
            {["verbal coop", "material coop", "verbal conflict", "material conflict"][value - 1]}
          </button>
        ))}
        {span && fullSpan && span.start !== fullSpan.start && (
          <button className="chip reset" onClick={() => setSpan(fullSpan)}>
            reset window ({formatStamp(span.start)} → {formatStamp(span.end)})
          </button>
        )}
      </nav>

      {error && <div className="error">{error}</div>}

      <div className="kpis">
        <Kpi label="matching articles" value={matched.toLocaleString()} />
        <Kpi label="mean share of coverage" value={`${share.toFixed(2)}%`} />
        <Kpi
          label="peak 15-minute volume"
          value={peak.date ? `${peak.value} @ ${formatStamp(peak.date).slice(5)}` : "—"}
        />
        <Kpi label="mean tone (shown articles)" value={meanTone.toFixed(2)} />
        <Kpi label="events in window" value={results.events.length.toLocaleString()} />
        {loading && <span className="loading">querying…</span>}
      </div>

      <main>
        <Panel
          title="Coverage volume"
          subtitle="share of all monitored articles (left) and raw article count (right); click to zoom the window"
          className="wide"
        >
          <VolumeTimeline
            volume={results.volume}
            raw={results.raw}
            onBrush={(from, to) => setSpan({ start: from, end: to })}
          />
        </Panel>

        <Panel title="Where the events happened" subtitle="ActionGeo of every linked CAMEO event" className="wide tall">
          {results.geo.length ? (
            <CoverageMap features={results.geo} onPick={(place) => addOperator("location", place)} />
          ) : (
            <p className="empty">no geolocated events for this query</p>
          )}
        </Panel>

        <Panel title="Average document tone" subtitle="MentionDocTone per bucket">
          <ToneTimeline timeline={results.tone} />
        </Panel>

        <Panel title="Tone distribution" subtitle="DOC 2.0 tonechart bins; click a bar to inspect">
          <ToneHistogram
            bins={results.tonechart}
            onSelect={(bin) =>
              setResults((current) => ({ ...current, articles: bin.toparts }))
            }
          />
        </Panel>

        <Panel title="Narrow the query" subtitle="click any value to add its operator" className="facet-panel">
          {results.facets && <FacetList facets={results.facets} onPick={addOperator} />}
        </Panel>

        <Panel title="Top CAMEO events" subtitle="ranked by article count; click a row for its coverage" className="wide">
          <EventTable events={results.events} onSelect={(event) => openEvent(event.globaleventid)} />
        </Panel>

        <Panel title="Articles" subtitle="GDELT DOC artlist shape, served from Mentions" className="wide">
          <ArticleTable articles={results.articles} onEvent={openEvent} />
        </Panel>
      </main>

      {selected && (
        <aside className="drawer">
          <header>
            <h2>
              {selected.event.rootlabel} <span className="muted">({selected.event.eventcode})</span>
            </h2>
            <button onClick={() => setSelected(null)}>close</button>
          </header>
          <dl>
            <div>
              <dt>event id</dt>
              <dd>{selected.event.globaleventid}</dd>
            </div>
            <div>
              <dt>actors</dt>
              <dd>
                {selected.event.actor1 || "—"}
                {selected.event.actor2 ? ` → ${selected.event.actor2}` : ""}
              </dd>
            </div>
            <div>
              <dt>location</dt>
              <dd>{selected.event.location || "—"}</dd>
            </div>
            <div>
              <dt>quad class</dt>
              <dd>{selected.event.quadlabel}</dd>
            </div>
            <div>
              <dt>goldstein / tone</dt>
              <dd>
                {selected.event.goldstein?.toFixed(1) ?? "—"} / {selected.event.avgtone?.toFixed(1) ?? "—"}
              </dd>
            </div>
            <div>
              <dt>mentions / articles</dt>
              <dd>
                {selected.event.nummentions} / {selected.event.numarticles}
              </dd>
            </div>
          </dl>
          <h3>Coverage in this slice</h3>
          <ol>
            {selected.articles.map((article) => (
              <li key={article.url}>
                <a href={article.url} target="_blank" rel="noreferrer">
                  {article.title}
                </a>
                <div className="muted">
                  {article.domain} · {formatStamp(article.seendate)} · tone {article.tone?.toFixed(1)}
                </div>
              </li>
            ))}
          </ol>
        </aside>
      )}

      <footer>
        {meta && (
          <details>
            <summary>
              What is real and what is derived (built {meta.built_at} from {meta.mention_files} Mentions
              files)
            </summary>
            <ul>
              {Object.entries(meta.field_provenance).map(([field, note]) => (
                <li key={field}>
                  <code>{field}</code> — {note}
                </li>
              ))}
            </ul>
            <p>
              Query operators: {meta.query_operators.join(", ")}. Window is inclusive of{" "}
              {formatStamp(meta.coverage.start)} to {formatStamp(meta.coverage.end)} (
              {Math.round(
                (parseStamp(meta.coverage.end).getTime() - parseStamp(meta.coverage.start).getTime()) /
                  3600000,
              )}{" "}
              hours, stamp {toStamp(new Date())} now).
            </p>
          </details>
        )}
      </footer>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi">
      <span className="kpi-value">{value}</span>
      <span className="kpi-label">{label}</span>
    </div>
  );
}
