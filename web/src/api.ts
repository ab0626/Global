const BASE = import.meta.env.VITE_API_BASE ?? "";

export type Article = {
  url: string;
  url_mobile: string;
  title: string;
  seendate: string;
  socialimage: string;
  domain: string;
  language: string;
  sourcecountry: string;
  tone: number | null;
  numevents: number;
  themes: string[];
  locations: string[];
  eventids: number[];
};

export type TimelineSeries = {
  series: string;
  data: { date: string; value: number }[];
};

export type ToneBin = { bin: number; count: number; toparts: Article[] };

export type GeoFeature = {
  geometry: { coordinates: [number, number] };
  properties: { name: string; count: number; tone: number; theme: string; html: string };
};

export type GdeltEvent = {
  globaleventid: number;
  day: string;
  dateadded: string | null;
  actor1: string;
  actor2: string;
  eventcode: string;
  rootlabel: string;
  quadclass: number;
  quadlabel: string;
  goldstein: number | null;
  avgtone: number | null;
  nummentions: number;
  numsources: number;
  numarticles: number;
  location: string;
  countrycode: string;
  lat: number | null;
  lon: number | null;
  sourceurl: string;
};

export type Facets = {
  matched_articles: number;
  window: { start: string; end: string };
  domains: { value: string; count: number }[];
  languages: { value: string; count: number }[];
  sourcecountries: { value: string; count: number }[];
  themes: { value: string; count: number }[];
  locations: { value: string; count: number }[];
  actors: { value: string; count: number }[];
};

export type Meta = {
  coverage: { start: string; end: string };
  articles: number;
  events: number;
  article_event_links: number;
  mention_files: number;
  source: string;
  built_at: string;
  doc_modes: string[];
  sort_modes: string[];
  query_operators: string[];
  field_provenance: Record<string, string>;
};

export type Window = { start: string; end: string };

async function get<T>(path: string, params: Record<string, string | number | undefined>): Promise<T> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const response = await fetch(`${BASE}${path}?${search.toString()}`);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? `request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

const windowParams = (query: string, span: Window) => ({
  query,
  startdatetime: span.start,
  enddatetime: span.end,
});

export const fetchMeta = () => get<Meta>("/api/v2/ext/meta", {});

export const fetchArticles = (query: string, span: Window, sort: string, maxrecords = 100) =>
  get<{ articles: Article[] }>("/api/v2/doc/doc", {
    ...windowParams(query, span),
    mode: "artlist",
    sort,
    maxrecords,
  });

export const fetchTimeline = (query: string, span: Window, mode: string) =>
  get<{ timeline: TimelineSeries[] }>("/api/v2/doc/doc", {
    ...windowParams(query, span),
    mode,
  });

export const fetchToneChart = (query: string, span: Window) =>
  get<{ tonechart: ToneBin[] }>("/api/v2/doc/doc", { ...windowParams(query, span), mode: "tonechart" });

export const fetchGeo = (query: string, span: Window, maxpoints = 300) =>
  get<{ features: GeoFeature[] }>("/api/v2/geo/geo", { ...windowParams(query, span), maxpoints });

export const fetchEvents = (query: string, span: Window, quadclass?: number, maxrecords = 60) =>
  get<{ events: GdeltEvent[] }>("/api/v2/ext/events", {
    ...windowParams(query, span),
    quadclass,
    maxrecords,
  });

export const fetchFacets = (query: string, span: Window) =>
  get<Facets>("/api/v2/ext/facets", { ...windowParams(query, span), limit: 10 });

export const fetchEventDetail = (id: number) =>
  get<{ event: GdeltEvent; articles: Article[] }>(`/api/v2/ext/events/${id}`, { maxrecords: 25 });

/** GDELT stamps look like 20190315T120000Z. */
export function parseStamp(value: string): Date {
  const [date, time] = value.split("T");
  return new Date(
    Date.UTC(
      Number(date.slice(0, 4)),
      Number(date.slice(4, 6)) - 1,
      Number(date.slice(6, 8)),
      Number(time.slice(0, 2)),
      Number(time.slice(2, 4)),
      Number(time.slice(4, 6)),
    ),
  );
}

export function formatStamp(value: string): string {
  return parseStamp(value).toISOString().slice(0, 16).replace("T", " ") + "Z";
}

export function toStamp(date: Date): string {
  return date.toISOString().replace(/[-:]/g, "").slice(0, 15) + "Z";
}
