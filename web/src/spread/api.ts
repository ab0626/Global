const BASE = `${import.meta.env.VITE_API_BASE ?? ""}/api/v2/attention`;

export type Meta = {
  window_start: string;
  window_end: string;
  resolution_model: string;
  documents_total: number;
  macro_events: number;
  timing_caveat?: string;
  [key: string]: unknown;
};

export type MacroEvent = {
  macro_event_id: number;
  family_id: number;
  title: string | null;
  label: string | null;
  start_time: string;
  end_time: string;
  event_country: string | null;
  lat: number | null;
  lon: number | null;
  event_types: string[];
  raw_documents: number;
  unique_domains: number;
  effective_reports: number;
  publisher_country_count: number;
  language_count: number;
  title_hits?: number;
};

export type Family = {
  family_id: number;
  title: string | null;
  label: string | null;
  incident_count: number;
  raw_documents: number;
  effective_reports: number;
  publisher_country_count: number;
  language_count: number;
  start_time: string;
  end_time: string;
  title_hits?: number;
};

export type SpreadDocument = {
  document_id: number;
  macro_event_id: number;
  incident_id: number;
  observed_time: string;
  publisher_country: string | null;
  publisher_country_confidence: number | null;
  source_domain: string;
  language: string | null;
  title: string | null;
  url: string;
};

export type SpreadCountry = {
  publisher_country: string;
  raw_documents: number;
  effective_reports: number;
  first_seen: string;
};

export type Spread = {
  macro_event_id?: number;
  family_id: number;
  include_family?: boolean;
  total: number;
  excluded_documents: number;
  offset: number;
  limit: number;
  countries: SpreadCountry[];
  documents: SpreadDocument[];
  meta: Meta;
};

export type CountryBaseline = {
  publisher_country: string;
  country_documents: number;
  country_domains: number;
  country_effective_reports: number;
  lat: number | null;
  lon: number | null;
  country_name?: string | null;
};

export type CountryAttention = {
  publisher_country: string;
  raw_documents: number;
  unique_domains: number;
  effective_reports: number;
  attention_ratio: number | null;
  lag_hours: number | null;
  onset: string | null;
  suppressed: boolean;
};

async function get<T>(path: string, params: Record<string, string | number | boolean>): Promise<T> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) search.set(key, String(value));
  const response = await fetch(`${BASE}${path}?${search}`);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return (await response.json()) as T;
}

export function search(q: string, limit = 12) {
  return get<{
    total: number;
    total_families: number;
    families: Family[];
    events: MacroEvent[];
    meta: Meta;
  }>("/search", { q, limit });
}

export function familyDetail(id: number) {
  return get<{ family: Family; events: MacroEvent[]; meta: Meta }>(`/families/${id}`, {});
}

export function countries() {
  return get<{ publisher_countries: CountryBaseline[]; meta: Meta }>("/countries", {});
}

export type EvidenceNeighbor = {
  document_id: number;
  incident_id: number;
  macro_event_id: number | null;
  title: string | null;
  language: string | null;
  source_domain: string | null;
  publisher_country: string | null;
  observed_time: string | null;
};

/** One gated graph edge between the explained document and a neighbour. */
export type EvidenceEdge = {
  neighbor: EvidenceNeighbor;
  same_incident: boolean;
  rank: number;
  title_score: number;
  event_score: number;
  url_score: number;
  entity_score: number;
  delta_hours: number;
  evidence_channels: number;
  gated: number;
  same_publisher_country: boolean;
};

export type Evidence = {
  document: SpreadDocument & { family_id: number; assignment_score: number };
  incident: MacroEvent | null;
  family: Family | null;
  assignment_score: number;
  checks: {
    title_similarity: number | null;
    shared_gdelt_event: boolean;
    shared_url_tokens: boolean;
    shared_entities: boolean;
    hours_to_nearest_support: number | null;
    other_publisher_country: boolean;
  };
  supporting: EvidenceEdge[];
  competing: EvidenceEdge[];
  note: string;
  meta: Meta;
};

export function documentEvidence(id: number) {
  return get<Evidence>(`/documents/${id}/evidence`, {});
}

export type CountriesResponse = {
  publisher_countries: CountryAttention[];
  world_onset: string | null;
  meta: Meta;
};

export function eventCountries(id: number) {
  return get<CountriesResponse>(`/events/${id}/countries`, {});
}

export function familyCountries(id: number) {
  return get<CountriesResponse>(`/families/${id}/countries`, {});
}

const PAGE = 2000;

/** Every document of an incident (`/events/{id}/spread`) or a story family
 * (`/families/{id}/spread`) in (observed_time, document_id) order, following the
 * stable pagination until `total` is reached. */
export async function fullSpread(
  target: { kind: "family" | "incident"; id: number },
  minConfidence: number,
  signal?: AbortSignal,
): Promise<Spread> {
  const path = `/${target.kind === "family" ? "families" : "events"}/${target.id}/spread`;
  let offset = 0;
  let first: Spread | null = null;
  const documents: SpreadDocument[] = [];
  for (;;) {
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const page = await get<Spread>(path, {
      min_country_confidence: minConfidence,
      limit: PAGE,
      offset,
    });
    first ??= page;
    documents.push(...page.documents);
    offset += page.documents.length;
    if (page.documents.length === 0 || offset >= page.total) break;
  }
  return { ...(first as Spread), documents };
}

/* ------------------------------------------------ country response analytics */

export type SupportStatus = {
  status: "ok" | "insufficient";
  min_covered_events: number;
  min_effective_reports: number;
  seed: number;
  bootstrap_samples: number;
};

/** Coverage + latency of one bag of eligible observations (family or incident level).
 * Latency fields are null when the support gate is not met. */
export type ResponseSummary = {
  eligible_events: number;
  covered_events: number;
  uncovered_events: number;
  coverage_rate: number | null;
  coverage_ci_low: number | null;
  coverage_ci_high: number | null;
  effective_reports: number;
  world_fallback_events: number;
  mean_response_hours: number | null;
  median_response_hours: number | null;
  p25_response_hours: number | null;
  p75_response_hours: number | null;
  latency_ci_low: number | null;
  latency_ci_high: number | null;
  fastest_response_hours: number | null;
  slowest_response_hours: number | null;
  support: SupportStatus;
};

export type TypeSummary = ResponseSummary & { event_types: string };
export type MagnitudeSummary = ResponseSummary & { magnitude: string };
export type OriginSummary = ResponseSummary & { origin_country: string; origin_country_name?: string };
export type DestinationSummary = ResponseSummary & {
  destination_country: string;
  destination_country_name?: string;
};

export type AnalyticsFilters = {
  level: "family" | "incident";
  event_types: string[];
  min_event_effective_reports: number;
  start: string | null;
  end: string | null;
  resolution_model: string | null;
  reference: "origin_preferred" | "origin_only" | "world";
};

export type AnalyticsEnvelope = {
  filters: AnalyticsFilters;
  support: Omit<SupportStatus, "status">;
  caveats: string[];
  meta: Meta;
};

export type CountryRow = {
  publisher_country: string;
  publisher_country_name: string;
  foreign: ResponseSummary;
  domestic: ResponseSummary;
};

export type CountryOverview = AnalyticsEnvelope & {
  publisher_country: string;
  publisher_country_name: string;
  foreign: ResponseSummary;
  domestic: ResponseSummary;
  by_event_type: TypeSummary[];
  by_origin: OriginSummary[];
  by_magnitude: MagnitudeSummary[];
  domestic_by_event_type: TypeSummary[];
};

export type ResponseEvent = {
  level: "family" | "incident";
  event_id: number;
  family_id: number;
  title: string | null;
  event_start: string;
  event_types: string[];
  event_effective_reports: number;
  origin_country: string;
  origin_country_name?: string;
  destination_country: string;
  destination_country_name?: string;
  origin_onset: string | null;
  destination_onset: string | null;
  world_onset: string | null;
  response_hours: number | null;
  response_reference: "origin" | "world_fallback" | "event_start" | "world" | null;
  covered: boolean;
  suppressed: boolean | null;
  has_documents: boolean;
  censor_hours: number | null;
  raw_documents: number | null;
  effective_reports: number | null;
  attention_ratio: number | null;
};

export type PairBlock = {
  origin_country: string;
  origin_country_name: string;
  destination_country: string;
  destination_country_name: string;
  kind: "foreign" | "domestic";
  summary: ResponseSummary;
  by_event_type: TypeSummary[];
  events?: ResponseEvent[];
};

export type PairResponse = AnalyticsEnvelope & {
  forward: PairBlock & { events: ResponseEvent[] };
  reverse?: PairBlock;
};

export type MatrixCell = ResponseSummary & {
  origin_country: string;
  origin_country_name?: string;
  destination_country: string;
  destination_country_name?: string;
  kind: "foreign" | "domestic";
};

export type MatrixResponse = AnalyticsEnvelope & {
  origins: { code: string; name: string }[];
  destinations: { code: string; name: string }[];
  cells: MatrixCell[];
  destination_rows: (CountryRow & { foreign_from_selected_origins: ResponseSummary })[];
};

export type OriginView = AnalyticsEnvelope & {
  origin_country: string;
  origin_country_name: string;
  world: ResponseSummary;
  domestic: ResponseSummary;
  by_destination: DestinationSummary[];
  by_event_type: TypeSummary[];
};

export type EventsResponse = AnalyticsEnvelope & {
  total: number;
  offset: number;
  limit: number;
  summary: ResponseSummary;
  events: ResponseEvent[];
};

/** Query-string form of the composable analytics filters + support gate. */
export type AnalyticsQuery = {
  level: "family" | "incident";
  event_type?: string[];
  min_event_effective_reports?: number;
  start?: string;
  end?: string;
  reference: "origin_preferred" | "origin_only" | "world";
  min_covered_events: number;
  min_effective_reports: number;
};

type QueryValue = string | number | boolean | string[] | undefined;

async function getMulti<T>(path: string, params: Record<string, QueryValue>): Promise<T> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue;
    if (Array.isArray(value)) for (const v of value) search.append(key, v);
    else search.set(key, String(value));
  }
  const response = await fetch(`${BASE}/analytics${path}?${search}`);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return (await response.json()) as T;
}

export function eventTypes() {
  return get<{ types: { type: string; events: number; raw_documents: number }[]; meta: Meta }>(
    "/event-types",
    {},
  );
}

export function analyticsCountries(q: AnalyticsQuery) {
  return getMulti<AnalyticsEnvelope & { publisher_countries: CountryRow[] }>("/countries", { ...q });
}

export function analyticsCountry(code: string, q: AnalyticsQuery, topOrigins = 25) {
  return getMulti<CountryOverview>(`/countries/${code}`, { ...q, top_origins: topOrigins });
}

export function analyticsPair(origin: string, destination: string, q: AnalyticsQuery, limit = 500) {
  return getMulti<PairResponse>("/pairs", { ...q, origin, destination, limit });
}

export function analyticsMatrix(origins: string[], destinations: string[], q: AnalyticsQuery) {
  return getMulti<MatrixResponse>("/matrix", { ...q, origin: origins, destination: destinations });
}

export function analyticsOrigin(code: string, q: AnalyticsQuery) {
  return getMulti<OriginView>(`/origins/${code}`, { ...q });
}

export function analyticsEvents(
  origin: string,
  destination: string,
  kind: "all" | "foreign" | "domestic",
  q: AnalyticsQuery,
  limit = 500,
) {
  return getMulti<EventsResponse>("/events", { ...q, origin, destination, kind, limit });
}
