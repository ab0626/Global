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
  macro_event_id: number;
  family_id: number;
  include_family: boolean;
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
  return get<{ total: number; events: MacroEvent[]; meta: Meta }>("/search", { q, limit });
}

export function countries() {
  return get<{ publisher_countries: CountryBaseline[]; meta: Meta }>("/countries", {});
}

export function eventCountries(id: number) {
  return get<{ publisher_countries: CountryAttention[]; world_onset: string | null; meta: Meta }>(
    `/events/${id}/countries`,
    {},
  );
}

const PAGE = 2000;

/** Every document of an event in (observed_time, document_id) order, following the
 * stable pagination until `total` is reached. */
export async function fullSpread(
  id: number,
  includeFamily: boolean,
  minConfidence: number,
  signal?: AbortSignal,
): Promise<Spread> {
  let offset = 0;
  let first: Spread | null = null;
  const documents: SpreadDocument[] = [];
  for (;;) {
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const page = await get<Spread>(`/events/${id}/spread`, {
      include_family: includeFamily,
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
