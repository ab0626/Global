# Signal in the Noise — Architecture and Agentic Implementation Plan

Voloridge / HackMIT 2026. Goal: turn the GDELT GKG archive (~5 TB raw) into a small,
queryable store of coherent real-world events, and answer one question end to end:

> "Show the spread of `<event>`" → every article about that event, with its
> observation time and publisher country, animated on a white globe in time order.

Everything below is rule-based or embedding-based. No LLMs touch the data path.
(An LLM may optionally rewrite the free-text query into an entity/date hint; that is
the only place one could sit, and it is not required for the demo.)

---

## 0. Ground truth we are designing around (verified in this repo)

| Fact | Consequence |
|---|---|
| GKG has one row per article: URL, domain, DATEADDED, V2 persons/orgs/locations, themes, tone, TranslationInfo, Extras. `PAGE_TITLE` in Extras exists only from ~2020. | Titles are the only signal that clustered cross-lingually (.87 purity / .98 recall on the 2023 quake). Pre-2020 events must use a weaker channel (entities + event IDs + URL slug). |
| Non-English GKG is in separate `translation.gkg` files; names are machine-translated before NER ("Our Lady", "Their Cathedral"). | Always ingest both file families. Never trust MT'd entity strings as a hard identity. |
| Mentions map URL → GlobalEventID; only ~30% of articles carry an event ID, and IDs are mostly single-country. | Event IDs are a precision channel, not a recall channel. |
| `MentionTimeDate` / `DATEADDED` are GDELT observation times, not publication times. | All timing is "observed media-attention onset". Use k-outlet / 10th-percentile onset, never first mention. |
| GDELT's May-2018 domain→country file resolves 91% of URLs; the US is ~53% of all URLs; one wire story fans out to 300+ URLs. | Publisher-country attribution is heuristic (carry confidence + method) and attention must be normalised per country and deduped for wire copies. |
| One day of 2019: 524k mention rows, 88.7k canonical docs, 215k GKG rows → typed Parquet + audit in minutes on a laptop (already built: `attention/preprocess.py`, `atomic.py`, `cluster.py`, `materialize.py`, `api/attention.py`). | The per-day pipeline exists; the work is the title channel, the scale-out, the query layer, and the frontend. |

Raw sizes: a 15-minute GKG zip is ~10–14 MB; with translation files ~2–2.5 GB/day
zipped, ~10–15 GB/day unzipped. The full archive is a batch job for a fleet, not a
laptop, so the design is **tiered**: a cheap full-archive reduction that is
embarrassingly parallel per 15-minute file, and expensive event resolution only
inside time windows that matter.

---

## 1. System overview

```
                     ┌──────────────────────────────────────────────────────────┐
  GDELT archive      │  TIER 0  Ingest + reduce  (per 15-min file, stateless)    │
  gkg / translation. │  raw zip → typed, deduped, string-encoded Parquet          │
  gkg / mentions /   │  audit.json per file (sha256, rows, malformed, nulls)      │
  events (S3 mirror) │  ~50–100× smaller than raw text; partitioned by day       │
                     └──────────────┬───────────────────────────────────────────┘
                                    ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │  TIER 1  Archive index  (per day, DuckDB/Polars)          │
                     │  documents  (canonical url, domain, country, lang, time,  │
                     │             title, entity ids, event ids, wire group)     │
                     │  entity_daily  (entity, day, doc count)  → burst detector │
                     │  title vectors (fp16, 384/768-d) in per-day .npy + FAISS  │
                     └──────────────┬───────────────────────────────────────────┘
                                    ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │  TIER 2  Event resolution  (per window, e.g. ±3 days)     │
                     │  blocking → pair scoring → sparse graph → Leiden          │
                     │  → incident clusters → story families                     │
                     │  → macro_events / macro_event_documents (+ confidence)    │
                     └──────────────┬───────────────────────────────────────────┘
                                    ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │  TIER 3  Attention store + query API (FastAPI + DuckDB)   │
                     │  country_event_attention (hourly), summaries, sources     │
                     │  /search  /events/{id}  /events/{id}/spread  /timeline    │
                     └──────────────┬───────────────────────────────────────────┘
                                    ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │  FRONTEND  "Show the spread of …"                          │
                     │  white globe, black borders, red markers in time order    │
                     └──────────────────────────────────────────────────────────┘
```

Two compute paths share Tier 0/1:

* **Precomputed path** (the demo): resolve events for a curated set of windows
  (≈ 20–40 famous events × ±3 days, 2015–2025) plus any continuous span we can afford.
  Query hits a precomputed macro-event.
* **On-demand path** (the ambition): a query with no precomputed hit runs the burst
  detector over `entity_daily` to find the window, then Tier 2 on that window only.
  Latency is measured (`resolution_runtime_ms`), not promised. Deferred until after
  M2; unnecessary if the sliding-window sweep covers the archive.

---

## 2. Tier 0 — Ingest and reduction (the "serious preprocessing" showpiece)

Unit of work: one 15-minute file. No state, so it parallelises across processes,
machines, or a spot fleet trivially; a manifest (file → status, sha256, row counts)
makes it resumable.

Per file:
1. Stream-decompress; validate column count per row; count malformed rows (already:
   0 malformed on the 2019 day, but the archive has known bad files).
2. Type everything: UTC timestamps, floats, ints; parse GKG delimited fields
   (themes, persons, orgs, locations with geo precision, tone, GCAM dropped).
3. Canonicalise URL (scheme/host lowercase, strip `utm_*`, `fbclid`, …, trailing
   slash) and domain; dedupe per canonical URL, keep earliest observation.
4. Dedupe entity lists per document; keep counts (mention frequency) as small ints.
5. Extract `PAGE_TITLE` (and `PAGE_LINKS`, `PAGE_AUTHORS` optionally) from Extras
   when present; store language from TranslationInfo.
6. **String encoding of the long tail**: themes, persons, orgs, locations become
   dictionary-encoded categoricals (Parquet dictionary pages). Extras/GCAM are either
   dropped or kept as one opaque string column so nothing is lost but nothing is wide.
7. Write `documents/day=YYYYMMDD/part-HHMM.parquet` (zstd), plus `audit/…json`.

Reduction metrics to report on the poster: raw bytes → Parquet bytes, rows in → docs
out (after URL dedupe), columns in (27 GKG + Extras) → columns out, and the wall-time
per file. Expected ≥ 20× on bytes before any semantic reduction.

Mentions/Events get the same treatment (already implemented) and are joined to
documents by canonical URL to give `document_events` (doc ↔ GlobalEventID, confidence,
InRawText).

### 2.1 Publisher-country resolution (asserted with confidence)

Every domain gets a `publisher_country`, a `country_confidence`, and a `mapping_method`,
resolved by a fixed ladder; the first rung that matches wins and is recorded:

| Rung | Method (`mapping_method`) | Confidence | Status |
|---|---|---|---|
| 1 | GDELT MAY-2018 domain→country list, exact domain (11.6k domains) | 0.9 | implemented; 91% of 2019-day URLs resolve via rungs 1–3 |
| 2 | same list, parent-domain match (`news.example.com` → `example.com`) | 0.7 | implemented |
| 3 | ccTLD (`.de` → DE; generic TLDs never map) | 0.5 | implemented |
| 4 | GDELT 2021 outlet-geography file (newer, larger outlet list) | 0.85 | to add |
| 5 | Wikidata: outlet item → headquarters / country of origin (P159 / P17) | 0.8 | to add, one-time batch, cached |
| 6 | `hreflang` / `<html lang>` + ccTLD agreement (from a one-time HEAD/GET of the homepage) | 0.6 | to add, cached |
| 7 | LLM over the *unresolved domain list only* (a few thousand strings, one-time, cached, `method=llm`), human spot-check of a random 100 | 0.4 | to add, last pass |
| — | unresolved | 0.0 | shown, never silently dropped |

Rung 7 is metadata enrichment of a domain list, not extraction from articles, so it
does not conflict with Voloridge's objection to LLM preprocessing; it is the only
place an LLM is permitted. The `sources` table also records `resolved_at` and the
spot-check outcome so the confidence numbers are auditable. Attention metrics can be
filtered by `min_country_confidence` (default 0.5) and every `/countries` response
reports the share of documents excluded by that filter.

`publisher_country` and `event_country` are always spelled out in full; no table or
endpoint uses a bare `country` column.

## 3. Tier 1 — Archive index (dimensional reduction)

Per day, from Tier 0 output:

* `documents` (one row per canonical article; ~90k/day in 2019, more later) with:
  `doc_id, day, time, url, domain, publisher_country, country_confidence, lang, title,
  event_ids[], person_ids[], org_ids[], loc_ids[], theme_ids[], wire_group`.
  `wire_group` = documents whose (title-normalised | entity set) are identical within
  6h; used as the effective-source unit.
* `entity_daily(entity_id, day, docs, countries)` — the input to burst detection
  (Kleinberg-style: rate today vs trailing 30-day background). This table is tiny
  (millions of rows for the whole archive) and is the "where do I look" index.
* Title vectors: multilingual sentence encoder (`paraphrase-multilingual-mpnet-base-v2`
  or `multilingual-e5-base`), fp16, one `.npy` per day + a FAISS IVF index per month.
  CPU cost measured at ~13k titles / 40 s; a day of ~150k titles ≈ 8 min/core; a
  GPU box does a year in hours.
* Pre-2020 fallback vector: TF-IDF over URL-slug tokens + entity IDs (weak but
  language-agnostic for event IDs; already implemented as the v2 URL channel).

Storage estimate: documents ≈ 30–60 MB/day, vectors ≈ 100–200 MB/day fp16.
A full decade ≈ 0.5–1 TB, i.e. an order of magnitude below raw and *queryable* with
DuckDB directly on Parquet.

## 4. Tier 2 — Event resolution (scalable, cross-lingual, uncertainty-aware)

Runs on a window (day range). Never global O(N²): candidate generation is blocked.

Configuration is explicit and frozen (no magic numbers in code):

```yaml
event_window_days: 3        # W_search: where an event may live
candidate_max_hours: 48     # W_candidate: max time gap for a candidate pair
temporal_decay_hours: 12    # tau_time: soft decay inside the candidate window
```

Two resolution regimes, calibrated and reported separately, never blended:
`title_multilingual_v1` (titles present, ≥ 2020) and `legacy_metadata_v1`
(URL-slug + entity + event-ID, pre-2020). Every macro-event carries
`resolution_model`; the UI shows legacy events with lower confidence.

1. **Nodes** = documents (one per canonical URL). Wire copies are *not* collapsed
   before clustering (that collapse caused false merges on the 2019 slice); they are
   attached to the cluster afterwards.
2. **Candidate retrieval** (union of blocks, top-k each):
   * FAISS top-50 on title vectors (≥2020) or slug/entity TF-IDF top-50 (pre-2020),
   * shared GlobalEventID,
   * shared rare entity (IDF-capped: an entity in > 10% of the window's docs is not a
     block key).
3. **Pair scoring** — channels, each in [0,1]:
   * title cosine, event-ID IDF-weighted overlap, entity Jaccard (persons+orgs; locations
     down-weighted — they pull toward geography), URL-token cosine, time proximity
     (soft, ±48h, never multiplicative so slow international coverage isn't erased).
   * **Evidence gate**: an edge survives only if ≥ 2 channels agree or one channel is
     very strong (this killed the junk agglomerations in the March experiment).
   * Weights are fixed initially; if we produce ~300 labelled pairs, fit a logistic
     pair scorer (the SOTA step; a cross-encoder is out of scope without labels).
4. **Graph → Leiden** (modularity, resolution tuned on the labelled set), isolates and
   clusters below a support floor → `unassigned`.
5. **Two levels, kept separate**: incident clusters (Leiden output, `incident_id`) →
   story families (`family_id`, linking incidents whose centroids/entities overlap
   across days: "fire" → "Macron rebuild pledge"). Both IDs are materialised on every
   membership row; clustering evaluation is done on incidents only, family linking
   is evaluated separately. The demo query returns the *family*; the timeline can
   drill into incidents.
   **Membership is not exclusive**: `incident_memberships(document_id, incident_id,
   assignment_score, is_primary)`. Leiden gives the primary; secondary rows are kept
   when a document's summed edge weight into another incident exceeds
   `secondary_min_score` (roundups, multi-angle pieces). `assignment_score` is a
   score, not a probability, until a labelled set exists to calibrate it.
6. **Uncertainty attached to everything**:
   * per document: `assignment_score` (edge weight into the winning incident
     relative to runners-up), `is_primary`, `is_wire_copy`, `country_confidence`;
   * per event: `cluster_confidence` (entity coherence), `candidate_recall_estimate`
     (share of labelled positives that were ever candidates), size, largest-fraction;
   * per attention curve: `suppressed` when a country never reaches 3 outlets.
7. **Evaluation** (Tier-2 acceptance): a hand-labelled set of ~300 pairs + ~5 fully
   labelled events (drawn as anchor neighbourhoods + hard negatives, wire groups kept
   together). Report pairwise P/R, B³, CEAF-e, cluster count, unassigned rate. Keyword
   regexes stay as a smoke test only.

Every intermediate is a Parquet file so any stage can be inspected on its own:

```
documents.parquet → document_events.parquet → title_embeddings.npy
  → candidate_pairs.parquet → pair_features.parquet → graph_edges.parquet
  → incident_memberships.parquet → event_families.parquet
  → country_event_attention.parquet
```

Each run writes `run.json` (config, `resolution_model`, git sha, row counts per
stage, `resolution_runtime_ms`). No latency is claimed anywhere until it is read from
that file.

## 5. Tier 3 — Attention store and query layer

Materialised per run (all DuckDB-queryable Parquet):

```
macro_events(incident_id, family_id, resolution_model, label, start, end,
             event_country, lat, lon, types[], people[], orgs[], themes[],
             raw_documents, unique_domains, effective_reports, country_count,
             cluster_confidence, centroid_vector)
event_families(family_id, label, start, end, incident_ids[], resolution_model)
incident_memberships(document_id, incident_id, family_id, assignment_score,
             is_primary, wire_group)
macro_event_documents(incident_id, family_id, doc_id, url, title, domain,
             publisher_country, country_confidence, lang, time, event_ids[],
             assignment_score, is_primary, wire_group)
country_event_attention(family_id, publisher_country, hour, raw_documents,
             unique_domains, effective_reports, raw_share, effective_share,
             world_effective_share, attention_ratio, cumulative_effective, onset_flag)
country_event_summary(family_id, publisher_country, raw_documents, unique_domains,
             effective_reports, raw_share, effective_share, attention_ratio,
             first_seen, third_source_seen, p10_seen, onset, lag_hours, suppressed)
country_baseline(publisher_country, window, raw_documents, effective_reports,
             domains)                                         -- denominators
sources(domain, publisher, publisher_country, country_confidence, mapping_method,
             resolved_at, spot_checked)
```

Semantics (three volume measures are always carried side by side):

```
raw_documents      = canonical URLs
unique_domains     = distinct publisher domains
effective_reports  = Σ_g min(1, n_{g,c,t})   over wire groups g   (primary numerator)

raw_share(c,e)       = raw_documents(c,e)      / raw_documents(c,window)
effective_share(c,e) = effective_reports(c,e)  / effective_reports(c,window)
attention_ratio(c,e) = effective_share(c,e)    / effective_share(world,e)

first_seen          = earliest observation in c
third_source_seen   = hour the 3rd distinct outlet in c appears
p10_seen            = 10th-percentile observation hour in c
onset               = max(third_source_seen, p10_seen);  null + suppressed if < 3 outlets
lag_hours           = onset(c) − onset(world)
```

All four onset ingredients are stored so any country's onset can be explained.

API (FastAPI; the `/api/v2/attention/*` router exists, add `/search` and `/spread`):

```
GET /search?q=notre dame fire        hybrid: BM25 over label/people/orgs/themes
                                     + cosine of encoded query vs event centroids
                                     + optional date hint; returns ranked events with
                                     confidence and "why matched"
GET /events/{id}                     metadata + sample docs
GET /events/{id}/spread              THE DEMO PAYLOAD: articles sorted by time with
                                     {time, publisher_country, country_confidence,
                                     lat, lon (country centroid or publisher city if
                                     known), domain, title, url, lang, is_wire_copy,
                                     assignment_score, is_primary}; paginated or
                                     streamed (NDJSON) so the globe can start
                                     animating immediately
GET /events/{id}/timeline            hourly per country (raw / normalised)
GET /events/{id}/countries           share, ratio, onset, lag, suppressed
GET /event-types/{type}/countries    who over-attends / reacts fastest by type
GET /countries                       baselines
```

Every response carries `meta`: window, denominators, timing semantics, run id,
`resolution_model`, `resolution_runtime_ms`, `min_country_confidence` and the share of
documents excluded by it. The on-demand path (burst detector → window → Tier-2 job →
`{"status": "resolving"}` + polling) is **deferred until after M2**; if the sliding
window sweeps the archive, `/search` over precomputed families already answers
arbitrary queries and the burst detector may never be needed.

## 6. Frontend — "Show the spread of …"

Stack: Vite + React + TypeScript (the repo's `web/` already uses this), globe via
`react-globe.gl`/`three-globe` (WebGL, smooth with tens of thousands of points) or D3
orthographic SVG if we want the flat editorial look. Palette: white sphere, black
borders (Natural Earth 110m TopoJSON), red markers, grey UI, blue for the selected
country/line.

Screens:
1. Landing: single line "Show the spread of" + text box; typeahead from `/search`.
2. Result: globe animating red markers in chronological order (marker size ∝ log docs
   in that country-hour; wire copies fade), a scrubber/timeline underneath showing
   cumulative articles per country as stacked area, a right rail with the event card
   (label, start, confidence, top entities) and the country table (share, ratio, onset,
   lag). Click a country → its article list. Toggle raw vs normalised.
3. Optional second view: event type → which countries over/under-attend (bar/ratio
   chart) — cheap, reuses existing endpoint, and answers "what does the US ignore?"

Performance: `/spread` streams NDJSON; the globe buffers by hour; 20k markers is fine
in WebGL. Country centroid from a static JSON; no per-request geo work.

## 7. Agentic implementation plan

**Order of work: backend first, tested, then frontend.** No frontend work starts on
live data until the backend for M1 passes its test gate below; the frontend can build
against the M0 fixtures in the meantime but is not the critical path.

Freeze the contracts first (M0), then the lanes run in parallel. Each lane is one
agent/session with a fixed interface and acceptance test.

**M0 — contracts (1 session, blocks everyone)**
* Parquet schemas above committed as `schemas/*.sql` (DuckDB DDL) + `docs/data-dictionary.md`.
* OpenAPI for `/search`, `/events/{id}/spread`, `/timeline`, `/countries` with a
  fixture response set (JSON) so the frontend can start without a backend.
* Target windows list (`windows.yaml`): ~25 events 2015–2025 with ±3 day ranges
  (Paris attacks 2015, Brexit vote, Trump election, Notre-Dame 2019, Christchurch,
  COVID WHO declaration, Beirut blast 2020, Capitol riot, Suez blockage, Kabul fall,
  Ukraine invasion 2022-02-24, Queen Elizabeth death, Turkey quake 2023, Oct 7 2023,
  Titan sub, etc.). Titles exist for the ≥2020 ones; pre-2020 are the "hard mode" set.

**Lane A — Tier 0/1 pipeline at scale**
* Generalise `attention.preprocess` to any day, both `gkg` + `translation.gkg`,
  Extras title extraction, dictionary encoding, manifest-based resume.
* Runner: `uv run python -m attention.ingest --window windows.yaml --workers N` on
  the EC2 box (or a spot fleet reading the S3 mirror); per-file audit.
* Build `entity_daily` and the burst detector; title embedding job with fp16 + FAISS.
* Acceptance: all target windows processed; reduction report (bytes, rows, cols,
  wall-time) in `docs/reduction.md`; audit shows malformed/missing files explicitly.

**Lane B — Event resolution**
* Add the title channel to `attention.cluster`, remove fingerprint node collapse,
  add story-family layer and assignment probabilities.
* Label set: ~300 pairs + 5 events (Notre-Dame 2019, Turkey quake 2023, Beirut 2020,
  Queen's death 2022, Ukraine invasion 2022) — two people, 1–2 hours, agreement
  reported. Evaluation script writes `docs/clustering-eval.md`.
* Acceptance: each target event forms one family with ≥ 0.8 purity on labels, ≥ 5
  countries, largest cluster < 20% of window docs, unassigned rate reported.

**Lane C — Store + API**
* `materialize` over families; `/search` (BM25 via DuckDB FTS + centroid cosine);
  `/spread` streaming; on-demand resolution job with status polling.
* Acceptance: fixture contract tests pass against real data; `/spread` for the Ukraine
  invasion (largest) returns in < 1 s first byte; every response has `meta`.

**Lane D — Frontend**
* Globe + animation from fixtures (day 1), swap to live API (day 2), polish palette,
  scrubber, country rail, raw/normalised toggle, event-type view.
* Acceptance: 60 fps with 20k markers, works for all 25 windows, empty/suppressed
  states handled, share-link URL encodes event id + time cursor.

**Backend test gate (must pass before frontend goes live)**

Deterministic pytest suite over the Turkey-quake window, run from a clean
`rm -rf data/` so nothing is hand-injected:

* Reproducibility: two runs with the same config produce identical
  `incident_memberships` (fixed seeds) and identical `run.json` row counts.
* Sensible results: the earthquake family exists, spans ≥ 5 publisher countries and
  ≥ 3 languages, TR onset ≤ every other country's onset, US/DE/BR lag within a few
  hours of the teammate's independent measurement (+1h), JP later (+4h); largest
  incident < 20% of window docs; unassigned rate reported and < 50%.
* Negative controls: a Chile-wildfire article and a Turkey country-profile page are
  *not* primary members of the quake family; a randomly permuted title column
  destroys the family (guards against the graph being held together by boilerplate).
* Edge cases explicitly tested: window with zero documents; a day with a missing
  15-minute file (audit records it, run does not crash); a document with no title
  under `title_multilingual_v1` (falls to legacy channels, flagged); a domain with
  unresolved country (excluded from shares at default confidence, counted in `meta`);
  a country with exactly 2 outlets (suppressed, null onset); a wire story on 300
  domains (effective_reports = 1 per country-hour, raw = 300); duplicate canonical
  URLs across `gkg` and `translation.gkg` (one document, language recorded); query
  strings with diacritics/CJK ("séisme", "地震") hit the family via centroid cosine;
  `/spread` pagination is stable under concurrent runs; `min_country_confidence=1.0`
  yields empty but well-formed responses.
* Contract: every endpoint validated against the OpenAPI fixtures; no bare `country`
  field anywhere.

**M1** = one event (Turkey quake 2023) end to end through live API on the globe,
backend gate green first.
**M2** = all 25 windows precomputed; on-demand path working for one unseen event.
**M3** = polish, reduction/eval write-ups, poster numbers.

Sequencing risk: Lane B is the only one with research risk; if titles under-deliver
on some pre-2020 window, the family layer still returns "entity+event-ID" clusters
with lower confidence, and the UI shows the confidence — degrade, don't fail.

## 8. What we explicitly do not claim

* Timing is observed GDELT attention onset, not publication time or "when a country
  found out".
* Country is publisher country with a confidence; not audience, not event location.
* Pre-2020 clusters are lower confidence and say so.
* No number is called precision/recall unless it comes from the hand-labelled set.
