# Global News Event Resolution & Attention Propagation — Project Report

Status as of 2026-09-20. Everything below is measured on the code in
https://github.com/ab0626/Global (PRs #1 → #2 → #4 → #5 for the pipeline, globe
and docs, #8 → #9 for country response analytics; all merged into `main`, which
is the only branch). Numbers come from `run.json`, `meta.json`, `eval/*.json` and the
Parquet tables of the deterministic Feb 6–8 2023 `_v5` run
(`data/clusters/20230206_v5`, `data/store/20230206_v5`) unless another run is
named. Where a number is an estimate or an inference rather than a measurement
it says so. The condensed version, with the maths written out, is the root
`readme.md`.

---

## 1. What the system does

Input: raw GDELT 2.x 15-minute files (GKG, translated GKG, Mentions, translated
Mentions, Events/export, translated export).

Output: a small, typed, queryable store of **incidents** (one news event, e.g.
"quake death toll passes 4,000"), **story families** (all incidents of one
real-world story, e.g. the Turkey–Syria earthquake), and **country-level
attention** (which publisher countries covered it, when, how much relative to
their normal output), served by a FastAPI API and visualised on a React Three
Fiber 3D globe where publisher countries light up in observation order, with a
per-article "why is this article in this event?" evidence panel. On top of the
store, a **country response analytics** layer (§4b, §6b) makes countries
queryable the way stories are: how quickly Germany's press responds to foreign
events, to French-origin events specifically, to its own domestic events, and
how any set of countries compares in an origin × destination matrix — every
number drillable to the story families that produced it.

```
raw 15-min zips ──preprocess──▶ typed Parquet (+audit)
                ──atomic─────▶ one row per canonical URL, wire groups, doc↔event links
                ──embed──────▶ multilingual title vectors (384-d) + FAISS
                ──cluster────▶ candidate pairs → pair features → graph → Leiden incidents → families
                ──materialize▶ macro_events / event_families / macro_event_documents /
                               country_event_attention / country_event_summary /
                               sources / country_baseline / meta.json
                               + country_family_* + document_evidence
                ──analytics──▶ country_response_observations (event × origin × destination)
                ──api────────▶ /search /families/{id}/spread /documents/{id}/evidence /analytics/* ...
                ──web────────▶ spread.html: Events tab (R3F globe, timeline, evidence panel)
                               + Country Stats tab (overview, A↔B, matrix/heatmap, drilldown)
                ──eval───────▶ labelled pairs/neighbourhoods → pair P/R, B³, CEAF-e → sweeps → flagship audit
```

No LLM is used anywhere in the pipeline. Every stage is rule-based or a
published, deterministic model (sentence-transformers, FAISS, Leiden with a fixed
seed). The only place an LLM is *planned* is a one-time, cached pass over the
~700 publisher domains the deterministic ladder cannot place (metadata
enrichment, `method=llm`, confidence 0.4, human spot-check) — not article
extraction and not clustering.

---

## 2. Data

### 2.1 Coverage

| Window | Files | What it is | Model used |
|---|---|---|---|
| **Feb 6–8 2023** (Turkey–Syria quake, Chinese balloon, State of the Union, Grammys, Ohio derailment) | 1,728 (all 6 families × 96 slots × 3 days), 4.33 GB zipped | Full multilingual store the demo runs on (`_v5`) | `title_multilingual_v1` |
| Apr 15 2019 (Notre-Dame fire) | English GKG + Mentions + Events only | Early vertical slice (PR #4) | `legacy_metadata_v1` (no titles pre-2020) |
| Aug 4–6 2020 (Beirut explosion) | 1,728 files, 5.33 GB zipped → 471 MB Parquet (11.3×); 1.716 M mention rows, 1.335 M docs, 805 k translated, 64 languages, 1.326 M titles (≈ 998 k distinct), 23,595 domains | **preprocess + atomic + embedding done** (embedding on a 48-core CPU EC2 box, 50 resumable 20k-title shards); clustering/materialisation status is in §9 — **not servable until they finish and the API smoke checks pass** | `title_multilingual_v1` |
| Feb 24–26 2022 (Ukraine invasion), Sep 8–10 2022 (Queen Elizabeth) | not fetched | queued | — |

GDELT itself runs Feb 2015 → today; the pipeline is date-agnostic
(`scripts/fetch_window.py` + `attention.preprocess --start --end`). Cost to add a
window: ~1.4–1.8 GB zipped per day to fetch; on this CPU-only box preprocess ≈ 10
min, embedding ≈ 38 min for 0.9 M titles when the box is otherwise idle
(embedding is the step a GPU would collapse to minutes; `attention.embed`
encodes each distinct title once and writes resumable 20k-title shards, so an
interrupted job restarts where it stopped), full-feature cluster ≈ 61 min
(`_v3`), re-gate + Leiden + families from cached `pair_features` ≈ 8.5 min
(`_v5`), materialize ≈ 4 min.

Constraint: **PAGE_TITLE exists in GKG `Extras` only from ~2020**. Pre-2020
windows fall back to metadata-only clustering, which our own experiments and the
teammate's Notre-Dame experiment showed is not cross-lingual (entities are
machine-translated per language; "Notre Dame" → "Our Lady"/"Her East").

### 2.2 Preprocessing results (Feb 6–8 2023)

```
raw_zip_bytes            4,325,889,501
parquet_bytes              347,883,358      (12.4× reduction)
mention_rows                 1,229,240      (all web mentions)
distinct_web_urls              258,784
event_rows                     509,150
gkg_documents                  914,199      (English 366k + translated 548k)
gkg_with_page_title            909,599      (99.4 %)
gkg_languages                       63
domains                         14,635
malformed_rows        mentions 0 / events 0 / gkg 99   (kept in the audit, dropped from tables)
missing_files                        0
```

What preprocessing does (all in `attention/preprocess.py`):
- Reads both English and `translation.*` file families; case-tolerant filenames
  (`mentions.CSV`, `gkg.csv`), zipped or unzipped.
- Per-file sha256, row count, malformed count → `audit.json`; missing slots are
  recorded (`--allow-partial`) instead of failing; an empty window produces empty
  typed tables instead of raising.
- Typed schemas for Events (61 cols), Mentions (16), GKG (27 + extracted fields);
  all timestamps → UTC.
- URL canonicalisation (scheme/host lowercased, `www.` and tracking params
  stripped, fragment removed) → `canonical_id`; domain canonicalisation.
- `PAGE_TITLE` extracted from the GKG `Extras` XML; `source_language` and
  `translated` flag from `TranslationInfo`.
- Dedupe of the same canonical URL appearing in both English and translated GKG
  (English row kept, provenance preserved).
- `sources` table = publisher-country resolution ladder:

  | method | confidence | Feb store domains |
  |---|---|---|
  | `gdelt_lookup` (GDELT's own domain→country list) | 0.9 | 11,176 |
  | `gdelt_lookup_parent` (parent domain matched) | 0.7 | 2,596 |
  | `cctld` (ISO ccTLD → FIPS mapping) | 0.5 | 145 |
  | `unresolved` | 0.0 | 718 |

  98.9 % of documents get a publisher country. The ccTLD path originally emitted
  ISO codes while GDELT uses FIPS (`.de` → `GM`, `.ge` → `GG`, `.jp` → `JA`,
  `.tr` → `TU`, `.ua` → `UP`); this was a real bug (German outlets landed under a
  nonexistent code, Georgian ones under Germany) and is fixed with tests.
  **Publisher country = where the outlet is, never where the event happened.**

### 2.3 Atomic layer (`attention/atomic.py`)

```
documents                          915,355   (one row per canonical URL; GKG ⟗ web Mentions)
documents_with_title               909,599
documents_with_events              258,784   (linked to ≥1 GlobalEventID)
documents_translated               548,362
documents_with_publisher_country   905,335
document_event_links             1,200,850
atomic_events                      511,993   (GlobalEventIDs; 385k have a single document)
distinct_wire_groups               693,628
```

`wire_group` = normalised title (fallback: GKG entity fingerprint). It is used
only to compute `effective_reports` (so 200 AP copies count once); graph identity
is always the individual document.

---

## 3. Event resolution (clustering)

### 3.1 Design

Every document is a node. Edges come from four channels, each producing a
bounded candidate set (never global O(N²)):

| channel | what | Feb pairs |
|---|---|---|
| `title` | FAISS top-30 cosine over multilingual title vectors (`paraphrase-multilingual-MiniLM-L12-v2`, 384-d, 64 tokens) | 21.4 M directed |
| `event` | IDF-weighted shared `GlobalEventID` | 3.4 M |
| `url` | shared rare URL-slug tokens (host boilerplate + tokens in >2000 articles removed) | 22.6 M |
| `entity` | shared rare GKG persons/orgs/names (features in >10 % of docs removed) | 16.0 M |

Pipeline numbers (Feb 6–8), three successive gates on the same candidates:

```
                          _fam (first demo)   _v3 (stricter gate)   _v5 (served)
candidate_pairs              47,447,177          47,457,112          47,457,112
pairs_in_window              43,300,513          43,313,145          43,313,145   (candidate_max_hours = 48)
pairs_gated                  14,975,318          13,550,918          12,221,967
edges                        13,260,330          12,653,115          11,414,026
incidents                       117,214             118,328             120,026   (CPM-Leiden, res 0.05, seed 2026)
  ≥10 documents                  17,386                   –              16,338
  ≥50 documents                   2,896                   –               2,561
  largest                           640                 640                 640
unassigned                       74,180 (8.1 %)      (9.2 %)    104,891 (11.5 %)  explicit incident_id = -1
secondary memberships           219,063             217,045             204,894   (edge mass into a non-winning cluster ≥ 0.25)
families                        101,626             102,166             102,857
cluster runtime                 2,140 s             3,666 s               506 s   (_v5 reuses _v3 pair_features)
embedding (cached)              2,265 s
```

Gate evolution: `_fam` = ≥2 channels agree or one channel above its floor;
`_v3` adds `single_channel_min_features=2` (a lone URL/entity/event channel needs
≥2 shared features) and `single_channel_title_veto=0.15` (a lone non-title channel
is vetoed when both titles exist and disagree); `_v5` adds
`cross_language_title_floor=0.60` (a lone title edge between documents in
different languages must score ≥ 0.60 instead of the same-language 0.55).

Lineage is persisted for audit: `candidate_pairs.parquet`, `pair_features.parquet`,
`graph_edges.parquet`, `incident_memberships.parquet` (`assignment_score`,
`is_primary`), `incidents.parquet`, `run.json` (all settings + counts).

### 3.2 Title calibration (why it was needed)

The first real run showed the multilingual model's *background* similarity is
language dependent: random Korean/Arabic title pairs sit at cos ≈ 0.3, English at
≈ 0.1. A single global threshold therefore produced same-language local-news blobs.
Fix (`TitleChannel`):

1. subtract the per-domain mean vector (≥50 titles) — removes "| Site – News" boilerplate;
2. subtract the per-language mean vector;
3. estimate a background quantile (p90 of 4,000 random cross pairs) per language
   pair, prior 0.17 for small groups; store in `title_background.parquet`;
4. `score = clip((cos − background) / (1 − background), 0, 1)`;
5. drop `(domain, title)` pairs repeated ≥10× (14,938 boilerplate titles).

### 3.3 Story families

Incidents are high-precision but fragmented (the quake is ~200 incidents: first
report, death-toll updates per language, aid pledges, …). Families re-link them.
The first family layer over-merged (one 24k-doc blob spanning quake + unrelated
Turkey/Syria coverage). Current `link_families`:

- candidate incident pairs from calibrated centroid title similarity ≥ 0.4;
- **temporal gate**: incident `first_seen` ranges must be within 48 h of overlapping;
- **corroboration**: keep only if the incidents share a top GKG entity or a
  `GlobalEventID`, *or* centroid title score ≥ 0.7;
- CPM-Leiden at resolution 0.5.

Effect on Feb 6–8: multi-incident families 19,274 → 9,375; largest family 24.3k →
18.1k docs and now coherent. Top three families by size:

| family | macro-events | content (sampled titles) |
|---|---|---|
| Turkey–Syria earthquake | 203 (16,444 docs, 128 publisher countries) | 25/25 random macro-event titles on-topic, ~30 languages |
| Chinese spy balloon | 58 | all on-topic |
| Biden State of the Union | 48 | all on-topic |

### 3.4 What is and is not claimed

- Claimed: on first-pass human labels the served model has family pair precision
  0.67 / F1 0.67 and incident pair precision 0.53; family precision within the
  five labelled neighbourhoods is 0.92–1.00; the five flagship families plus the
  Ohio control read as single stories in random and lowest-confidence samples;
  URL-only/entity-only cross-incident glue is gone (quake 3/0, others 0/0);
  11.5 % of documents are deliberately left unassigned.
- Not claimed: universal optimality of the settings (the gate/Leiden/family
  sweep stages were not exhausted); that recall is good (family pair recall
  0.67, B³ 0.585 — big stories are fragmented into hundreds of incidents and
  several families); performance on pre-2020 windows beyond the metadata
  fallback; that the labels are final (single first-pass labeller; 150 hardest
  pairs exported to `eval/review_20230206.csv` for a second human review, not
  yet done); that `assignment_score` is a probability (it is normalised edge
  mass).

### 3.5 Measured clustering quality (Feb 6–8 2023, first-pass labels)

Labels: `eval/pairs_20230206.jsonl` (300 hard pairs: 67 same_event, 31 related,
202 different; six strata) and `eval/neighborhoods_20230206.jsonl` (300 docs,
60 each for Turkey quake, Chinese balloon, State of the Union, Grammys,
Zelensky visit). "related" counts as *different* at incident level and *same*
at family level.

| run | level | pair P | pair R | pair F1 | B³ F1 (neigh.) | CEAF-e F1 | unassigned |
|---|---|---|---|---|---|---|---|
| `_fam` (demo store) | incident | 0.320 | 0.463 | 0.378 | 0.394 | 0.265 | 8.1 % |
| `_fam` (demo store) | family   | 0.497 | 0.796 | 0.612 | 0.647 | 0.208 | |
| `_v3` (stricter single-channel gate) | incident | 0.449 | 0.522 | **0.483** | 0.397 | 0.270 | 9.2 % |
| `_v3` (stricter single-channel gate) | family   | 0.579 | 0.714 | **0.639** | 0.590 | 0.202 | |
| `_v5` (`_v3` + cross-language lone-title floor 0.60, deterministic rerun) | incident | 0.528 | 0.418 | 0.467 | 0.402 | 0.273 | 11.5 % |
| `_v5` (`_v3` + cross-language lone-title floor 0.60, deterministic rerun) | family   | **0.681** | 0.653 | **0.667** | 0.574 | 0.226 | |

`_v5` is `eval/metrics_20230206_v5.json` on the deterministic rerun. The
pre-fix `_v5` run reported family P/R/F1/B³ 0.667/0.673/0.670/0.585 with
identical settings — that gap is the family-stage drift described in §7 and no
longer occurs.

Reading: `_v3` trades ~8 pts of family recall on the five big neighbourhoods
(B³ 0.647→0.590) for +13 pts incident precision and +8 pts family precision on
the hard pairs; the `merged_cross_language` stratum (the weakest) moves from
P=0.24 to 0.32 at incident level. Neither run is good enough to oversell; a
gate/Leiden/family sweep against the same labels is in `data/sweeps/`.

**Candidate retrieval (independent of the labelled pairs).** Over the five
neighbourhoods' regex-matched documents (`eval/candidate_recall_20230206.json`):
98.9 % of labelled documents have at least one candidate partner, 99.6 % of gold
documents fall in the largest connected component of the induced candidate
graph (99.4 % of all regex documents). All-pairs candidate recall is 0.3–3 % by
construction (top-30 retrieval over stories with 7k–36k documents); reach and
connectivity are the quantities that bound Leiden recall, not pairwise recall.

**Explainability caught a real error.** The "why is this article here?" panel on
the `_fam` store showed a Thai K-POP audition article inside the quake family,
held there by three URL-only edges (`url=1.00`, one channel, `title=0.00`) to
boilerplate-titled `infotag.md` pages ("Infotag"). In `_v3`
(`single_channel_min_features=2`) those edges are gated out and the article
lands in a 9-document K-pop family — the intended behaviour, and a concrete
example of the gate change.

**Per-neighbourhood family metrics** (`attention.evaluate`, majority family =
the run's family holding most of a story's gold docs): family precision within
the sample is 0.92–1.00 for both runs on all five stories; `_v3` lowers
within-sample recall on Grammys (0.66→0.50), quake (0.67→0.60) and Zelensky
(0.65→0.58) — i.e. the B³ drop is fragmentation, not new false merges.

**Flagship audit** (`scripts/inspect_flagship.py` → `eval/flagship_20230206_{fam,v3}.md`;
random 25, lowest-score 25, bridge documents, single-channel cross-incident
edges): in `_fam` the quake family had 52 URL-only and 151 entity-only
cross-incident edges, SOTU 65 entity-only; in `_v3` these are 3/0 and 0/0.
Quake `_v3`: 18,147 docs, 308 incidents, 144 publisher countries, 56 languages;
all 25 lowest-score docs are on-topic; the remaining bridge documents are
title-less/boilerplate pages (`tap.info.tn`, `primeiraedicao.com.br`) with
cross-incident degree ≤ 18.

**Bridge-trust refinement (→ `_v5`).** The `_v3` audit's remaining errors were
weak *bridge* nodes, not URL/entity glue: in the Ohio-derailment control an
Indian Railways hygiene story and a SEPTA derailment sat in the family, joined by
lone title edges scoring 0.55–0.60 *across languages* (the multilingual encoder
rates topical kin — "train", "derailment" — that high). On the labelled pairs,
cross-language title-only gated edges were 49 different vs 20 same_event, 33 vs
6 of them in the 0.55–0.60 bin. Two targeted gate rules were added
(`ClusterSettings.cross_language_title_floor`, `titleless_min_channels`) plus a
`bridge_risk` ranking in the audit
(`1·titleless + 1·generic_title + 0.25·log(1+cross_degree) − mean title − event
share`; heuristic, untuned). Measured (adaptive sweep, `data/sweeps/20230206_v3/bridge`):

| config | fam P | fam F1 | fam B³ | inc P | unassigned |
|---|---|---|---|---|---|
| `_v3` (floor 0.55 = off) | 0.575 | 0.633 | 0.590 | 0.449 | 9.2 % |
| floor 0.60 (**`_v5`**) | 0.667–0.691¹ (0.681 deterministic) | 0.670–0.677 (0.667) | 0.585–0.591 (0.574) | 0.528 | 11.5 % |
| floor 0.65 | 0.690 | 0.649 | 0.589 | 0.591 | 12.8 % ✗ |
| floor 0.60 + title-less pairs need 2 channels (`_v4`) | 0.656 | 0.649 | 0.578 | 0.551 | 12.0 % ✗ |

¹ the family stage showed run-to-run variation before the determinism fix
(sweep run vs full `_v5` run with identical settings; incident level was always
identical). The fix (§7) makes the stage reproducible; the bracketed values are
the deterministic rerun and are the numbers to quote. Higher floors
were not run: 0.65 already breaches the constraint. The title-less rule removes
the `Primeira Edição`/untitled bridges but measured worse than the language
floor alone, so it ships as a knob (default 1). The 2.2 M cross-language
title-only edges (16 % of all gated edges) removed by the floor cost recall
(fam R 0.714→0.673) but on the flagship stories the top-family share of
regex-matched docs is unchanged (quake 0.283→0.286, balloon 0.573→0.584); the
Ohio family loses both false positives (in the pre-fix run it split 870→678+498;
in the deterministic rerun it is one 938-document family of 8 incidents). Flagship
audit on the deterministic rerun: `eval/flagship_20230206_v5.md`
(URL-only/entity-only cross-incident edges: quake 3/0, all others 0/0; all 50
sampled Ohio docs on-topic; top `bridge_risk` docs are high-degree specific
syndicated stories, i.e. useful bridges).

**Model selection (decided 2026-09-20).** The demo serves `_v5`
(`data/store/20230206_v5`; `_v3` and `_fam` kept for rollback). Selection is
constrained optimisation, not max F1 or min unassigned:
`max 0.45·P_fam + 0.25·F1_fam + 0.15·B³_fam + 0.15·F1_inc` subject to
unassigned ≤ 12 % and a manual flagship audit showing no catastrophic merge
(`scripts/eval_sweep.py`, `scripts/rank_sweep.py`). Design principle: *Ripple
prefers abstention over false certainty* — 11.5 % of documents are left
unassigned rather than forced into an event; a false merge is visible in the
globe and in the evidence panel, a split costs some recall. The 18-config gate
sweep and the Leiden/family stages were not exhausted (≈13 min/config); the
bridge stage was run adaptively and stopped once the constraint boundary was found.

---

## 4. Materialised store (`attention/materialize.py`)

Served store: `data/store/20230206_v5` (`_v3`, `_fam`, `_cal`, base kept on disk
for rollback/comparison; `20190415` is the legacy-model day).

```
                          _fam        _v3         _v5 (served)
macro_events               5,072       5,032       4,518   (incidents with ≥30 docs, ≥5 effective reports, country conf ≥0.5)
documents_in_macro_events 327,983     323,738     295,787
event_families (served)    3,205       –       2,895
incidents_total          117,214     118,328     120,026
families_total           101,626     102,166     103,076
unassigned_rate            8.1 %       9.2 %      11.5 %
country_baseline             199 countries (194 with centroid lat/lon; names from the GDELT lookup)
sources                   14,635 domains
```

Tables (all Parquet + `meta.json`):

- `macro_events`: `macro_event_id, incident_id, family_id, title, label, start/end_time, event_country, lat, lon, event_types[], actors, people, organizations, themes, atomic_event_count, raw_documents, unique_domains, effective_reports, publisher_country_count, language_count, entity_coherence, title_coherence, cluster_confidence, resolution_model`.
- `event_families`: one row per family with a macro-event: title (medoid), macro-event list, doc/country/language counts.
- `macro_event_documents`: one row per (document, macro-event): `observed_time, source_domain, publisher_country, publisher_country_confidence, mapping_method, source_language, title, assignment_score, is_primary`.
- `country_event_attention` (hourly) / `country_event_summary` (country × incident) and, since the family-first change, `country_family_attention` / `country_family_summary` (country × family) with the same columns, so lag and attention ratio exist at both levels:

  ```
  raw_documents        = canonical URLs
  unique_domains       = distinct outlets
  effective_reports    = Σ_g min(1, n_g)  over wire groups
  raw_share            = raw_documents(c,e) / raw_documents(c, window)
  effective_share      = effective_reports(c,e) / effective_reports(c, window)
  attention_ratio      = effective_share(c,e) / effective_share(world,e)
  first_seen, third_source_seen, p10_seen
  onset                = max(third_source_seen, p10_seen)
  lag_hours            = onset(country) − onset(world)      (null + suppressed if <3 outlets)
  ```
- `document_evidence`: per document the top-3 gated same-incident edges and the
  top-1 edge into another incident (`title_score, event_score, url_score,
  entity_score, delta_hours, evidence_channels, gated`) — the lineage the
  explainability endpoint reads; channels absent from a run are filled with 0.
- `sources`: domain → `publisher_country, confidence, mapping_method`.
- `meta.json`: window, seed, model, all cluster settings (`cluster_run`),
  denominators, filters, and the caveats every API response repeats (observed
  time ≠ publication time; publisher country ≠ event geography; event types are
  provisional rule-based labels).

Served-store flagship families (`eval/flagship_20230206_v5.md`):

| family | docs | incidents | publisher countries | languages | URL-only / entity-only cross-incident edges |
|---|---|---|---|---|---|
| Turkey–Syria earthquake (`5928`) | 17,204 (15,834 in served macro-events) | 291 (195) | 143 | 55 | 3 / 0 |
| Chinese balloon (`19`) | 5,239 | 126 | 108 | 50 | 0 / 0 |
| State of the Union (`4467`) | 3,982 | 59 | 78 | 28 | 0 / 0 |
| Grammys (`144`) | 2,828 | 37 | 93 | 37 | 0 / 0 |
| Ohio derailment (`5792`, control) | 938 | 8 | 28 | 12 | 0 / 0 |

Document counts are whole-family (all incidents); the API and globe surface only
incidents retained as macro-events (≥ 30 docs, ≥ 5 effective reports), hence the
smaller served figure for the earthquake.

---

## 4b. Country response analytics (`attention/analytics.py`)

Built strictly on the store above — no change to clustering, onset or the
attention tables. `scripts/run_window.sh` runs `python -m attention.analytics
data/store/<tag>` after materialisation and writes
`country_response_observations.parquet`; the API derives it on first use if the
file is missing.

**Observation unit.** One row per `level × event × origin_country ×
destination_country`, at two levels: **family** (default — one story family is
one analytical observation, so the earthquake cannot contribute hundreds of
correlated incident rows) and **incident** (drilldown). The origin country $O$ of
an event is the `event_country` carrying the most effective reports across its
incidents (`origin_share` is recorded; 98 % of families with an origin exceed
0.5). Events with no `event_country` (680 of 4,518 macro-events) have no origin
and are excluded from analytics, not silently folded in. Every eligible event is
crossed with **all 199 baseline publisher countries**, so a country that never
reaches onset stays in the denominator as an uncovered, right-censored row
(`covered = false`, `censor_hours` = hours from reference to window end).

```
response_hours_origin  = onset(E, C) − onset(E, O)          foreign, reference = "origin"
response_hours_world   = onset(E, C) − onset(E, world)      used when O never reached onset,
                                                            reference = "world_fallback"
self_response_hours    = onset(E, O) − event_start(E)       domestic, reference = "event_start"
covered                = destination has a valid, non-suppressed onset
onset                  = max(third_source_seen, p10_seen)   unchanged from §4
```

Domestic response is measured against the event's observed start, never as
`onset − onset` (which would be identically zero), and is labelled *observed
domestic response*: `event_start` is GDELT's first observation, not the
physical occurrence time. Negative foreign values are kept — they mean the
destination's press reached onset before the origin's (or before the world) —
and are a real finding on this window (e.g. Japan → US, Brazil → US).

**Statistics** (`summarize`, `pair_matrix`). Per destination country, per
directed pair, per matrix cell and per breakdown row:

```
coverage_rate        = covered_events / eligible_events, Wilson 95 % interval
mean / median / p25 / p75 response hours   conditional on coverage
latency_ci_low/high  = bootstrap of the median, seed 2026, 1,000 resamples (deterministic)
support              = "ok" iff covered_events ≥ 5 and Σ effective_reports ≥ 15 (configurable);
                       otherwise counts and coverage stay visible, latency fields are null
```

Latency and coverage are never collapsed into one score: fast-but-rare coverage
is visible as such. Filters (`Filters`) AND together: level, event types,
`min_event_effective_reports` (inclusive) / `max_event_effective_reports`
(exclusive), date range, resolution model, reference policy
(`origin_preferred` / `origin_only` / `world`), origin and destination sets.
Breakdowns by origin country, event type and event-size bin
(`5–19 / 20–99 / 100–499 / ≥500 effective reports`) carry the exact filter
bounds that reproduce them, so a drilldown returns precisely the events that
were aggregated (asserted by `tests/test_analytics.py`).

**Measured on the served `_v5` store (family level).**

```
observation rows            1,248,128 (both levels); 484,366 family-level
eligible events with origin 2,434 families × 199 destinations
domestic rows               2,430 eligible, 1,638 covered
foreign rows                481,936 eligible, 7,848 covered (5,257 origin-relative, 2,591 world fallback)
negative foreign responses  1,356 of 7,848 covered
```

| query | covered / eligible | median | mean |
|---|---|---|---|
| Germany ← foreign (all origins) | 361 / 2,369 | 7.0 h | 12.0 h |
| France ← foreign | 260 / 2,371 | 7.0 h | 11.2 h |
| Brazil ← foreign | 224 / 2,393 | 11.0 h | 15.7 h |
| Japan ← foreign | 37 / 2,410 | 10.0 h | 15.0 h |
| Germany domestic (observed) | 46 / 62 | 6.4 h | 10.8 h |
| France → Germany | 14 / 63 | 4.0 h | 8.1 h |
| Germany → France | 9 / 65 | 13.0 h | 17.2 h |
| Turkey → Germany (`origin_preferred`) | 36 / 160 | 19.0 h | 16.3 h |
| Turkey → Germany (`origin_only`) | 19 / 63 | 2.0 h | 9.9 h |
| Turkey → Germany (`world`) | 36 / 160 | 21.5 h | 18.8 h |

Default reference policy is `origin_preferred` (origin-relative where the
origin reached onset, world fallback otherwise). The Turkey rows show why the
policy is exposed and every observation is flagged: restricted to the 63
Turkish-origin families where Turkish outlets themselves reached onset, German
onset follows by a median 2.0 h; measured against world onset over all 160 the
median is 21.5 h, and the mixed default lands at 19.0 h.

The two directions of a pair are separate measurements and differ here by a
factor of three. On a single 3-day window most pair cells are thin: with the
default gate, 9 of the 16 cells of the Germany/France/Japan/Brazil matrix show
*insufficient support* (counts shown, latency withheld), and roughly a third of
covered foreign observations rely on the world fallback. Adding a second window
(Beirut) is the direct fix; the gate exists so that sparse cells are never
shown as authoritative.

**Terminology** (repeated in every response's `caveats`): *observed media
response*, *observed response latency*, *publisher-country coverage*. Not
"Germany learned about the event after N hours": publisher country ≠ audience,
publisher country ≠ event location, GDELT observation time ≠ publication time,
correlation ≠ causal transmission.

---

## 5. API (`api/attention.py`, FastAPI, prefix `/api/v2/attention`)

| route | purpose |
|---|---|
| `GET /search?q=` | NFKD-folded, casefolded search over family/event title, label, entities; whole-word for alphabetic scripts, substring for CJK. **Family-first**: returns story families ranked title hits → publisher-country count → effective reports, each with its incidents ("explore incidents"). `turkey earthquake` → family 5928; `ohio derailment` → 5792; `Erdbeben`, `地震`, `spy balloon`, `Grammy`, `State of the Union`, `ChatGPT` return the expected stories. |
| `GET /events`, `/events/{id}` | macro-event (incident) list / detail |
| `GET /events/{id}/spread?include_family=&min_country_confidence=&limit=&offset=` | chronological documents with publisher country; stable `(observed_time, document_id)` pagination; `excluded_documents` count; per-country summary |
| `GET /events/{id}/timeline`, `/events/{id}/countries` | hourly country attention; per-country onset / lag / ratio |
| `GET /families/{id}`, `/families/{id}/spread`, `/families/{id}/timeline`, `/families/{id}/countries` | the same three views at story-family level (family-level lag/ratio from `country_family_*`) |
| `GET /event-types`, `/event-types/{t}/countries` | type-level roll-ups (min 3 events, min 1,000 country docs before ranking) |
| `GET /documents/{id}/evidence` | "why is this article here?": document, incident, family, `assignment_score`, `checks` (`title_similarity`, `shared_gdelt_event`, `shared_url_tokens`, `shared_entities`, `hours_to_nearest_support`, `other_publisher_country`), `supporting[]` (same-incident edges), `competing[]` (best edge into another incident), score caveat; 404 unknown doc, 503 if the store predates the table |
| `GET /countries` | baseline per publisher country incl. name, lat, lon |

Country analytics (`api/analytics.py`, prefix `/api/v2/attention/analytics`;
all accept the §4b filters and support minimums, and add `filters`, `support`,
`caveats` to the envelope):

| route | purpose |
|---|---|
| `GET /countries` | every publisher country: foreign response (mean, median, P25/P75, bootstrap CI), coverage (Wilson CI), observed domestic response, N eligible / covered |
| `GET /countries/{c}` | one country's overview plus breakdowns by origin country (top-N), event type, event-size bin and domestic event type, each row carrying its exact drilldown filter |
| `GET /pairs?origin=FR&destination=GM` | `FR → GM` and `GM → FR` as separate blocks (stats, fastest/slowest, by event type) with contributing events |
| `GET /matrix?origin=…&destination=…` | origin × destination cells (diagonal = observed domestic response) for any origin / destination sets, plus per-destination rows |
| `GET /origins/{c}`, `/destinations/{c}` | how every destination responds to events in `c`; who `c` responds to, per origin |
| `GET /events?origin=&destination=&kind=&…&limit=&offset=` | the event-level observations behind any number: event, date, types, origin onset, destination onset, response hours, reference, effective reports, attention ratio |

Every response carries `meta`. No response has a bare `country` field — only
`publisher_country` / `event_country`. The older `/api/v2/{doc,geo,ext}` routes
from PR #4 remain mounted.

Run: `GDELT_ATTENTION_DATA=data/store/20230206_v5 uv run uvicorn api.app:app --port 8000`.

---

## 6. Frontend (`web/spread.html`, `web/src/spread/`)

Vite + React 19 + TypeScript + Three.js + React Three Fiber (+ drei). The D3
canvas globe was replaced by `RippleGlobe` (`web/src/spread/globe/`).

**Globe** (`Earth.tsx`, `Sky.tsx`, `Markers.tsx`, `config.ts`, `geo.ts`):
textured Earth (`earth_atmos/normal/specular_2048`, byte-identical to three.js
example assets, MIT — recorded in `web/README.md`), normal-map relief with
tunable exaggeration, ocean specular, optional 1024 cloud shell, custom rim-glow
atmosphere shader with per-background parameters, `OrbitControls` (damped, zoom
bounded, auto-rotate after idle, fly-to-origin on selection). Dark mode adds a
drei starfield, a sun disc and a moon; in both modes the key light follows the
sub-solar point of the playhead's UTC time (`geo.subsolarPoint`, `geo.moonPoint`
— low-order approximations, illumination-accurate, not an ephemeris; legend says
"daylight follows the clock (UTC)").

**Data layer**: blue **event-location** beacon (three expanding rings at the
GDELT lat/lon); one marker per **publisher country** at the `country_baseline`
centroid, grey → red when `now ≥ first observation`, size `1 + 0.16·log₂(articles
so far)`, activation ring, great-circle arc from origin fading over ~4 s (legend:
"arcs show attention order, not transmission"). All per-frame animation goes
through refs; no React state in `useFrame`. Tooltips are DOM, labelled
"publisher country (outlet base)" vs "event location". `GlobeConfig` exposes
`background, textureQuality, terrainExaggeration, atmosphereIntensity, clouds,
markerSize, arcs, rippleStrength, animationSeconds, autoRotate, realSun, camera,
colors`.

**Page** (`App.tsx`, `Evidence.tsx`, `api.ts`): "Show the spread of ___" →
`/search` (family-first) → auto-select top hit → `/families/{id}/spread` paged →
timeline (play/pause/replay, scrubber, speed 0.5–4×, UTC clock, 40 s span, order
`(observed_time, document_id)`); scope toggle **whole story** (default) / **this
incident only**; side panel with title, types, start, global onset, articles,
countries, incidents, top-countries table (first seen, lag, articles, effective
reports, ratio — now at both scopes) whose rows turn red as their country
activates; **Earliest articles** list → click → `EvidencePanel` ("Why is this
article here?") rendering `/documents/{id}/evidence`. Explicit states: loading,
API error / 502 recovery, no match, countries without centroid, hidden
low-confidence article count, "observed time ≠ publication time" caveat; dark
mode and arcs toggles.

**Country Stats tab** (`CountryStats.tsx`; top-level `Events | Country Stats`
switch, Events state preserved). Three modes over the §4b routes:

1. **Country overview** — foreign response (mean, median, coverage with CI,
   eligible / covered) beside *observed domestic response*; breakdown tables by
   origin country, event type, event size and domestic event type. Clicking any
   row lists exactly the story families aggregated in it (same row collapses,
   another row replaces).
2. **Country ↔ Country** — `A → B` and `B → A` side by side (they are separate
   measurements), fastest/slowest event, by-type breakdown, contributing events,
   and a one-click *View B → A* swap.
3. **Multi-country comparison** — checkbox/search selection (Germany, France,
   Japan, Brazil, … preset), sortable destination table, and an origin ×
   destination heatmap switchable between median, mean, coverage and event count;
   diagonal cells show domestic response, unsupported cells are muted with their
   N; tooltips carry mean/median/coverage/covered ÷ eligible/95 % CI; clicking a
   cell opens the events behind it.

Every event in a drilldown opens on the globe in Events mode. In Country Stats
the globe highlights the selected publisher country(ies) and the event-origin
countries with supported estimates; the legend states that arcs are observed
media-attention relationships, not transmission. Dark mode keeps active controls
readable (fixed in #9).

Run: `cd web && npm install && npm run dev` → `http://localhost:5173/spread.html`
(proxies `/api` to port 8000). `npx tsc -b && npm run lint && npm run build`
clean (one pre-existing warning in the old explorer, one chunk-size notice).

---

## 7. Verification

Backend: `uv run pytest -q tests` → **106 passed** (17 dependency-deprecation
warnings from FastAPI/Starlette/matplotlib); `uv run ruff check`, `ruff format
--check`, `uv run ty check attention api scripts` clean. No CI on the repository;
all checks are local and re-run before each push.

**Reproducibility.** Before the fix, identical `_v5` settings produced different
family assignments between runs (family P 0.667 vs 0.691) while incidents were
identical. Cause: unordered structures feeding the family stage — `set`/dict
iteration in `incident_top_entities`/`incident_top_places` tie-breaking, the
edge dict order handed to igraph in `link_families`, and unsorted group-by
output when writing `incident_memberships`/`incidents`. Fix: top-k ties are
totally ordered, `incident_leiden` sorts edges before building the graph,
memberships are written sorted by `(document_id, incident_id)` and incidents by
`(documents desc, incident_id)`. Verification on real data: two independent
re-runs from the same cached pair features (`data/clusters/20230206_v5_run3`,
`_run4`) are **byte-identical** on `candidate_pairs`, `pair_features`,
`graph_edges`, `title_background`, `incident_memberships` and `incidents`, with
identical audit counts (120,026 incidents, 103,076 families, unassigned
0.11459); only `runtime_seconds`, output paths and the command line in
`run.json` differ. `tests/test_attention.py` covers tie ordering and edge-order
invariance of `incident_leiden`; `tests/test_backend_pipeline.py::test_clustering_is_deterministic`
asserts byte identity of every cluster Parquet on the synthetic window.

- `tests/test_backend_pipeline.py` (28 tests) synthesises raw GDELT zips for a
  known world (40 quake docs in 6 languages / 7 publisher countries with a
  wire-copy pair and an exactly-two-outlet country, 15 Chile-wildfire docs as
  negative control, 6 noise singletons, one URL in both GKG families, a
  title-less doc, an unresolvable domain, a missing slot, a malformed row) and
  runs the whole pipeline + API twice (title model and legacy fallback):
  quake is one cross-lingual incident (recall/purity ≥ 0.9), wildfire separate,
  noise unassigned, determinism, title permutation destroys structure, temporal
  gate, wire copies collapse `effective_reports`, 2-outlet country suppressed,
  onset order TR ≤ US ≤ JP, unresolved docs retained but excluded from country
  tables, spread pagination stable, `min_country_confidence=0.99` → valid empty
  response, `/documents/{id}/evidence` shape, OpenAPI lists all routes.
- `tests/test_attention.py` unit-tests the gate (title veto, feature-count
  rule, title-less rule, cross-language floor), ccTLD→FIPS, URL canonicalisation,
  calibration, evaluation metrics, top-entity tie ordering, Leiden edge-order
  invariance, resumable embedding shards and the distinct-title mapping;
  `tests/test_api.py` the older routes.
- `tests/test_analytics.py` (14 tests) builds a synthetic world with known
  answers and asserts: A → B and B → A are separate and correct; A → A uses
  `event_start`, not `onset − onset`; mean/median/P25/P75; the multi-country
  matrix with multiple origins and destinations; domestic vs foreign
  eligibility; world fallback flagged; negative responses preserved; uncovered
  countries stay in the denominator with `censor_hours`; Wilson interval;
  bootstrap reproducibility under the fixed seed; support gating (counts kept,
  latency null); every filter incl. the exclusive magnitude ceiling; and that
  each event-size breakdown row's bounds reproduce exactly its own events.
  `tests/test_backend_pipeline.py` exercises the `/analytics/*` routes on the
  synthetic store.
- Real-data evaluation: §3.5 (labels, sweep, flagship audit, candidate reach).
- UI: a recorded browser test of the R3F globe (PR #5 comment) passed search →
  auto-select → fly-to, beacon, chronological markers + arcs, pause/scrub/speed/
  replay, tooltips, drag/zoom, dark mode, incident isolation, API outage → 502
  notice → recovery; it found two layout/empty-state bugs, both fixed. A second
  recorded pass on the deterministic `_v5` store covers the evidence panel, the
  sky layer (stars, sun/moon, UTC-driven terminator), terrain relief and the
  small-spread Ohio family; its screenshots are the README gallery
  (`docs/assets/`). A third recorded pass (PR #8/#9) covers Country Stats
  against the `_v5` API: Germany overview (361 / 2,369, median 7.0 h), France →
  Germany 4.0 h (14 / 63) vs Germany → France 13.0 h (9 / 65) with the swap
  button, the DE/FR/JP/BR matrix with metric switching and muted unsupported
  cells, matrix-cell / origin-row / event-type / event-size drilldowns whose
  headings match the aggregated denominators, event → globe navigation and
  dark-mode contrast. It found two issues (type/size rows did not drill; dark
  active tabs unreadable), fixed in #9. Not exercised: drilldowns beyond the
  500-row page.

---

## 8. Research grounding

Field semantics: GDELT 2.1 GKG codebook. GDELT duplication/coverage bias → dedupe
and per-country normalisation: Wang et al. 2016 (Science), Kwak & An 2014.
Near-duplicate/wire detection: Broder 1997. Streaming multilingual story
clustering features (TF-IDF + entities + time, rank-then-merge): Miranda et al.
EMNLP 2018. Feature fusion and CEAF-e/cluster-count evaluation: Saravanakumar et
al. EACL 2021. Two-level event→story hierarchy: Liu et al. Story Forest CIKM 2017.
Burst weighting (planned, not implemented): Kleinberg 2002. Multilingual sentence
embeddings: Reimers & Gurevych EMNLP 2020. Leiden: Traag et al. 2019. FAISS:
Johnson et al. 2017. B³: Bagga & Baldwin 1998; CEAF: Luo 2005. No paper
establishes this exact recipe as optimal; the design is justified by those
references plus our measured comparisons (entities-only .25 purity vs titles
.87/.98 on the teammate's benchmark; the calibration, family and gate changes
above, each measured on the same labels).

Open-source survey: no drop-in library fits (gdelt-pulse scrapes titles live and
is English-only greedy assignment; Priberam/Story Forest implementations need
full text or are not runnable on GDELT metadata).

---

## 9. Known limitations and open work

1. **One strong-model window is servable** (Feb 6–8 2023). Beirut Aug 2020 is
   preprocessed and embedded (EC2, 48 CPU cores, no GPU); clustering and
   materialisation were run remotely and the window is only servable once its
   API smoke checks pass — see the README "Data windows" table for the state at
   the time of the last commit. Ukraine Feb 2022 and Queen Sep 2022 are not
   fetched. No cross-window aggregate claim is made.
2. **Pre-2020 = metadata-only model**; cross-lingual quality there is poor.
3. **Labels are single-pass** (one labeller, 300 pairs + 300 neighbourhood
   docs). `eval/review_20230206.csv` (150 hardest pairs) awaits a second human.
   Quality numbers should be quoted as "first-pass labels".
4. **Recall / fragmentation**: the quake is 291 incidents and the regex-matched
   quake documents' top family holds ~29 % of them; family pair recall 0.65.
   The remaining sweep stages (Leiden resolution, family thresholds) target
   this and were not exhausted (~13 min/config on cached pair features).
5. **Family-stage nondeterminism — resolved** (§7): the deterministic rerun is
   the run whose numbers are quoted everywhere; earlier `_v5` figures
   (family P 0.667, quake family 18,089 docs / 317 incidents) come from the
   pre-fix run and are labelled as such where they still appear.
6. **Publisher-country enrichment** beyond the GDELT list (718 unresolved
   domains on Feb 2023, 1,333 on Aug 2020): 2021 outlet-geography file →
   Wikidata HQ → hreflang/ccTLD heuristics → cached LLM pass over the residue
   only. Planned, not done. Codes `PC`, `RB`, `CK` from the lookup need checking.
7. **Scale**: a 3-day window is ~1 h on one CPU box (embedding dominates);
   the multi-TB archive needs Tier-0 preprocessing on a fleet (design in
   `docs/architecture-plan.md`). The Feb 2023 window ran entirely on the
   8-core dev box; Beirut embedding/clustering ran on a 48-core CPU EC2 box.
8. **No CI** on the repository; all checks are local.
9. **Sun/moon** are illumination approximations, not an ephemeris; `bridge_risk`
   weights are heuristic; `assignment_score` is not calibrated.
10. **Family-linking sweep** (`scripts/family_sweep.py`, incident→family only,
    document gate fixed) was started but not exhausted (≈ 7.4 min/config on
    cached pair features); `_v5` family settings are therefore provisional.
11. **Country analytics are sample-limited by the single window.** Pair cells
    are thin (France → Germany N = 14, Germany → France N = 9), 9/16 cells of
    the DE/FR/JP/BR matrix fall below the support gate, and ~⅓ of covered
    foreign observations use the world fallback. Origin is the dominant
    `event_country`, so multi-country events are attributed to one origin.
    Not implemented (P1): origin/destination *region* filters (needs a
    FIPS → region table), a free-form event-magnitude filter beyond the exact
    size bins, and analytics-specific arcs on the globe.

## 10. Repository map

```
attention/preprocess.py     raw zips → typed Parquet + audit + sources (publisher-country ladder)
attention/atomic.py         documents, wire groups, doc↔event links
attention/embed.py          multilingual title vectors (cached), boilerplate-title filter
attention/cluster.py        channels, TitleChannel calibration, evidence gate (ClusterSettings), Leiden incidents, families, lineage, --reuse-pairs
attention/materialize.py    store tables incl. country_family_*, document_evidence, meta.json
attention/analytics.py      country_response_observations, Filters/Support, Wilson + seeded bootstrap, summaries, pair matrix, event records
attention/evaluate.py       pair P/R/F1, B³, CEAF-e, per-neighbourhood majority-family metrics
api/attention.py            FastAPI routes (family-first search, spread, evidence)
api/analytics.py            /analytics/{countries,pairs,matrix,origins,destinations,events}
tests/test_backend_pipeline.py, test_attention.py, test_analytics.py, test_api.py (+ older experiment tests)
web/spread.html, web/src/spread/{App,Evidence,CountryStats}.tsx, api.ts, spread.css, globe/{RippleGlobe,Earth,Sky,Markers,config,geo,types}
eval/pairs_20230206.jsonl, neighborhoods_20230206.jsonl, review_20230206.csv, metrics_*.json, candidate_recall_20230206.json, flagship_20230206_{fam,v3,v5}.md
scripts/fetch_window.py, run_window.sh, sample_eval_pairs.py, sample_eval_neighborhoods.py, export_review_csv.py,
        eval_sweep.py, rank_sweep.py, leiden_sweep.py, family_sweep.py, candidate_recall.py,
        inspect_flagship.py, inspect_incident.py, inspect_run.py
docs/project-report.md (this file), docs/architecture-plan.md, docs/article-graph-results.md, docs/prototypes.md, docs/assets/ (README screenshots), web/README.md
data/{raw,clean,features,embeddings,clusters,store,sweeps}/<window>[_variant]   (not in git)
```
