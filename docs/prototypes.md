# Earlier prototypes (March 2019 slice, April 2019 slice, v0)

Historical record of the exploratory work that preceded the `attention/` pipeline
described in the top-level README. Commands here still run, but none of this is
used by the served Ripple store.

## GDELT-shaped API

Build the tables once from downloaded raw files (see *Acquire data* below), then serve them:

```sh
uv run python build_api_dataset.py --mentions data/mentions --events data/events --output data/api
GDELT_API_DATA=data/api uv run uvicorn api.app:app --port 8000
```

| Endpoint | Shape |
|---|---|
| `GET /api/v2/doc/doc` | DOC 2.0: `mode=artlist\|timelinevol\|timelinevolraw\|timelinetone\|tonechart`, `sort`, `maxrecords`, `timespan`, `startdatetime`, `enddatetime` |
| `GET /api/v2/geo/geo` | GEO 2.0 GeoJSON `FeatureCollection` of `ActionGeo` points |
| `GET /api/v2/ext/events`, `/ext/events/{id}` | event rows and one event's coverage (not published by the public APIs) |
| `GET /api/v2/ext/facets`, `/ext/meta`, `/ext/health` | top domains/languages/themes, dataset provenance, liveness |

Query syntax is a subset of the DOC language: bare terms, `"quoted phrases"`, `-negation`,
`a OR b`, parenthesised OR groups, and `domain: domainis: sourcelang: sourcecountry: theme:
location: actor: quadclass:`. Unsupported operators (for example `tone>5`) return HTTP 400
instead of being silently dropped. Interactive OpenAPI docs are at `/docs`.

### What the archive can and cannot support

The live DOC API searches article **text** and returns publisher-supplied headlines and
social images. The open archive carries neither, so this backend is explicit about it and
`/api/v2/ext/meta` returns the per-field provenance it was built with:

- `title` is **derived from the URL slug**, not a headline.
- `sourcecountry` is **derived from the domain's ccTLD**, blank for `.com`/`.org`/etc.
- `socialimage` and `url_mobile` are always empty.
- search matches the derived title, the domain and the CAMEO labels (actors, locations,
  action types) of the events an article mentions — not article body text.
- `seendate` is `MentionTimeDate`, GDELT's observation time, not publication time.
- Only `MentionType=1` (web) documents are served.

## Explorer front end

```sh
cd web && npm install && npm run dev   # proxies /api to http://127.0.0.1:8000
```

The dashboard drives the API above: query bar with the operator syntax, coverage-volume
timeline (click to zoom the window), tone timeline and tonechart histogram, a Leaflet map of
event locations, clickable facets that append operators to the query, a CAMEO event table and
a per-event drawer listing the articles that mention it. Set `VITE_API_BASE` to point the
build at a remote API.

## Acquire data on the EC2 instance

Read the [challenge README](https://voloridge-hack-mit-2026.s3.us-east-1.amazonaws.com/src/gdelt/README.md)
and use its supplied fetcher. No AWS credentials are required for public S3 data.
The following commands assume this checkout and `uv sync --locked` are ready.

```sh
mkdir -p src/gdelt
curl -fSL https://voloridge-hack-mit-2026.s3.us-east-1.amazonaws.com/src/gdelt/README.md \
  -o src/gdelt/README.md
curl -fSL https://voloridge-hack-mit-2026.s3.us-east-1.amazonaws.com/src/gdelt/fetch.py \
  -o src/gdelt/fetch.py

# Always list each table before downloading it.
uv run python src/gdelt/fetch.py --table mentions --start-date 20190310 --end-date 20190317 --list
uv run python src/gdelt/fetch.py --table events --start-date 20190310 --end-date 20190317 --list
uv run python src/gdelt/fetch.py --table mentions --start-date 20190310 --end-date 20190317 \
  --output-dir data/mentions
uv run python src/gdelt/fetch.py --table events --start-date 20190310 --end-date 20190317 \
  --output-dir data/events
```

Do not download GKG. The fetcher's default 5 GiB guard stays enabled. March 10–17
is **eight inclusive days**, not seven. The stale v2 archive covers 2015-02-18
through 2019-04-16.

The September 19, 2026 listing contained **767 objects per table** (883.6 MiB
Mentions, 502.4 MiB Events), rather than 768: `20190313030000` was missing in both.
Check the listing before accepting this omission. An explicit `--allow-partial`
is therefore needed for this known gap; the audit still enumerates every missing
file. The flag also permits larger gaps, so always inspect `audit.json`.

```sh
uv run python story_clusters.py \
  --mentions data/mentions --events data/events \
  --start-date 20190310 --end-date 20190317 --allow-partial \
  --output results/march10-17-v0
```

Use a fresh output directory for each run. If memory is insufficient, list/fetch
March 15–16 first and change both date arguments and the output directory. The
script holds the slice in memory. Its pair-contribution guard protects the sparse
projection; it does not guarantee the loader fits memory. Events are optional
labels, but a same-range Events slice will not label older event IDs first recorded
before that slice.

## Schema and interpretation

The [official GDELT v2 codebook](https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf)
confirms the 16 Mentions fields in `MENTION_COLUMNS`. Files are TSV with no header;
UTF-8 decoding replaces invalid bytes. The loader checks every row's width,
retains leading zeros in event codes, and records bytes, SHA-256, replacement
characters, row counts, null counts and null rates.

One correction to the challenge README: **`ActionGeo_FullName` is zero-based
column 52**, not 53 (which is its country code); `Actor2Geo_*` spans 43–50.
The full Events schema is encoded in `EVENT_COLUMNS` and checked with a location
versus country-code regression test.

`EventTimeDate` means when GDELT first recorded an event. `MentionTimeDate` is
GDELT's observation/update time. Neither is a reliable publication or occurrence
timestamp. Nonempty `MentionDocTranslationInfo` records machine translation; its
absence does not rule out human translation or prove the original was English.
The audit counts nonempty records, URLs and parsed source-language codes.

## Fixed baseline

1. Keep WEB (`MentionType=1`) HTTP(S) URLs. Compare each URL's complete, unique set
   of `(GlobalEventID, SentenceID, Actor1CharOffset, Actor2CharOffset,
   ActionCharOffset, MentionDocLen)` tuples. Identical sets are candidate wire
   copies, not proven duplicates. Require at least two distinct events and no
   missing fingerprint values; preserve each URL, hostname, time and translation
   observation.
2. Drop URLs with more than 25 events, rows with confidence below 40, and events
   present in at least 1,000 canonical articles over at least two observed days.
   Wire fingerprints are computed before these filters.
3. Project the sparse canonical-article/event matrix with 64-bit counts. Require
   at least two shared canonical articles and an event-first-seen gap of at most
   three days. Stop above 50 million candidate pair contributions.
4. Compare Jaccard thresholds 0.05/0.1/0.2 and count thresholds 2/3/5, using both
   connected components and weighted Louvain with seed 2026. Retain isolated
   events so fragmentation remains visible. These are fixed exploratory settings,
   not parameters selected for the four expected stories.
5. Expand wire groups back to observed URLs for validation. Each article can
   appear in several event communities. Do not sum cluster article totals as
   though they were disjoint, or sum syndication copies as independent evidence.

Settings are available as flags; use `uv run python story_clusters.py --help`.
The support default remains two (`--min-shared-articles 1` relaxes it).
Add `--methods components louvain leiden` to include weighted Leiden modularity;
both community algorithms use resolution one.

## Controlled article-graph experiment

After the same list/download steps above:

```sh
uv run python clustering_experiment.py \
  --mentions data/mentions --events data/events \
  --start-date 20190310 --end-date 20190317 --allow-partial \
  --output results/article-comparison
```

This runs 13 fixed comparisons on identical input and eligible wire groups:

- Event projection at Jaccard 0.1, shared-article support one/two, each with
  components, Louvain and Leiden.
- Article graph with event-ID-only or URL-word-only cosine at 0.3, using Leiden.
- Equal event/URL cosine average at 0.15/0.3/0.45 using Leiden, plus components
  and Louvain on the identical 0.3 graph.

Each channel uses binary features weighted by `log((N+1)/(df+1))+1`, normalized
per canonical article. URL words are unioned across a wire group's decoded paths;
hostnames and queries are excluded. Date directory segments, long numeric IDs,
short tokens and fixed stopwords are removed. This is lexical, not multilingual
semantic matching. A missing channel contributes zero, without renormalizing the
other channel. Events actor/location labels are inspection metadata only.

Candidate pairs are the symmetric union of each channel's top 50 neighbors,
then scored in both channels. Ties use article order. Features in more than 2,000
canonical articles are excluded before normalization/retrieval. Blocked sparse
multiplication avoids a complete article-by-article matrix, with explicit bounds
on feature pair contributions, block nonzeros and candidate pairs. These bounds
abort rather than silently truncate; frequency filtering and top-k retrieval
still lose possible matches. Defaults are exploratory, not fitted to story labels.

Article runs use the same confidence, roundup and recurring-event eligibility as
event runs. They have no time gate; event projection retains its three-day
event-first-seen guard. Active canonical articles receive one cluster, including
isolates; filtered URLs remain unassigned. Event-based memberships still overlap.
Large clusters and high URL-keyword scores do **not** demonstrate story accuracy:
URL-based models use related evidence for features and evaluation, making those
scores circular. Independent pair/cluster labels are needed for precision,
recall, B³, CEAF-e and retrieval recall; this experiment does not fabricate them.

Outputs reuse the baseline inspection sheets and add `comparison.csv`,
`article_pairs.parquet` (candidate union with channel scores), feature/candidate
audits and each article run's `canonical_clusters.parquet`. For article runs,
`event_clusters.parquet` is an overlapping event/cluster relation, not an event
partition. Singleton-only URL counts and canonical size quantiles help compare
fragmentation without counting wire copies as independent support.

For a reusable preparation checkpoint, add `--prepare-only`. Continue into a new
output directory with `--prepared results/prepared-input`; the input settings are
checked and provenance is inherited. Intermediate data and reports stay out of
Git. Use `--help` for resource bounds; do not interpret an aborted run as a result.

## v2 article graph

```sh
uv run python clustering_v2.py --prepared results/prepared-input \
  --output results/v2 --max-pair-contributions 3000000000 \
  --max-candidate-pairs 80000000
```

Runs the fixed v1 reference (combined 0.3 Leiden) next to: URL tokens cleaned of
hex/numeric junk, near-uniform-by-day tokens and per-host boilerplate; an Events
`actor:`/`geo:` label channel; an evidence gate (two channels agree or one is
>= 0.6) at 0.2/0.3/0.4; and a story-family layer linking incident clusters by
centroid cosine. The benchmark is 15 URL regexes (still proxies). Full-week
findings and the recommended configuration are in
`docs/article-graph-results.md`, section 3.

## One-day attention slice (April 15, 2019)

Raw Mentions + Events + GKG -> typed Parquet -> atomic events -> blocked
clustering -> macro-event / country-attention store -> `/api/v2/attention/*`.

```sh
for t in mentions events gkg; do
  uv run python src/gdelt/fetch.py --table $t --start-date 20190415 \
    --end-date 20190415 --output-dir data/day/$t
done
uv run python -m attention.preprocess --date 20190415 --mentions data/day/mentions \
  --events data/day/events --gkg data/day/gkg \
  --domain-lookup ~/gdelt-reference/domains_by_country.txt --output data/clean/20190415
uv run python -m attention.atomic --clean data/clean/20190415 --output data/features/20190415
uv run python -m attention.cluster --features data/features/20190415 --output results/day-clusters-1
uv run python -m attention.materialize --clean data/clean/20190415 \
  --features data/features/20190415 --clusters results/day-clusters-1 \
  --output data/store/20190415
GDELT_ATTENTION_DATA=data/store/20190415 uv run uvicorn api.app:app
```

Endpoints: `GET /api/v2/attention/events`, `/events/{id}`, `/events/{id}/timeline`,
`/events/{id}/countries`, `/event-types`, `/event-types/{type}/countries`,
`/countries`. Every response carries `meta` with denominators and semantics.

Caveats: `mention_time` is GDELT observation time, not publication time; onset
is observed media-attention onset (later of 3rd distinct outlet and the 10th
percentile of documents), not "when a country found out"; `event_types` are
provisional theme/CAMEO rules; country is publisher country from the GDELT
domain lookup (confidence and method in `sources`). The macro-events are a
filtered baseline (>= 50 documents, >= 10 effective sources, coherence >= 0.7),
not validated against manual labels - the largest cluster (Notre-Dame) still
contains unrelated documents.

## Outputs and validation

| File | Contents |
|---|---|
| `audit.json` | Input scope, gaps, versions, schema, hashes, nulls, translation metadata, filters and dedup counts |
| `articles.parquet` | URL-to-canonical mapping and retained source/time provenance |
| `report.md`, `sweep.json`, `story_scores.csv` | Measurements across all 12 runs |
| `cluster_sizes.svg` | Survival distributions of URLs per cluster |
| `<run>/event_clusters.parquet` | Every retained event's community |
| `<run>/article_clusters.parquet` | Overlapping URL/community memberships |
| `<run>/cluster_sizes.csv`, `metrics.json` | Event/article/domain counts, observation spans, size quantiles and separation |
| `<run>/keyword_distribution.csv` | Per-cluster keyword matches, recall fractions and keyword purity |
| `<run>/clusters.md` | Top 20 clusters plus keyword winners; tokens, actors, locations, up to 10 seeded sample URLs and nonmatching samples |

The expected-story regular expressions are `ethiopian|boeing|737`,
`christchurch|mosque`, `idai|cyclone|mozambique`, and `brexit`. They match decoded
URL **paths**, excluding hostnames and query strings. They are noisy proxies:
`cyclone` also matches unrelated US storms. Missing keywords lower measured
purity, but false-positive keywords can inflate it; this is **not a guaranteed
lower bound on true purity**.

Review the top 20 and all four winners by eye, record judgments separately, and
inspect nonmatching URLs before deciding go/no-go. Winner distinctness alone
cannot establish true story separation. Keyword distributions overlap; their
fractions need not sum to one. Numeric/opaque and non-English slugs are especially
poor labels.

## Execution status

The full 2019-03-10..17 slice (767 of 768 quarter-hours, 4.42M Mentions rows) was
run locally through the 13-run comparison matrix; the write-up and derived tables
are in `docs/article-graph-results.md` and `docs/article-graph-results/`. The EC2
run remains **pending working access**. The earlier local smoke run used only
`20190315120000.mentions.csv` and its Events file: 7,173 Mentions rows, 1,017 URLs,
107 candidate wire copies collapsed. This is one quarter-hour update, not the
requested week or a representative random sample.

At Jaccard 0.1/Louvain, it produced 2,006 clusters, 1,961 containing a single event;
the largest had 28 URLs (2.75%). Keyword winner recall ranged from 13.33% to 46.15%.
The purported Idai winner was actually US “bomb cyclone” coverage. These findings
justify further validation and expose failure modes; they do not validate the
approach for country-level attention analysis. No full-week conclusion is claimed.

## v0: can GDELT articles be clustered by event across languages?

Ran on the shared EC2 box (code in `~/global`, uv at `~/.local/bin/uv`). Raw GDELT zips live in `/dev/shm/gdelt_raw/<window>` (tmpfs, lost on reboot), filtered tables and embeddings in `v0/data/<window>/`. The window, event start, keyword pattern and outlet list are in `v0/config.py`.

```
export UV_CACHE_DIR=/dev/shm/uv-cache HF_HOME=/dev/shm/hf   # root disk is only 8 GB
uv sync
uv run python v0/download.py           # GKG + Mentions zips for the window, ~20 s
uv run python v0/extract.py            # -> articles.parquet (outlets from 5 countries, with titles)
uv run python v0/extract_mentions.py   # -> mentions.parquet (GDELT event ids per URL)
uv run python v0/embed.py              # -> title_embeddings.npy (multilingual mpnet, ~40 s on CPU)
uv run python v0/cluster.py --title-weight 1 --thresholds 0.4,0.5,0.6 --detail-threshold 0.5 --detail-resolution 1.0
```

`--title-weight` blends entity TF-IDF cosine (0) with title-embedding cosine (1). `--entity-types` picks the entity token sources: P persons, O organizations, L locations, N all names, E GDELT event ids. Logs from the runs are in `v0/results/`: `cluster_*.log` are the 2019 Notre-Dame window (entities only, no titles in GKG before 2020), `results_2023_*.log` the 2023 Turkey earthquake window comparing entities, titles, and a blend.
