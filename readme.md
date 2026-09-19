# GDELT story clustering prototype

An auditable, metadata-only test of whether communities of co-mentioned GDELT event
IDs recover news stories. `story_clusters.py` compares connected components and
Louvain using Jaccard similarity and co-mention counts. It does not infer publisher
countries or publication times.

## Setup and checks

Use [uv](https://docs.astral.sh/uv/) with the committed Python version and lockfile:

```sh
curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

There are no pre-commit hooks. The script is the entrypoint; there is no web server
or separate build. Local tests include a complete CLI run, an empty filtered graph,
wire-fingerprint safeguards, sparse integer counts, and missing-file detection.

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
