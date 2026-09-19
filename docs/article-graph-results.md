# Full-week experiment: support threshold vs. article graph

Input: GDELT Mentions + Events, 2019-03-10..17 inclusive (767 of 768 quarter-hours;
`20190313030000` missing upstream). 4,421,084 mention rows, 684,956 URLs,
522,661 canonical articles after wire dedup (162,295 URLs collapsed), 475,501
canonical articles retained after the confidence/roundup filters. All 13 runs
share this input. Raw outputs live in `results/week-comparison-2` (gitignored,
~1 GB); the small derived tables are in `docs/article-graph-results/`.

Every quality number below is a URL-keyword proxy. It is a lower bound for
recall/purity and it is **circular** for the runs that use URL words as
features. No independent same-story labels exist yet, so candidate recall,
pairwise precision/recall, B³ and CEAF-e are all unmeasured.

## 1. Support 1 vs. support 2 (event-node projection, Jaccard >= 0.1)

| run | clusters | singleton clusters | largest cluster (% of URLs) | URLs in >1 cluster | canonical size p50 / p90 / p99 |
|---|---|---|---|---|---|
| support 2, Leiden | 579,200 | 87.7% | 3,177 (0.46%) | 232,085 | 1 / 2 / 12 |
| support 1, Leiden | 178,056 | 75.3% | 3,932 (0.57%) | 83,870 | 1 / 3 / 34 |
| support 1, components | 177,691 | 75.5% | 146,033 (21.3%) | 73,392 | 1 / 3 / 29 |

Louvain and Leiden give the same partition to within a few clusters in every
case. Relaxing support to 1 removes 400k clusters, but the median cluster is
still one canonical article and 20% of assigned URLs sit only in
single-article clusters. The extra edges also create a 21% giant component
that only modularity optimisation breaks up.

Manual URL review of the 20 largest clusters (`manual_review.json`):
support-1 Leiden 17/20 mixed, 2 roundup-contaminated, 1 family-compatible;
support-2 Leiden 13/20 mixed, 7 family-compatible. Typical false merges:
Christchurch + Trump border veto; Boeing grounding + college-admissions
charges; Brexit + Manafort sentencing; Pell sentencing + a rooster ban.
The support-2 clusters are purer but each expected story is scattered over
thousands of clusters (Christchurch URLs land in 9,611 clusters at support
2, 2,091 at support 1; the best cluster holds 7-11% of its keyword URLs).

Conclusion: the support cutoff is not the cause of fragmentation. Exact
GlobalEventID co-mention is too sparse to connect same-story articles and,
where it does connect them, it connects unrelated same-day politics too.

## 2. Article graph (canonical articles as nodes; IDF-cosine on event IDs and URL words)

Retrieval: 7.99M directed event-channel neighbours + 22.7M URL-channel
neighbours -> 21,870,583 candidate pairs (16.6M have URL evidence only). The
URL channel needed 723,230,788 feature-pair contributions, above the 200M
default guard; the run used `--max-pair-contributions 1000000000`. 252 URL
tokens appearing in >2,000 articles were dropped. Resumed article stage:
408 s, peak RSS 6.8 GiB.

| run | clusters | singleton clusters | largest cluster (% of URLs) | canonical p90 / p99 |
|---|---|---|---|---|
| event-only cosine 0.3, Leiden | 171,406 | 78.4% | 6,076 (0.89%) | 3 / 14 |
| URL-only cosine 0.3, Leiden | 48,494 | 91.2% | 26,987 (3.9%) | 1 / 5 |
| combined 0.15, Leiden | 20,201 | 88.7% | 33,635 (4.9%) | 2 / 73 |
| **combined 0.30, Leiden** | 152,807 | 79.5% | 6,854 (1.0%) | 2 / 12 |
| combined 0.45, Leiden | 228,990 | 81.1% | 3,512 (0.51%) | 2 / 14 |
| combined 0.30, Louvain | 152,861 | 79.5% | 8,318 (1.2%) | 2 / 12 |
| combined 0.30, components | 152,338 | 79.7% | 337,292 (49.2%) | 2 / 9 |

31% of the combined-0.3 edges (1,417,459 of 4,578,781) join articles with no
shared event ID at all: the URL channel supplies connectivity that the
projection cannot. Without modularity optimisation the same edges form a
49% giant component; Louvain and Leiden again agree closely.

### Expected stories, URL-keyword proxy (best single cluster)

| story (URLs matching) | support-2 Leiden | support-1 Leiden | combined 0.3 Leiden | combined 0.15 Leiden |
|---|---|---|---|---|
| crash/grounding (29,411) | 7.4% recall, 0.84 purity | 7.7%, 0.78 | 15.0%, 0.77 | 38.3%, 0.44 |
| christchurch (14,239) | 10.7%, 0.70 | 11.3%, 0.41 | 27.6%, 0.57 | 43.4%, 0.31 |
| brexit (16,961) | 8.3%, 0.65 | 9.7%, 0.50 | 29.6%, 0.74 | 60.6%, 0.43 |
| idai, broad regex (2,283) | 14.9%, 1.00 | 14.9%, 1.00 | 21.2%, 0.94 | 41.9%, 0.19 |
| **idai, `\bidai\b` only (339)** | 22.1%, 0.84 | 22.1%, 0.84 | **76.4%, 0.50** | 91.4%, 0.29 |

The brief's Idai regex (`idai|cyclone|mozambique`) is dominated by the US
"bomb cyclone" wire story (one event, 341 URLs) - the 1.00-purity event
clusters are that story, not Idai. With the stricter token, the combined-0.3
article graph puts 259 of 339 Idai URLs into one 513-URL cluster (cluster
234, top tokens cyclone/zimbabwe/mozambique/idai/malawi, ActionGeo Beira and
Chimanimani). Only 30 of its 513 URLs fail even the broad cyclone regex, and
the sampled ones (BBC world-africa, Reuters idAFKCN1QY0TK, AllAfrica) are
opaque IDs rather than obvious outsiders, so the strict-token purity of 0.50
understates it.
This is the first configuration in which the natural-disaster story is
recovered as a cluster; the event projection never gets above 22%.

Separation holds in all Leiden/Louvain runs: the four best-matching clusters
are distinct (`best_clusters_all_distinct` is false only for the two giant
component runs). At 0.3 the crash and the grounding are partly separated -
cluster 12 (grounding/FAA/bans, 5,696 URLs), cluster 27 (Ethiopian crash,
survivors, victims, 4,019 URLs) and cluster 29 (a mix of crash, black boxes
and early groundings, 4,300 URLs) - so the acceptable crash/grounding merge
happens only partially here; at 0.15 one 25,465-URL cluster holds both.

### Manual review of the combined-0.3 Leiden top 20

Reading `clusters.md` for the 20 largest clusters (URL/metadata only):
about ten are readable story or story-family clusters - Christchurch
(cluster 34, 4,518 canonical articles), Brexit (97; 3,992), 737 MAX grounding
(12; 3,740), Ethiopian crash (27), crash/grounding mix (29), Trump border
veto (15), Pelosi on impeachment (19), Iran/Iraq/Rouhani visit (467), and the
college-admissions charges split into Loughlin (109) and Huffman (81)
clusters. Their non-matching samples are mostly reactions or sub-stories
(Chelsea Clinton/Ilhan Omar, PewDiePie, CARICOM statement in the
Christchurch cluster), but also real false merges: American Airlines'
Venezuela suspension rides into the grounding cluster on the "airlines"
token. The other ~10 large clusters are low-coherence agglomerations of
local crime, courts, Indian election and business briefs (e.g. cluster 37:
3,828 canonical articles spanning Lok Sabha candidates, a Gmail outage and
Kashmir traffic; clusters 230/148/293: police/man/arrested/shooting). That is
the URL channel's failure mode: generic newsroom vocabulary survives the
2,000-article frequency cap because each token is individually below it, and
host-specific path boilerplate (`newsdisplay`, `aspx`, `articleshow`, `php`,
`node`) is not stripped. The Idai cluster is 21st by size.

At 0.15 everything above collapses into 20-30k-URL mixed clusters; at 0.45
the four stories fragment again (Idai-specific recall falls to 28%). The
usable window for this scorer is narrow and was not tuned - 0.3 was the
pre-registered middle setting.

## 3. Honest assessment

- The article graph is a clear improvement over the event projection on the
  four expected stories (2-4x best-cluster recall, Idai recovered), and it
  is not tuned to them. It is still not good enough to build country-level
  attention claims on: 79% of clusters are singletons, the largest clusters
  are dominated by junk agglomerations, and a false merge through shared
  generic tokens is easy to produce.
- Every number here is URL-proxy and, for the article runs, circular. The
  next thing to buy is a labelled pair set (anchor-story neighbourhoods +
  hard negatives + random background, wire groups kept together) so that
  candidate recall, pairwise precision and CEAF-e can be measured.
- Highest-value changes, in order: (1) a per-token IDF floor or document
  frequency cap by *rate* rather than count to suppress newsroom boilerplate;
  (2) score pairs with the actor/location labels from Events as a third
  channel (currently inspection-only); (3) a second layer that links incident
  clusters (crash, grounding, investigation) into story families rather than
  choosing one threshold for both granularities.
- Not observed: any translated article. All 4.4M rows have blank
  `MentionDocTranslationInfo`, so language strata cannot be reported and
  "English" cannot be asserted either.
