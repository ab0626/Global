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

## 3. v2 article graph (`clustering_v2.py`, `results/week-v2-1`)

Same prepared checkpoint, same Leiden/modularity/seed, same URL-proxy
scorer. Runtime 1,352 s, peak RSS 9.5 GiB (pair guard raised explicitly to
3e9 contributions / 8e7 candidate pairs; nothing truncated). Changes tested:

1. **URL token cleanup**: hex fragments (`11e9`, `94ab`) and non-3-digit
   numbers dropped (`737`/`302` kept); tokens with >= 100 articles whose
   day-of-week distribution is near-uniform (normalised entropy >= 0.95) are
   treated as newsroom boilerplate and dropped (1,964 tokens: `police`,
   `man`, `national`, `local`, `php`, ... but also `trump`, which is a
   cost); per-host tokens appearing in >= 20% of a host's articles are
   dropped for that host (2,213 host/token pairs: `newsdisplay`, `aspx`,
   `articleshow`, section names).
2. **Events label channel**: `actor:Actor1Name`, `actor:Actor2Name`,
   `geo:ActionGeo_FullName` per article via the mentioned event IDs, same
   burst filter (594 labels dropped, e.g. `actor:UNITED STATES`). Action
   geography is content metadata; publisher country is never a feature.
3. **Evidence gate**: an edge needs positive similarity in >= 2 channels or
   >= 0.6 in one; the score is the mean over channels both articles have.
4. **Story-family layer**: incident clusters are linked when their
   normalised centroids have cosine >= 0.25 (connected components over the
   top-10 centroid neighbours).
5. Benchmark widened from 4 to 15 URL regexes (Idai-strict, border veto,
   college admissions, Manafort, Pell, Smollett, bomb cyclone, Pelosi/
   impeachment, Venezuela blackout, R. Kelly, Gmail outage). Still proxies;
   `gmail_outage` (81 URLs) and `venezuela_blackout` are too loose to read.

The v1 reference row below is re-run through the v2 scorer (mean over
available channels instead of a fixed 1/2 weight), so it differs slightly
from section 2 (9,805 vs 6,854 largest cluster) and is the fair baseline.

| run | edges | clusters | singleton-only URLs | largest cluster (% of URLs) |
|---|---:|---:|---:|---:|
| v1 reference, combined 0.3 | 4.63M | 147,028 | 20.4% | 9,805 (1.4%) |
| **v2 URL cleanup, combined 0.3** | 5.47M | 108,336 | 15.3% | 17,788 (2.6%) |
| v2 URL + labels, combined 0.3 | 12.1M | 40,778 | 5.7% | 20,987 (3.1%) |
| v2 URL + labels, gated 0.2 | 16.3M | 30,102 | 4.3% | 33,999 (5.0%) |
| v2 URL + labels, gated 0.3 | 12.0M | 43,450 | 6.1% | 22,697 (3.3%) |
| v2 URL + labels, gated 0.4 | 5.53M | 104,341 | 14.2% | 15,943 (2.3%) |
| + story families on gated 0.3 | - | 43,374 families | 6.1% | 31,695 (4.6%) |

Best-cluster recall / keyword purity (`story_scores.csv`):

| story (URLs) | v1 ref | v2 URL cleanup | v2 URL+labels gated 0.3 | + families |
|---|---|---|---|---|
| crash_grounding (29,411) | .24 / .72 | .36 / .80 | .36 / .71 | **.70** / .65 |
| christchurch (14,239) | .33 / .60 | **.67** / .54 | .75 / .47 | .75 / .47 |
| brexit (16,961) | .29 / .63 | **.58** / .62 | .57 / .58 | .65 / .42 |
| idai_strict (339) | .79 / .43 | .88 / .22 | .95 / .09 | .95 / .09 |
| border_veto (1,902) | .90 / .44 | .93 / .29 | .75 / .12 | .91 / .07 |
| college_admissions (2,553) | .29 / .89 | .47 / .29 | .16 / .68 | .16 / .68 |
| manafort (2,878) | .68 / .82 | .72 / .59 | .56 / .18 | .56 / .18 |
| pell (904) | .79 / .52 | .86 / .47 | .85 / .18 | .85 / .18 |
| smollett (2,568) | .43 / .72 | .72 / .81 | .71 / .43 | .71 / .43 |
| pelosi_impeachment (1,513) | .74 / .30 | .81 / .17 | .40 / .36 | .40 / .36 |
| r_kelly (318) | .57 / .71 | .57 / .71 | .60 / .02 | .60 / .02 |
| bomb_cyclone (1,408) | .27 / .17 | .41 / .22 | .28 / .08 | .28 / .08 |

Reading of the 20 largest clusters (URL/metadata only, `clusters.md`):

- **URL cleanup is the win.** Recall roughly doubles on Christchurch, Brexit
  and Smollett with purity held; the crash (8,566 URLs: `ethiopian airlines
  crash plane victims`) and the 737 MAX grounding (13,356: `boeing max 737
  ground faa`) come out as *separate* incident clusters, which is why
  `crash_grounding` recall is only .36 for a single cluster. 10 of the top
  20 are one identifiable story (Christchurch, Brexit, grounding, crash,
  Pelosi/impeachment, border veto, Facebook/Instagram outage, climate
  strike, Holocaust-swastika school story, Loughlin/Huffman). The remaining
  crime/court agglomerations (clusters 123, 27, 124, 119, 101, 230; 4-8k
  URLs each) survive: once `police`/`man`/`shooting` are removed they are
  still glued by second-tier local-news vocabulary (`vegas`, `missouri`,
  `arrested`, `charged`) and by shared generic CAMEO events.
- **The Events label channel is a negative result at equal weight.** It
  adds 6.6M edges and cuts singletons to 6%, but purity collapses on the
  narrow stories (Manafort .59 -> .18, Pell .47 -> .18, R. Kelly .71 -> .02
  after his 255-URL cluster is absorbed into an 11,384-URL "court/case/
  judge" cluster) and it fuses India/China/Pakistan/Masood Azhar with Nigel
  Farage (cluster 40) and Honda recalls with Lilly Singh (cluster 43). The
  burst filter removes the worst offenders (`actor:UNITED STATES`,
  `actor:POLICE`, `actor:PRESIDENT`, `actor:GOVERNMENT`), but the surviving
  labels are still coarse: one actor name or one city is shared by many
  unrelated stories in a week. The evidence gate barely changes this (12.1M
  vs 12.0M edges), which suggests most label-backed pairs also clear the
  gate through a single weak URL or event match.
- **The story-family layer behaves as intended on top of the incident
  partition.** 71 of 43,374 families have more than one incident; the large
  ones are crash + grounding + Lion Air comparison + FAA airworthiness +
  delivery halt (5 incidents, 31,695 URLs -> `crash_grounding` .70), Brexit
  main + Brexit sidebar + Corbyn confidence motion, the two border-veto
  clusters (Senate vote vs. Trump veto), and Nebraska flooding + a 3-URL
  fragment. No family merged two of the four expected stories.
- Threshold behaviour is unchanged: 0.2 has the best broad recall and the
  worst purity; 0.4 restores fragmentation (104k clusters). 0.3 remains a
  pre-registered middle, not a validated optimum.

## 4. Honest assessment

- The article graph is a clear improvement over the event projection on the
  four expected stories (2-4x best-cluster recall, Idai recovered), and it
  is not tuned to them. v2's URL cleanup roughly doubles recall again on the
  wider 15-story set with purity held, and the incident/family split gives
  both granularities (crash vs. grounding as incidents, one family). It is
  still not good enough to build country-level attention claims on: 15% of
  URLs are singleton-only, a third of the largest clusters are local-crime
  agglomerations, and the Events label channel showed how cheaply a coarse
  shared feature manufactures false merges.
- Recommended configuration for downstream work today: **v2 URL cleanup,
  combined 0.3, Leiden** for incidents, plus the family layer for
  story-level roll-ups (measured here only on the label run; it needs a
  re-run on the URL-only partition); do not use the label channel at equal
  weight.
- Every number here is URL-proxy and, for the article runs, circular. The
  next thing to buy is a labelled pair set (anchor-story neighbourhoods +
  hard negatives + random background, wire groups kept together) so that
  candidate recall, pairwise precision and CEAF-e can be measured.
- Highest-value changes, in order: (1) a labelled pair set, without which
  every further threshold choice is guesswork; (2) use the Events labels
  only as a *veto/tie-breaker* (down-weighted, or required to agree when the
  URL score is marginal) rather than as an equal channel, and keep
  story-bearing tokens like `trump` out of the burst filter by requiring a
  minimum host spread as well as flat daily entropy; (3) break the crime
  agglomerations with a per-cluster coherence test (e.g. modularity
  contribution or centroid dispersion) that splits or unassigns clusters
  whose top tokens are all sub-1% of the cluster.
- Not observed: any translated article. All 4.4M rows have blank
  `MentionDocTranslationInfo`, so language strata cannot be reported and
  "English" cannot be asserted either.
