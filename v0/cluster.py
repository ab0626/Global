import argparse
from dataclasses import dataclass

import igraph as ig
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ARTICLES_PATH, EVENT_START, EVENT_TITLE_PATTERN, MENTIONS_PATH, OUTLETS_BY_COUNTRY, TITLE_EMBEDDINGS_PATH

RESOLUTIONS = (0.5, 1.0, 2.0)
TIMELINE_HOURS = 36


def names_from_mentions(field: str | None) -> list[str]:
    if not field:
        return []
    return [segment.rsplit(",", 1)[0] for segment in field.split(";") if segment]


def names_from_locations(field: str | None) -> list[str]:
    if not field:
        return []
    return [segment.split("#")[1] for segment in field.split(";") if segment]


def ids_from_list(field: list[str] | None) -> list[str]:
    return field or []


ENTITY_PARSERS = {
    "P": ("persons", names_from_mentions),
    "O": ("organizations", names_from_mentions),
    "L": ("locations", names_from_locations),
    "N": ("all_names", names_from_mentions),
    "E": ("event_ids", ids_from_list),
}


def entity_tokens(row: dict, entity_types: str) -> list[str]:
    tokens: list[str] = []
    for prefix in entity_types:
        column, parse = ENTITY_PARSERS[prefix]
        tokens.extend(f"{prefix}:{name}" for name in parse(row[column]))
    return tokens


def load_articles() -> pl.DataFrame:
    event_ids = pl.read_parquet(MENTIONS_PATH).group_by("url").agg(event_ids=pl.col("event_id"))
    return pl.read_parquet(ARTICLES_PATH).join(event_ids, on="url", how="left")


def build_entity_vectors(articles: pl.DataFrame, entity_types: str, max_df: float) -> tuple[sp.csr_matrix, np.ndarray]:
    documents = [entity_tokens(row, entity_types) for row in articles.iter_rows(named=True)]
    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens, sublinear_tf=True, min_df=2, max_df=max_df)
    vectors = vectorizer.fit_transform(documents)
    print(f"{vectors.shape[0]} articles x {vectors.shape[1]} entity terms (types={entity_types}, max_df={max_df})", flush=True)
    return vectors.tocsr(), vectorizer.get_feature_names_out()


def similarity_matrix(entity_vectors: sp.csr_matrix, title_embeddings: np.ndarray, title_weight: float) -> np.ndarray:
    entity_similarity = (entity_vectors @ entity_vectors.T).toarray().astype(np.float32)
    title_similarity = title_embeddings @ title_embeddings.T
    similarity = (1 - title_weight) * entity_similarity + title_weight * title_similarity
    np.fill_diagonal(similarity, 0)
    return similarity


def thresholded_graph(similarity: np.ndarray, threshold: float) -> ig.Graph:
    rows, cols = np.nonzero(np.triu(similarity, k=1) >= threshold)
    weights = similarity[rows, cols]
    return ig.Graph(n=similarity.shape[0], edges=np.column_stack([rows, cols]).tolist(), edge_attrs={"weight": weights.tolist()})


@dataclass(frozen=True)
class ClusterReport:
    config: str
    n_clusters: int
    size: int
    purity: float
    recall: float
    by_country: dict[str, int]
    by_language: dict[str, int]


def title_matches(articles: pl.DataFrame) -> np.ndarray:
    return articles.get_column("title").str.contains(EVENT_TITLE_PATTERN).to_numpy()


def event_cluster_label(articles: pl.DataFrame, labels: np.ndarray, anchor_country: str | None) -> int:
    anchored = title_matches(articles)
    if anchor_country is not None:
        anchored &= (articles.get_column("country") == anchor_country).to_numpy()
    return int(np.bincount(labels[anchored]).argmax())


def evaluate(config: str, articles: pl.DataFrame, labels: np.ndarray, anchor_country: str | None) -> ClusterReport:
    in_cluster = labels == event_cluster_label(articles, labels, anchor_country)
    matching = title_matches(articles)
    cluster = articles.filter(in_cluster)
    return ClusterReport(
        config=config,
        n_clusters=int(labels.max()) + 1,
        size=int(in_cluster.sum()),
        purity=float(matching[in_cluster].mean()),
        recall=float(in_cluster[matching].mean()),
        by_country={c: n for c, n in cluster.group_by("country").len().sort("country").iter_rows()},
        by_language={l: n for l, n in cluster.group_by("language").len().sort("len", descending=True).iter_rows()},
    )


def print_summary(reports: list[ClusterReport]) -> None:
    print("\n=== event cluster by config (purity = share of cluster titles matching the event keywords; recall = share of all matching titles captured) ===")
    print(f"{'config':<22}{'clusters':>9}{'size':>7}{'purity':>8}{'recall':>8}  countries / languages")
    for r in reports:
        print(f"{r.config:<22}{r.n_clusters:>9}{r.size:>7}{r.purity:>8.2f}{r.recall:>8.2f}  {r.by_country} / {r.by_language}")


def top_terms(vectors: sp.csr_matrix, feature_names: np.ndarray, members: np.ndarray, count: int = 12) -> str:
    centroid = np.asarray(vectors[members].mean(axis=0)).ravel()
    return ", ".join(feature_names[i] for i in np.argsort(-centroid)[:count])


def print_detail(articles: pl.DataFrame, labels: np.ndarray, vectors: sp.csr_matrix, feature_names: np.ndarray, config: str, anchor_country: str | None) -> None:
    label = event_cluster_label(articles, labels, anchor_country)
    tagged = articles.with_columns(
        label=pl.Series(labels),
        in_cluster=pl.Series(labels == label),
        matches=pl.Series(title_matches(articles)),
    )
    print(f"\n=== detail for {config}: event cluster {label} ===")
    print(f"top terms: {top_terms(vectors, feature_names, labels == label)}")

    print("\n-- where each country's matching titles landed (cluster: count, size, top terms) --")
    for country in OUTLETS_BY_COUNTRY:
        destinations = tagged.filter(pl.col("matches") & (pl.col("country") == country)).group_by("label").len().sort("len", descending=True).head(4)
        print(f"  {country}:")
        for other, count in destinations.iter_rows():
            print(f"    cluster {other}: {count:>4}  size {int((labels == other).sum()):>5}  {top_terms(vectors, feature_names, labels == other, 8)}")

    print("\n-- possible contamination: in cluster, title does not match (up to 30) --")
    for source, title in tagged.filter(pl.col("in_cluster") & ~pl.col("matches")).select("source", "title").head(30).iter_rows():
        print(f"  [{source}] {title[:110]}")

    print("\n-- possible fragmentation: title matches but outside cluster (up to 30) --")
    for source, title, other in tagged.filter(~pl.col("in_cluster") & pl.col("matches")).select("source", "title", "label").head(30).iter_rows():
        print(f"  [{source}] (cluster {other}) {title[:100]}")

    cluster = tagged.filter(pl.col("in_cluster")).with_columns(
        hours_after=((pl.col("batch_time") - EVENT_START).dt.total_minutes() / 60).floor().cast(pl.Int32)
    )
    print("\n-- first article per country in cluster (hours after event start, source, title) --")
    firsts = cluster.sort("batch_time").group_by("country", maintain_order=True).first()
    for country, hours, source, title in firsts.select("country", "hours_after", "source", "title").sort("hours_after").iter_rows():
        print(f"  {country} {hours:+d}h [{source}] {title[:90]}")

    print(f"\n-- cluster articles per country per hour after event start (first {TIMELINE_HOURS}h) --")
    timeline = (
        cluster.filter(pl.col("hours_after").is_between(0, TIMELINE_HOURS - 1))
        .group_by("hours_after", "country").len()
        .pivot(on="country", index="hours_after", values="len")
        .fill_null(0)
        .sort("hours_after")
    )
    with pl.Config(tbl_rows=TIMELINE_HOURS, tbl_cols=len(OUTLETS_BY_COUNTRY) + 1):
        print(timeline.select("hours_after", *[c for c in OUTLETS_BY_COUNTRY if c in timeline.columns]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-types", default="PONE", help="subset of P (persons), O (orgs), L (locations), N (all names), E (GDELT event ids)")
    parser.add_argument("--max-df", type=float, default=0.5)
    parser.add_argument("--title-weight", type=float, default=0.0, help="0 = entity TF-IDF only, 1 = title embeddings only")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5")
    parser.add_argument("--anchor-country", default=None, help="pick the event cluster by this country's matching titles instead of all countries")
    parser.add_argument("--detail-threshold", type=float, default=0.4)
    parser.add_argument("--detail-resolution", type=float, default=1.0)
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    articles = load_articles()
    vectors, feature_names = build_entity_vectors(articles, args.entity_types, args.max_df)
    title_embeddings = np.load(TITLE_EMBEDDINGS_PATH) if args.title_weight > 0 else np.zeros((articles.height, 1), dtype=np.float32)
    similarity = similarity_matrix(vectors, title_embeddings, args.title_weight)
    print(f"title_weight={args.title_weight}; similarity quantiles 50/90/99/99.9%: {np.quantile(similarity[np.triu_indices(min(5000, similarity.shape[0]), 1)], [0.5, 0.9, 0.99, 0.999]).round(3)}", flush=True)
    reports: list[ClusterReport] = []
    detail_labels: np.ndarray | None = None
    for threshold in thresholds:
        graph = thresholded_graph(similarity, threshold)
        print(f"threshold {threshold}: {graph.ecount()} edges", flush=True)
        reports.append(evaluate(f"t={threshold} components", articles, np.array(graph.connected_components().membership), args.anchor_country))
        for resolution in RESOLUTIONS:
            membership = graph.community_leiden(objective_function="modularity", weights="weight", resolution=resolution, n_iterations=-1).membership
            labels = np.array(membership)
            reports.append(evaluate(f"t={threshold} leiden r={resolution}", articles, labels, args.anchor_country))
            if threshold == args.detail_threshold and resolution == args.detail_resolution:
                detail_labels = labels
    print_summary(reports)
    if detail_labels is None:
        return
    print_detail(articles, detail_labels, vectors, feature_names, f"t={args.detail_threshold} leiden r={args.detail_resolution} title_weight={args.title_weight}", args.anchor_country)


if __name__ == "__main__":
    main()
