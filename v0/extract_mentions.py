from multiprocessing import Pool
from pathlib import Path

import polars as pl

from config import ARTICLES_PATH, MENTIONS_PATH, RAW_DIR
from extract import PARALLELISM, country_expression, read_gdelt_csv

# GDELT 2.0 Mentions column positions (0-based).
COLUMNS: dict[int, str] = {
    0: "event_id",
    4: "source",
    5: "url",
    11: "confidence",
}


def extract_file(path: Path) -> pl.DataFrame:
    frame = read_gdelt_csv(path, COLUMNS)
    return frame.filter(country_expression().is_not_null()).select("url", "event_id", "confidence")


def main() -> None:
    paths = sorted(RAW_DIR.glob("*.mentions.CSV.zip"))
    print(f"extracting {len(paths)} mention files", flush=True)
    with Pool(PARALLELISM) as pool:
        frames = pool.map(extract_file, paths)
    article_urls = pl.read_parquet(ARTICLES_PATH, columns=["url"])
    mentions = (
        pl.concat(frames)
        .with_columns(confidence=pl.col("confidence").cast(pl.Int16))
        .join(article_urls, on="url", how="semi")
        .unique(subset=["url", "event_id"])
    )
    mentions.write_parquet(MENTIONS_PATH)
    print(f"{mentions.height} mentions, {mentions.get_column('url').n_unique()} of {article_urls.height} articles have >=1 event id")
    print(mentions.group_by("url").len().get_column("len").describe())


if __name__ == "__main__":
    main()
