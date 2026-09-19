import html
import zipfile
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from config import ARTICLES_PATH, DATA_DIR, DOMAIN_TO_COUNTRY, RAW_DIR

# GKG 2.1 column positions (0-based), in ascending order so they line up with polars' projection order.
COLUMNS: dict[int, str] = {
    1: "batch_date",
    3: "source",
    4: "url",
    10: "locations",
    12: "persons",
    14: "organizations",
    23: "all_names",
    25: "translation_info",
    26: "extras",
}
PARALLELISM = 48


def source_matches(domain: str) -> pl.Expr:
    return (pl.col("source") == domain) | pl.col("source").str.ends_with(f".{domain}")


def country_expression() -> pl.Expr:
    return pl.coalesce(
        [pl.when(source_matches(domain)).then(pl.lit(country)) for domain, country in DOMAIN_TO_COUNTRY.items()]
    )


def read_gdelt_csv(path: Path, columns: dict[int, str]) -> pl.DataFrame:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0])
    frame = pl.read_csv(
        raw,
        separator="\t",
        has_header=False,
        quote_char=None,
        encoding="utf8-lossy",
        infer_schema=False,
        columns=list(columns),
        truncate_ragged_lines=True,
        ignore_errors=True,
    )
    frame.columns = list(columns.values())
    return frame


def extract_file(path: Path) -> pl.DataFrame:
    frame = read_gdelt_csv(path, COLUMNS)
    return frame.with_columns(country=country_expression()).filter(pl.col("country").is_not_null())


def derive_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        batch_time=pl.col("batch_date").str.to_datetime("%Y%m%d%H%M%S"),
        language=pl.col("translation_info").str.extract(r"srclc:(\w+)").fill_null("eng"),
        title=pl.col("extras")
        .str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>")
        .fill_null("")
        .map_elements(html.unescape, return_dtype=pl.String),
    ).drop("batch_date", "translation_info", "extras")


def main() -> None:
    paths = sorted(RAW_DIR.glob("*.gkg.csv.zip"))
    print(f"extracting {len(paths)} files", flush=True)
    with Pool(PARALLELISM) as pool:
        frames = pool.map(extract_file, paths)
    articles = (
        derive_columns(pl.concat(frames))
        .sort("batch_time")
        .unique(subset="url", keep="first", maintain_order=True)
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    articles.write_parquet(ARTICLES_PATH)
    print(f"{articles.height} articles -> {ARTICLES_PATH}")
    print(articles.group_by("country", "language").len().sort("country", "len", descending=[False, True]))
    print(articles.group_by("source").len().sort("len", descending=True))
    print(f"empty titles: {(articles.get_column('title') == '').sum()}")


if __name__ == "__main__":
    main()
