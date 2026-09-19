from datetime import datetime, timedelta
from pathlib import Path

WINDOW_START = datetime(2023, 2, 6, 0, 0)
WINDOW_END = datetime(2023, 2, 8, 23, 45)
EVENT_START = datetime(2023, 2, 6, 1, 17)
BATCH_INTERVAL = timedelta(minutes=15)

GDELT_BASE_URL = "https://data.gdeltproject.org/gdeltv2"
FILE_KINDS = ("gkg.csv", "translation.gkg.csv", "mentions.CSV", "translation.mentions.CSV")

WINDOW_NAME = WINDOW_START.strftime("%Y%m%d")
RAW_DIR = Path("/dev/shm/gdelt_raw") / WINDOW_NAME
DATA_DIR = Path(__file__).resolve().parent / "data" / WINDOW_NAME
ARTICLES_PATH = DATA_DIR / "articles.parquet"
MENTIONS_PATH = DATA_DIR / "mentions.parquet"
TITLE_EMBEDDINGS_PATH = DATA_DIR / "title_embeddings.npy"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

OUTLETS_BY_COUNTRY: dict[str, tuple[str, ...]] = {
    "US": ("nytimes.com", "cnn.com", "washingtonpost.com", "foxnews.com", "nbcnews.com", "apnews.com", "usatoday.com", "latimes.com"),
    "FR": ("lemonde.fr", "lefigaro.fr", "liberation.fr", "francetvinfo.fr", "20minutes.fr", "leparisien.fr", "bfmtv.com", "ouest-france.fr"),
    "DE": ("spiegel.de", "zeit.de", "faz.net", "sueddeutsche.de", "welt.de", "tagesschau.de", "bild.de", "n-tv.de"),
    "JP": ("asahi.com", "mainichi.jp", "nikkei.com", "yomiuri.co.jp", "nhk.or.jp", "sankei.com", "jiji.com", "japantimes.co.jp"),
    "BR": ("globo.com", "folha.uol.com.br", "estadao.com.br", "uol.com.br", "terra.com.br", "r7.com", "veja.abril.com.br", "correiobraziliense.com.br"),
}
DOMAIN_TO_COUNTRY: dict[str, str] = {
    domain: country for country, domains in OUTLETS_BY_COUNTRY.items() for domain in domains
}

EVENT_TITLE_PATTERN = r"(?i)earthquake|quake|erdbeben|s[ée]isme|terremoto|sismo|地震"


def batch_timestamps() -> list[str]:
    count = int((WINDOW_END - WINDOW_START) / BATCH_INTERVAL) + 1
    return [(WINDOW_START + i * BATCH_INTERVAL).strftime("%Y%m%d%H%M%S") for i in range(count)]


def raw_file_name(timestamp: str, kind: str) -> str:
    return f"{timestamp}.{kind}.zip"
