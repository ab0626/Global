import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from config import ARTICLES_PATH, EMBEDDING_MODEL, TITLE_EMBEDDINGS_PATH


def main() -> None:
    titles = pl.read_parquet(ARTICLES_PATH, columns=["title"]).get_column("title").to_list()
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(titles, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    embeddings[np.array([title == "" for title in titles])] = 0
    np.save(TITLE_EMBEDDINGS_PATH, embeddings.astype(np.float32))
    print(f"{embeddings.shape} -> {TITLE_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
