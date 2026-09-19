starting off

## v0: can GDELT articles be clustered by event across languages?

Runs on the EC2 box (`student@18.205.236.122`, code in `~/global`, uv at `~/.local/bin/uv`). Raw GDELT zips live in `/dev/shm/gdelt_raw/<window>` (tmpfs, lost on reboot), filtered tables and embeddings in `v0/data/<window>/`. The window, event start, keyword pattern and outlet list are in `v0/config.py`.

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
