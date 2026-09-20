#!/usr/bin/env sh
# Fetch and run the whole attention pipeline for one window.
#   scripts/run_window.sh <tag> <YYYYMMDD start> <YYYYMMDD end>
# e.g. scripts/run_window.sh 20200804 20200804 20200806
set -eu
tag=$1; start=$2; end=$3
lookup=${GDELT_DOMAIN_LOOKUP:-$HOME/gdelt-reference/domains_by_country.txt}
cd "$(dirname "$0")/.."
uv run python scripts/fetch_window.py --start "${start}00" --end "${end}23" --output "data/raw/$tag"
uv run python -m attention.preprocess --raw "data/raw/$tag" --start "$start" --end "$end" \
  --domain-lookup "$lookup" --output "data/clean/$tag" --allow-partial
uv run python -m attention.atomic --clean "data/clean/$tag" --output "data/features/$tag"
uv run python -m attention.embed --features "data/features/$tag" --output "data/embeddings/$tag"
uv run python -m attention.cluster --features "data/features/$tag" --embeddings "data/embeddings/$tag" \
  --output "data/clusters/$tag"
uv run python -m attention.materialize --clean "data/clean/$tag" --features "data/features/$tag" \
  --clusters "data/clusters/$tag" --embeddings "data/embeddings/$tag" --output "data/store/$tag" \
  --domain-lookup "$lookup"
uv run python -m attention.analytics "data/store/$tag"
echo "done $tag"
