"""Build the API tables from downloaded GDELT v2 files; see readme.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api.build import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mentions", type=Path, default=Path("data/mentions"))
    parser.add_argument("--events", type=Path, default=Path("data/events"))
    parser.add_argument("--output", type=Path, default=Path("data/api"))
    arguments = parser.parse_args()
    meta = build(arguments.mentions, arguments.events, arguments.output)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
