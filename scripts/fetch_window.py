"""Download a window of GDELT 2.x 15-minute files from data.gdeltproject.org.

Fetches all file kinds needed by the attention pipeline (English and translated GKG
and Mentions, plus Events export) and records per-file status in a manifest so the
download is resumable and missing quarter-hours are visible.

    uv run python scripts/fetch_window.py --start 2023020600 --end 2023020823 \
        --output data/raw/turkey-quake
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.gdeltproject.org/gdeltv2"
KINDS = (
    "gkg.csv",
    "translation.gkg.csv",
    "mentions.CSV",
    "translation.mentions.CSV",
    "export.CSV",
    "translation.export.CSV",
)
STEP = timedelta(minutes=15)


def timestamps(start: datetime, end: datetime) -> list[str]:
    out = []
    current = start
    while current <= end:
        out.append(current.strftime("%Y%m%d%H%M%S"))
        current += STEP
    return out


def fetch(name: str, directory: Path) -> dict[str, object]:
    target = directory / name
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return {"file": name, "status": "present", "bytes": target.stat().st_size, "sha256": digest}
    try:
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        return {"file": name, "status": f"http_{error.code}", "bytes": 0, "sha256": None}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"file": name, "status": f"error:{error}", "bytes": 0, "sha256": None}
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(body)
    partial.rename(target)
    return {
        "file": name,
        "status": "downloaded",
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYYMMDDHH (inclusive)")
    parser.add_argument("--end", required=True, help="YYYYMMDDHH (inclusive hour)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y%m%d%H")
    end = datetime.strptime(args.end, "%Y%m%d%H") + timedelta(minutes=45)
    kinds = args.kinds.split(",")
    args.output.mkdir(parents=True, exist_ok=True)
    names = [f"{ts}.{kind}.zip" for ts in timestamps(start, end) for kind in kinds]

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda name: fetch(name, args.output), names))

    manifest = {
        "base_url": BASE_URL,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "kinds": kinds,
        "expected_files": len(names),
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({str(row["status"]) for row in rows})
        },
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
