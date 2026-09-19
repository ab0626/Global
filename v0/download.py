import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import FILE_KINDS, GDELT_BASE_URL, RAW_DIR, batch_timestamps, raw_file_name

PARALLELISM = 32


def download(file_name: str) -> str:
    target = RAW_DIR / file_name
    if target.exists():
        return f"skip {file_name}"
    with urllib.request.urlopen(f"{GDELT_BASE_URL}/{file_name}") as response:
        body = response.read()
    partial = target.with_suffix(".part")
    partial.write_bytes(body)
    partial.rename(target)
    return f"ok   {file_name} {len(body) // 1024} KB"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_names = [raw_file_name(ts, kind) for ts in batch_timestamps() for kind in FILE_KINDS]
    with ThreadPoolExecutor(PARALLELISM) as pool:
        for line in pool.map(download, file_names):
            print(line, flush=True)


if __name__ == "__main__":
    main()
