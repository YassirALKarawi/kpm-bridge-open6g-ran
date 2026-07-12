#!/usr/bin/env python3
"""Download and hash the deterministic public ColO-RAN evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM = "wineslab/colosseum-oran-coloran-dataset"
COMMIT = "bd86629d07d5fbfb778ebe3afd9d0b05e5191c6b"
BASE_URL = f"https://raw.githubusercontent.com/{UPSTREAM}/{COMMIT}"
SCHEDULERS = ("sched0", "sched1", "sched2")
TRAINING_CONFIGS = ("tr0", "tr9", "tr18")
EXPERIMENTS = ("exp1", "exp2")
UE_BY_BASE_STATION = {
    # One UE from each documented traffic slice per selected BS.
    "bs1": ("ue2", "ue3", "ue4"),
    "bs4": ("ue19", "ue20", "ue21"),
}


def selected_paths() -> list[str]:
    return [
        f"rome_static_medium/{sched}/{training}/{experiment}/{bs}/{ue}.csv"
        for sched in SCHEDULERS
        for training in TRAINING_CONFIGS
        for experiment in EXPERIMENTS
        for bs, user_equipments in UE_BY_BASE_STATION.items()
        for ue in user_equipments
    ]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_one(relative_path: str, root: Path, retries: int = 4) -> dict[str, object]:
    target = root / relative_path
    url = f"{BASE_URL}/{relative_path}"
    if target.exists():
        payload = target.read_bytes()
    else:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "KPM-Bridge-reproducibility/1.0"},
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = response.read()
                break
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
                time.sleep(0.75 * (attempt + 1))
        else:
            raise RuntimeError(f"failed to download {url}") from last_error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    if not payload.startswith(b"time,cc,pci,earfcn,"):
        raise ValueError(f"unexpected schema in {relative_path}")
    return {
        "path": relative_path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/colosseum"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    paths = selected_paths()
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_one, path, args.root): path for path in paths}
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda item: str(item["path"]))
    manifest = {
        "dataset": "ColO-RAN",
        "repository": f"https://github.com/{UPSTREAM}",
        "upstream_commit": COMMIT,
        "license": "GPL-3.0",
        "selection": {
            "schedulers": list(SCHEDULERS),
            "training_configs": list(TRAINING_CONFIGS),
            "experiments": list(EXPERIMENTS),
            "user_equipments_by_base_station": {
                key: list(value) for key, value in UE_BY_BASE_STATION.items()
            },
        },
        "file_count": len(records),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    manifest_path = Path("data/colosseum_subset_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("file_count", "total_bytes", "upstream_commit")}, indent=2))


if __name__ == "__main__":
    main()
