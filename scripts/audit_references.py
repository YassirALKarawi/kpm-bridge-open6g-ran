#!/usr/bin/env python3
"""Cross-check DOI metadata and explicit no-DOI records in references.bib."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def normalize(value: str) -> str:
    value = re.sub(r"\\['`^.uvHcbdkrt]\{?([A-Za-z])\}?", r"\1", value)
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(value.split())


def entries(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    start = 0
    while True:
        match = re.search(r"@\w+\{([^,]+),", text[start:])
        if not match:
            break
        key = match.group(1)
        entry_start = start + match.start()
        brace = 0
        opened = False
        end = entry_start
        for index, character in enumerate(text[entry_start:], start=entry_start):
            if character == "{":
                brace += 1
                opened = True
            brace -= character == "}"
            if opened and brace == 0:
                end = index + 1
                break
        found.append((key, text[entry_start:end]))
        start = end
    return found


def field(entry: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*=\s*\{{(.*?)\}}\s*,?\n", entry, flags=re.I | re.S)
    return match.group(1).strip() if match else None


def main() -> None:
    bibliography = Path("manuscript/references.bib").read_text(encoding="utf-8")
    rows: list[dict[str, object]] = []
    for key, entry in entries(bibliography):
        title = field(entry, "title") or ""
        doi = field(entry, "doi")
        note = field(entry, "note") or ""
        row: dict[str, object] = {"key": key, "bib_title": title, "doi": doi or ""}
        if not doi:
            row.update(
                {
                    "status": "NO_DOI_DECLARED" if "no doi" in note.lower() else "MISSING_DOI_NOTE",
                    "registry_title": "",
                    "title_similarity": "",
                }
            )
        else:
            url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
            request = urllib.request.Request(url, headers={"User-Agent": "KPM-Bridge-reference-audit/1.0"})
            registry = "Crossref"
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    message = json.load(response)["message"]
                registry_title = message.get("title", [""])[0]
            except HTTPError as error:
                if error.code != 404:
                    raise
                datacite_url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="")
                datacite_request = urllib.request.Request(
                    datacite_url, headers={"User-Agent": "KPM-Bridge-reference-audit/1.0"}
                )
                with urllib.request.urlopen(datacite_request, timeout=30) as response:
                    attributes = json.load(response)["data"]["attributes"]
                registry_title = attributes.get("titles", [{"title": ""}])[0]["title"]
                registry = "DataCite"
            similarity = SequenceMatcher(None, normalize(title), normalize(registry_title)).ratio()
            row.update(
                {
                    "status": "VERIFIED" if similarity >= 0.90 else "TITLE_MISMATCH",
                    "registry": registry,
                    "registry_title": registry_title,
                    "title_similarity": round(similarity, 4),
                }
            )
            time.sleep(0.05)
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = Path("reproducibility/outputs/reference_audit.csv")
    frame.to_csv(output, index=False)
    counts = frame.status.value_counts().to_dict()
    print(json.dumps({"records": len(frame), "status": counts, "output": str(output)}, indent=2))
    if any(status not in {"VERIFIED", "NO_DOI_DECLARED"} for status in frame.status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
