#!/usr/bin/env python3
"""Execute the full deterministic KPM-Bridge benchmark."""

from __future__ import annotations

import json

from kpm_bridge.evaluation import run_full_benchmark


if __name__ == "__main__":
    print(json.dumps(run_full_benchmark(), indent=2))
