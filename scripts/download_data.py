"""Fetch the three public Habermas Machine dataset files the Fig. 4C analysis reads into data/.

Source: the public release accompanying Tessler et al. (2024), Science 386, adq2852 -- Google DeepMind's
GCS bucket `habermas_machine`, prefix `datasets/`, served read-only over https://storage.googleapis.com/.
The fourth released file, hm_all_final_preference_rankings.parquet, is not used here. The MD5s below are
the bucket's ETags, checked against the copies this report was built from.

    python scripts/download_data.py                  # skips files already present with the right checksum
    python scripts/download_data.py --data-dir data
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/habermas_machine/datasets"
FILES = {
    "hm_all_candidate_comparisons.parquet": "22dddc10acd72ee975f77d0e3c4adc63",
    "hm_all_position_statement_ratings.parquet": "3890689576997c745a61accab950cb3d",
    "hm_all_round_survey_responses.parquet": "2bfc92da02a0021a422df00d4ae83fc5",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, want: str, data_dir: Path) -> None:
    """Download one file unless it is already there with the right checksum."""
    out = data_dir / name
    if out.exists() and md5(out) == want:
        print(f"have {name}", flush=True)
        return
    print(f"downloading {name} ...", flush=True)
    part = out.with_name(out.name + ".part")
    with urllib.request.urlopen(f"{BASE}/{name}") as response, open(part, "wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)
    got = md5(part)
    if got != want:
        part.unlink()
        raise SystemExit(f"checksum mismatch for {name}: got {got}, want {want}")
    part.replace(out)  # atomic: a download killed part way never appears as data/<name>
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, want in FILES.items():
        fetch(name, want, data_dir)


if __name__ == "__main__":
    main()
