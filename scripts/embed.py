"""Embed the texts the Fig. 4C analysis needs with each Sentence-T5 size.

One cache directory per model, `<out-dir>/<model>/`. The cache is one parquet per chunk and is
resumable: re-running skips whatever is already embedded.

    python scripts/embed.py                                  # all four sizes
    python scripts/embed.py --models st5-base st5-large      # a subset
    python scripts/embed.py --texts prepared/texts.parquet --out-dir embeddings
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hm_fig4c.embed import run  # noqa: E402

MODELS = ["st5-base", "st5-large", "st5-xl", "st5-xxl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    ap.add_argument("--texts", default="prepared/texts.parquet")
    ap.add_argument("--out-dir", default="embeddings")
    a = ap.parse_args()
    for tag in a.models:
        print(f"=== {tag} ===", flush=True)
        run(Path(a.texts), f"sentence-transformers/sentence-t5-{tag.removeprefix('st5-')}",
            Path(a.out_dir) / tag)


if __name__ == "__main__":
    main()
