"""Embed texts with a Sentence-T5 model; incremental, resumable cache of chunk parquet files."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# The report uses the pre-registered rounds of cohorts 1-3 plus the position-statement endpoints,
# which is what priority <= MAX_PRIORITY and prereg select (see hm_fig4c.data.build_text_table).
MAX_PRIORITY = 3


def load_cache(cache_dir: Path) -> pd.DataFrame:
    files = sorted(Path(cache_dir).glob("chunk_*.parquet"))
    if not files:
        return pd.DataFrame({"text_id": pd.Series(dtype=str), "embedding": pd.Series(dtype=object)})
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("text_id")


def load_embeddings(cache_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    """Return (text_id -> row index, matrix)."""
    df = load_cache(cache_dir)
    mat = np.stack(df["embedding"].values).astype(np.float32) if len(df) else np.zeros((0, 768), np.float32)
    return {t: i for i, t in enumerate(df["text_id"])}, mat


def run(texts_path: Path, model_name: str, cache_dir: Path, max_seq_length: int = 512,
        batch_size: int = 16, chunk_size: int = 256):
    """Embed the texts the analysis needs into cache_dir, in fp32, skipping what is already cached."""
    import torch
    from sentence_transformers import SentenceTransformer

    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    texts = pd.read_parquet(texts_path)
    if "prereg" not in texts:
        texts["prereg"] = True
    texts = texts.sort_values(["priority", "prereg", "n_words"], ascending=[True, False, True])
    texts = texts[(texts["priority"] <= MAX_PRIORITY) & texts["prereg"]]
    done = set(load_cache(cache_dir)["text_id"])
    todo = texts[~texts["text_id"].isin(done)].reset_index(drop=True)
    print(f"{model_name}: {len(done)} cached, {len(todo)} to embed", flush=True)
    if not len(todo):
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"  # whatever this machine has
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float32})
    model.max_seq_length = max_seq_length
    n_done = 0
    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start:start + chunk_size]
        emb = model.encode(chunk["text"].tolist(), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        out = pd.DataFrame({"text_id": chunk["text_id"].values, "embedding": list(emb.astype(np.float32))})
        final = cache_dir / f"chunk_{int(time.time() * 1000)}_{start}.parquet"
        out.to_parquet(final.with_suffix(".tmp"), index=False)
        final.with_suffix(".tmp").replace(final)  # atomic: a chunk killed mid-write never appears in the cache glob
        n_done += len(chunk)
        print(f"  {n_done}/{len(todo)} done", flush=True)
    print("EMBED_DONE", model_name, flush=True)
