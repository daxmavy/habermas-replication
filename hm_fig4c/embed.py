"""Embed texts with a Sentence-T5 model; incremental, resumable cache of chunk parquet files."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


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


def run(texts_path: Path, model_name: str, cache_dir: Path, max_seq_length: int, batch_size: int, chunk_size: int, max_priority: int | None, dtype: str = "fp32", prereg_only: bool = False):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    texts = pd.read_parquet(texts_path)
    if "prereg" not in texts:
        texts["prereg"] = True
    texts = texts.sort_values(["priority", "prereg", "n_words"], ascending=[True, False, True])
    if max_priority is not None:
        texts = texts[texts["priority"] <= max_priority]
    if prereg_only:
        texts = texts[texts["prereg"]]
    done = set(load_cache(cache_dir)["text_id"])
    todo = texts[~texts["text_id"].isin(done)].reset_index(drop=True)
    print(f"{model_name}: {len(done)} cached, {len(todo)} to embed", flush=True)
    if not len(todo):
        return
    kw = {"model_kwargs": {"torch_dtype": torch.bfloat16}} if dtype == "bf16" else {}
    model = SentenceTransformer(model_name, device="cpu", **kw)
    model.max_seq_length = max_seq_length
    print("max_seq_length =", model.max_seq_length, "| threads =", torch.get_num_threads(), "| dtype =", dtype, flush=True)
    t0 = time.time(); n_done = 0
    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start:start + chunk_size]
        emb = model.encode(chunk["text"].tolist(), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        out = pd.DataFrame({"text_id": chunk["text_id"].values, "embedding": list(emb.astype(np.float32))})
        out.to_parquet(cache_dir / f"chunk_{int(time.time() * 1000)}_{start}.parquet", index=False)
        n_done += len(chunk)
        el = time.time() - t0
        print(f"  {n_done}/{len(todo)} done | {n_done / el:.2f} texts/s | eta {(len(todo) - n_done) / (n_done / el) / 60:.1f} min", flush=True)
    print("EMBED_DONE", model_name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="prepared/texts.parquet")
    ap.add_argument("--model", default="sentence-transformers/sentence-t5-base")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--max-priority", type=int, default=None, help="only embed texts with priority <= this")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--prereg-only", action="store_true", help="only texts belonging to pre-registered rounds (and their questions)")
    a = ap.parse_args()
    run(Path(a.texts), a.model, Path(a.cache_dir), a.max_seq_length, a.batch_size, a.chunk_size, a.max_priority, a.dtype, a.prereg_only)
