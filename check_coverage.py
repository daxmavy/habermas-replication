"""Print number of texts with priority <= MAXP missing from an embedding cache. Usage: check_coverage.py CACHE_DIR MAXP"""
import sys, pandas as pd
from hm_fig4c.embed import load_cache
cache, maxp = sys.argv[1], int(sys.argv[2])
t = pd.read_parquet("prepared/texts.parquet"); done = set(load_cache(cache)["text_id"])
print(int((~t[t["priority"] <= maxp]["text_id"].isin(done)).sum()))
