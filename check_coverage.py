"""Print number of texts with priority <= MAXP missing from an embedding cache. Usage: check_coverage.py CACHE_DIR MAXP"""
import sys, pandas as pd
from hm_fig4c.embed import load_cache
cache, maxp = sys.argv[1], int(sys.argv[2]); prereg_only = len(sys.argv) > 3 and sys.argv[3] == "--prereg-only"
t = pd.read_parquet("prepared/texts.parquet"); done = set(load_cache(cache)["text_id"])
t = t[t["priority"] <= maxp]
if prereg_only:
    t = t[t["prereg"]]
print(int((~t["text_id"].isin(done)).sum()))
