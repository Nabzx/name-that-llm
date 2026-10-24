"""Build the training pool once (#26). Every pool experiment reads its output.

Writes pool.parquet (text, model, source, route) and a count per model and route.
"""

import glob
import json
import shutil
import sys

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import pandas as pd

from nthl.labels import LABELS
from nthl.pool import ROUTES, build_pool

train = pd.read_csv("/kaggle/input/competitions/name-that-llm/train.csv")
pool = build_pool("/kaggle/input/datasets", train)

missing = [ref for ref in ROUTES if ref not in set(pool["source"])]
assert not missing, f"pools that gave no rows: {missing}"
assert set(pool["model"]) == set(LABELS)
assert pool["text"].is_unique
assert pool["text"].map(lambda t: isinstance(t, str)).all()
print(pool["source"].value_counts().to_string())

pool.to_parquet("/kaggle/working/pool.parquet", index=False)
counts = pd.crosstab(pool["model"], pool["route"]).reindex(list(LABELS))
counts.to_csv("/kaggle/working/pool_counts.csv")
summary = {"rows": len(pool), "by_route": pool["route"].value_counts().to_dict(), "by_model_min": int(pool["model"].value_counts().min()), "by_model_max": int(pool["model"].value_counts().max())}
json.dump(summary, open("/kaggle/working/pool_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(counts.to_string())
