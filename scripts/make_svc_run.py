"""Write a Kaggle notebook folder for one SVC run of nthl.svc.run.

    python scripts/make_svc_run.py <name> [--pool] [--weight W] [--routes api openrouter] [--drop-pool-labels L ...] [--C C]

Creates notebooks/svc-<name>/ with kernel-metadata.json and run.py. With
--pool the run reads pool.parquet from the nabilshah/nthl-pool-build output,
so every pool run trains on the same rows.
"""

import argparse
import json
from pathlib import Path

TEMPLATE = '''"""SVC run {name} (#26). Config: {config}"""

import glob
import json
import shutil
import sys

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {{code}}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import pandas as pd

from nthl.folds import load_folds
from nthl.svc import run

CONFIG = {config}
COMP = "/kaggle/input/competitions/name-that-llm"

train = pd.read_csv(f"{{COMP}}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{{COMP}}/test.csv")
assert len(train) == 2699

pool = None
if CONFIG.pop("pool"):
    path = glob.glob("/kaggle/input/**/pool.parquet", recursive=True)
    assert len(path) == 1, f"expected one pool.parquet, found {{path}}"
    pool = pd.read_parquet(path[0])

summary = run(train, test, pool=pool, out="/kaggle/working", name="{name}", **CONFIG)
print(json.dumps(summary, indent=1))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--routes", nargs="*", default=None)
    ap.add_argument("--drop-pool-labels", nargs="*", default=None)
    ap.add_argument("--C", type=float, default=0.5)
    args = ap.parse_args()

    config = {"pool": args.pool, "pool_weight": args.weight, "routes": args.routes, "C": args.C}
    if args.drop_pool_labels:
        config["drop_pool_labels"] = args.drop_pool_labels
    folder = Path(__file__).resolve().parent.parent / "notebooks" / f"svc-{args.name}"
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": f"nabilshah/nthl-svc-{args.name}",
        "title": f"nthl svc {args.name}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["nabilshah/nthl-src"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": ["nabilshah/nthl-pool-build"] if args.pool else [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / "run.py").write_text(TEMPLATE.format(name=args.name, config=repr(config)))
    print(folder)


if __name__ == "__main__":
    main()
