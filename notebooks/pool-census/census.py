"""Pool census (#19): what is in each public passage pool? Fits nothing.

For every attached dataset, find each table with a text column and a label
column, keep rows whose label is one of the 18 models, and count: passages
per model, length, line breaks, markdown, punctuation that breaks the host's
habits, and overlap with the host train passages.
"""

import glob
import json
import os
import shutil
import sys

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nthl.labels import LABELS

ROOT = "/kaggle/input/datasets"
COMP = "/kaggle/input/competitions/name-that-llm"
OUT = "/kaggle/working"
LABELSET = set(LABELS)


def read(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return None


def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def extract(df):
    wide = [c for c in df.columns if c in LABELSET]
    if len(wide) >= 5:
        return df[wide].melt(var_name="model", value_name="text").dropna()[["text", "model"]], "wide"
    shares = {c: df[c].dropna().astype(str).isin(LABELSET).mean() for c in df.columns if is_text(df[c])}
    shares = {c: s for c, s in shares.items() if s > 0.5}
    if not shares:
        return None, "no label column"
    label = max(shares, key=shares.get)
    texts = [c for c in df.columns if c != label and is_text(df[c])]
    if not texts:
        return None, "no text column"
    text = max(texts, key=lambda c: df[c].dropna().astype(str).str.len().mean())
    out = df[[text, label]].rename(columns={text: "text", label: "model"}).dropna()
    return out[out["model"].isin(LABELSET)], f"text={text} label={label}"


rows, report = [], []
for ds in sorted(glob.glob(f"{ROOT}/*/*")):
    name = "/".join(ds.split("/")[-2:])
    if name == "nabilshah/nthl-src":
        continue
    for path in sorted(glob.glob(f"{ds}/**/*", recursive=True)):
        if not os.path.isfile(path) or os.path.getsize(path) > 2e9:
            continue
        try:
            df = read(path)
        except Exception as e:
            report.append((name, path, f"unreadable: {type(e).__name__}", 0))
            continue
        if df is None:
            continue
        got, how = extract(df)
        report.append((name, path.replace(ds + "/", ""), how, 0 if got is None else len(got)))
        if got is not None and len(got):
            got = got.assign(dataset=name, file=path.replace(ds + "/", ""))
            rows.append(got)

pd.DataFrame(report, columns=["dataset", "file", "how", "rows"]).to_csv(f"{OUT}/files.csv", index=False)
pool = pd.concat(rows, ignore_index=True)
pool["text"] = pool["text"].astype(str)
pool["norm"] = pool["text"].str.split().str.join(" ")
pool = pool.drop_duplicates(["dataset", "norm", "model"])

train = pd.read_csv(f"{COMP}/train.csv")
train_norm = set(train["text"].str.split().str.join(" "))

words = pool["norm"].str.split().str.len()
curly = pool["text"].str.contains("[‘’“”]")
emdash = pool["text"].str.contains("—")
lab = pool["model"].str.split("-").str[0].str.rstrip("0123456789.")
pool = pool.assign(
    words=words,
    in_length=words.between(30, 300),
    newline=pool["text"].str.contains("\n"),
    markdown=pool["text"].str.contains(r"\*\*|^#|\n\s*[-*] ", regex=True),
    exact_train=pool["norm"].isin(train_norm),
    breaks_habit=((lab.isin(["claude", "glm"]) & curly) | ((lab == "gemini") & emdash)),
)

vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(train["text"])
tr = vec.transform(train["text"])
near = np.zeros(len(pool))
for start in range(0, len(pool), 20000):
    block = vec.transform(pool["text"].iloc[start:start + 20000])
    near[start:start + 20000] = (block @ tr.T).max(axis=1).toarray().ravel()
pool["near_train"] = near >= 0.3

pool["clean"] = pool["in_length"] & ~pool["newline"] & ~pool["markdown"] & ~pool["exact_train"] & ~pool["breaks_habit"] & ~pool["near_train"]

by_ds = pool.groupby("dataset").agg(
    rows=("text", "size"), models=("model", "nunique"), median_words=("words", "median"),
    in_length=("in_length", "mean"), newline=("newline", "mean"), markdown=("markdown", "mean"),
    exact_train=("exact_train", "mean"), near_train=("near_train", "mean"),
    breaks_habit=("breaks_habit", "mean"), clean=("clean", "sum"),
).round(3)
by_ds.to_csv(f"{OUT}/by_dataset.csv")

clean = pool[pool["clean"]].drop_duplicates(["norm", "model"])
dup_across = clean.duplicated("norm", keep=False).mean()
clean_u = clean.drop_duplicates("norm")
by_model = pd.DataFrame({
    "all_rows": pool.groupby("model").size(),
    "clean_unique": clean_u.groupby("model").size(),
}).reindex(list(LABELS)).fillna(0).astype(int)
by_model.to_csv(f"{OUT}/by_model.csv")
pd.crosstab(clean_u["model"], clean_u["dataset"]).reindex(list(LABELS)).fillna(0).astype(int).to_csv(f"{OUT}/model_by_dataset.csv")

# Each pool on its own (no cross-pool attribution), and the union without the
# two pools whose licence is not open.
pd.crosstab(pool.loc[pool["clean"], "model"], pool.loc[pool["clean"], "dataset"]).reindex(list(LABELS)).fillna(0).astype(int).to_csv(f"{OUT}/model_by_dataset_own.csv")
EXCLUDE = {"olegzholobov/name-that-llm-response-corpus", "ankiteshtiwari/ntl-curated-pool-routed"}
kept = clean[~clean["dataset"].isin(EXCLUDE)]
conflict = kept.groupby("norm")["model"].nunique()
kept = kept[kept["norm"].map(conflict) == 1].drop_duplicates("norm")
kept.groupby("model").size().reindex(list(LABELS)).fillna(0).astype(int).rename("open_licence_union").to_csv(f"{OUT}/open_union_by_model.csv")

summary = {
    "rows_read": int(len(pool)),
    "clean_rows": int(pool["clean"].sum()),
    "clean_unique_texts": int(len(clean_u)),
    "share_clean_texts_in_two_datasets": round(float(dup_across), 3),
    "texts_with_two_labels": int(clean.groupby("norm")["model"].nunique().gt(1).sum()),
    "open_licence_union": int(len(kept)),
    "open_licence_conflicts_dropped": int(conflict.gt(1).sum()),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(by_ds.to_string())
print(by_model.to_string())
