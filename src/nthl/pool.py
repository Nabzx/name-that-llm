"""One training pool from the public passage datasets (#19).

Reads every table in the open-licence pools, keeps rows labelled with one of
the 18 models and drops what the host never shows: under 30 or over 300
words, line breaks, markdown, copies or near copies of a train passage, and
punctuation a model never uses in train. Texts that two pools label
differently are dropped. A text found in several pools keeps its most direct
source, so a row from an official API wins over the same row in a merge.
"""

import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nthl.labels import LABELS

LABELSET = set(LABELS)
NEAR_TRAIN = 0.3

# How each open-licence pool was made. olegzholobov and ankiteshtiwari are
# left out on purpose: licence, chat format and conflicting labels (#19).
ROUTES = {
    "seyitkaangunes/ntl-extra-api-passages": "api",
    "lxdlxdlxd/ntl-synth-generations-v1": "api",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset": "api",
    "willrice/name-that-llm-openrouter-harvest": "openrouter",
    "maniceet/name-that-llm-openrouter-openai-generations": "openrouter",
    "alperen5252525/ntl-vertex-gemini-proxy-gpt-generations": "proxy",
    "prvsiyan/ntl-own-prompt-generations-v1": "proxy",
    "haileycheng/name-that-llm-same-prompt-passages": "proxy",
    "kaplinskiy/ntl-gpt-proxy-passages": "proxy",
    "lucasboesen/ntl-free-proxy-generations": "proxy",
    "abdullah1afify/ntl-generations": "proxy",
    "dophuchung/ntl-proxy-generations": "proxy",
    "ziyuzhou1207/ntl-zz-proxy-generations": "proxy",
    "keshavatearth/ntl-gpt-new-prompts-v1": "proxy",
    "darragh1walsh/ntl-kaggle-ai-gen-run-1": "proxy",
    "zbigniewkirchner/ntl-extra-gpt-passages": "unknown",
    "michaelsemenoff/ntl-own-extra-v1": "merge",
    "nishantdahal/ntl-handwriting-blend": "merge",
    "prvsiyan/ntl-public-in-style-pool-v1": "merge",
}
ROUTE_RANK = {"api": 0, "openrouter": 1, "proxy": 2, "unknown": 3, "merge": 4}


def read_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return None


def is_text(s):
    """True if every non-null value is a real string, not a dict or list."""
    values = s.dropna()
    return len(values) > 0 and values.map(lambda v: isinstance(v, str)).all()


def extract(df):
    """The (text, model) rows of a table, or None if it has no label column."""
    wide = [c for c in df.columns if c in LABELSET]
    if len(wide) >= 5:
        return df[wide].melt(var_name="model", value_name="text").dropna()[["text", "model"]].reset_index(drop=True)
    shares = {c: df[c].dropna().isin(LABELSET).mean() for c in df.columns if is_text(df[c])}
    shares = {c: s for c, s in shares.items() if s > 0.5}
    if not shares:
        return None
    label = max(shares, key=shares.get)
    texts = [c for c in df.columns if c != label and is_text(df[c])]
    if not texts:
        return None
    text = max(texts, key=lambda c: df[c].dropna().str.len().mean())
    out = df[[text, label]].rename(columns={text: "text", label: "model"}).dropna()
    return out[out["model"].isin(LABELSET)].reset_index(drop=True)


def normalise(texts):
    return texts.astype(str).str.split().str.join(" ")


def clean_mask(pool, train_texts, near_min_df=2):
    text = pool["text"].astype(str)
    words = text.str.split().str.len()
    lab = pool["model"].str.split("-").str[0].str.rstrip("0123456789.")
    breaks_habit = (lab.isin(["claude", "glm"]) & text.str.contains("[‘’“”]")) | ((lab == "gemini") & text.str.contains("—"))
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=near_min_df).fit(train_texts)
    near = (vec.transform(text) @ vec.transform(train_texts).T).max(axis=1).toarray().ravel() >= NEAR_TRAIN
    return (
        words.between(30, 300)
        & ~text.str.contains("\n")
        & ~text.str.contains(r"\*\*|^#", regex=True)
        & ~normalise(text).isin(set(normalise(pd.Series(train_texts))))
        & ~breaks_habit
        & ~near
    )


def load_pools(root, routes=ROUTES):
    frames = []
    for ref, route in routes.items():
        for path in sorted(glob.glob(os.path.join(str(root), ref, "**", "*"), recursive=True)):
            if not os.path.isfile(path):
                continue
            table = read_table(path)
            got = None if table is None else extract(table)
            if got is not None and len(got):
                frames.append(got.assign(source=ref, route=route))
    return pd.concat(frames, ignore_index=True)


def build_pool(root, train, routes=ROUTES, near_min_df=2):
    raw = load_pools(root, routes)
    pool = raw[clean_mask(raw, train["text"], near_min_df).to_numpy()].copy()
    pool["norm"] = normalise(pool["text"])
    labels_per_text = pool.groupby("norm")["model"].nunique()
    pool = pool[pool["norm"].map(labels_per_text) == 1]
    pool["rank"] = pool["route"].map(ROUTE_RANK)
    pool = pool.sort_values(["rank", "source", "norm"], kind="stable").drop_duplicates("norm")
    pool = pool[["text", "model", "source", "route"]].sort_values(["model", "source", "text"]).reset_index(drop=True)
    pool.attrs["loaded_rows"] = {ref: int(n) for ref, n in raw["source"].value_counts().sort_index().items()}
    return pool
