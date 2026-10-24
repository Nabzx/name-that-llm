import pandas as pd

from nthl.pool import build_pool, clean_mask, extract

FILLER = " ".join(f"pool{i}" for i in range(40))
HOST = " ".join(f"host{i}" for i in range(40))


def test_extract_finds_text_and_label_columns():
    df = pd.DataFrame({"id": [1, 2], "prompt": ["q", "q"], "text": ["a long passage", "another one"], "model": ["kimi-k3", "glm-5.3"]})
    got = extract(df)
    assert list(got.columns) == ["text", "model"]
    assert got["model"].tolist() == ["kimi-k3", "glm-5.3"]


def test_extract_melts_a_wide_table():
    df = pd.DataFrame({"prompt": ["q"], "gpt-5.5": ["x"], "kimi-k3": ["y"], "glm-5.3": [None], "qwen3.8-max": ["z"], "claude-opus-5": ["w"], "gemini-3.6-flash": ["v"]})
    got = extract(df)
    assert sorted(got["model"]) == sorted(["gpt-5.5", "kimi-k3", "qwen3.8-max", "claude-opus-5", "gemini-3.6-flash"])


def test_extract_skips_a_table_with_no_labels():
    assert extract(pd.DataFrame({"prompt_id": ["g1"], "prompt": ["Explain tides."]})) is None


def test_clean_mask_drops_what_the_host_never_shows():
    train = pd.Series(["The host passage about tides and the moon " + HOST])
    pool = pd.DataFrame({
        "text": [
            "A fine passage " + FILLER,                       # kept
            "Too short.",                                     # under 30 words
            "Line one " + FILLER + "\nline two",              # line break
            "**Bold** " + FILLER,                             # markdown
            train[0],                                         # copy of a train passage
            "It’s Claude with a curly quote " + FILLER,       # Claude never uses curly quotes
            "Gemini with an em dash — " + FILLER,             # Gemini never uses em dashes
        ],
        "model": ["kimi-k3", "kimi-k3", "kimi-k3", "kimi-k3", "kimi-k3", "claude-opus-5", "gemini-3.6-flash"],
    })
    assert clean_mask(pool, train, near_min_df=1).tolist() == [True, False, False, False, False, False, False]


def write(root, ref, df):
    d = root / ref
    d.mkdir(parents=True)
    df.to_csv(d / "data.csv", index=False)


def test_build_pool_drops_two_label_texts_and_prefers_api_rows(tmp_path):
    shared = "A shared passage " + FILLER
    clash = "A passage two pools disagree on " + FILLER
    write(tmp_path, "a/api-set", pd.DataFrame({"text": [shared, clash], "model": ["kimi-k3", "kimi-k3"]}))
    write(tmp_path, "b/merge-set", pd.DataFrame({"text": [shared, clash, "Only here " + FILLER], "model": ["kimi-k3", "glm-5.3", "qwen3.8-max"]}))
    train = pd.DataFrame({"text": ["Unrelated host text about volcanoes " + HOST]})
    pool = build_pool(tmp_path, train, routes={"a/api-set": "api", "b/merge-set": "merge"}, near_min_df=1)
    assert list(pool.columns) == ["text", "model", "source", "route"]
    assert len(pool) == 2
    assert pool.set_index("model").loc["kimi-k3", "route"] == "api"
    assert clash not in set(pool["text"])
    assert pool.attrs["loaded_rows"] == {"a/api-set": 2, "b/merge-set": 3}


def test_extract_ignores_a_column_of_dicts():
    df = pd.DataFrame({
        "settings": [{"temperature": 1.0, "notes": "x" * 500}, {"temperature": 0.7, "notes": "y" * 500}],
        "text": ["a passage", "another passage"],
        "model": ["kimi-k3", "glm-5.3"],
    })
    got = extract(df)
    assert got["text"].tolist() == ["a passage", "another passage"]
