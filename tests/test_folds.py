from pathlib import Path

import pandas as pd
import pytest

from nthl.folds import FOLDS_CSV, N_FOLDS, build_folds, load_folds

TRAIN = Path(__file__).resolve().parent.parent / "data" / "raw" / "train.csv"


def test_no_prompt_sits_in_two_folds():
    folds = load_folds()
    assert folds.groupby("prompt_id")["fold"].nunique().max() == 1


def test_every_passage_has_exactly_one_fold():
    folds = load_folds()
    assert folds["id"].is_unique
    assert set(folds["fold"]) == set(range(N_FOLDS))


def test_folds_hold_the_same_number_of_prompts():
    per_fold = load_folds().groupby("fold")["prompt_id"].nunique()
    assert per_fold.max() - per_fold.min() <= 1


@pytest.mark.skipif(not TRAIN.exists(), reason="needs data/raw/train.csv")
def test_frozen_file_matches_a_fresh_build():
    train = pd.read_csv(TRAIN)
    fresh = build_folds(train)
    frozen = load_folds()
    assert len(frozen) == len(train)
    pd.testing.assert_frame_equal(
        fresh.sort_values("id").reset_index(drop=True),
        frozen.sort_values("id").reset_index(drop=True),
    )


@pytest.mark.skipif(not TRAIN.exists(), reason="needs data/raw/train.csv")
def test_every_model_appears_in_every_validation_fold():
    train = pd.read_csv(TRAIN).merge(load_folds()[["id", "fold"]], on="id")
    counts = train.groupby(["fold", "model"]).size().unstack()
    assert counts.shape[1] == 18
    assert counts.min().min() >= 29


def test_frozen_file_lives_next_to_the_code():
    assert FOLDS_CSV.name == "folds.csv"
    assert FOLDS_CSV.parent.name == "nthl"
