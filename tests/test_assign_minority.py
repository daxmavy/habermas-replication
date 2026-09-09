"""The minority/non-minority split of SM 5.4.1, and the tie rule of SM 4.1.2.1.

A "round" here is one group answering one question. Each participant gives a pre-deliberation
Likert rating on 1-7; 4 is neutral. These tests pin the three decisions that turn those ratings
into a minority set, because the paper states them in two different sections and the wrong
reading silently changes the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hm_fig4c.analysis import assign_minority

KEY = ["metadata.version", "launch_id", "round_id"]


def round_of(*ratings: int, launch: str = "L1") -> pd.DataFrame:
    """One round whose participants gave `ratings`, in a frame shaped like prepared/opinions.parquet."""
    return pd.DataFrame({
        "metadata.version": "EVAL_COHORT1_ABLATION_IID_V1",
        "launch_id": launch,
        "round_id": "R1",
        "participant_id": [f"p{i}" for i in range(len(ratings))],
        "pre_rating": list(ratings),
        "score": np.linspace(-1, 1, len(ratings)),
    })


def test_tied_round_is_kept_with_the_disagree_side_as_minority():
    """SM 4.1.2.1: "In the case of a tie ... we arbitrarily set the majority direction to AGREE."

    So a 2-agree/2-disagree round has a majority (AGREE) and a minority (DISAGREE); it is not
    a round "without a majority" and is not dropped.
    """
    out = assign_minority(round_of(6, 6, 2, 2), neutral="as_majority", ties="majority_agree")

    assert len(out) == 4, "the tied round should survive"
    assert out.loc[out["pre_rating"] < 4, "is_minority"].all(), "the disagree side is the minority"
    assert not out.loc[out["pre_rating"] > 4, "is_minority"].any(), "the agree side is the majority"
    assert out["k_min"].unique().tolist() == [2]
    assert out["n_div"].unique().tolist() == [4]


def test_mirror_rule_puts_the_agree_side_in_the_minority():
    """The mirror of the SM's arbitrary choice, kept so the sensitivity analysis can sweep it."""
    out = assign_minority(round_of(6, 6, 2, 2), neutral="as_majority", ties="majority_disagree")

    assert out.loc[out["pre_rating"] > 4, "is_minority"].all()


def test_exclude_drops_the_tied_round():
    out = assign_minority(round_of(6, 6, 2, 2), neutral="as_majority", ties="exclude")

    assert len(out) == 0


def test_neutral_ratings_count_as_non_minority_and_stay_in_the_group():
    """SM 5.4.1 counts a neutral (4) opinion as non-minority, so it enlarges n but never k."""
    out = assign_minority(round_of(6, 6, 6, 2, 4), neutral="as_majority", ties="majority_agree")

    assert out["n_div"].unique().tolist() == [5], "the neutral stays in the group"
    assert out["k_min"].unique().tolist() == [1], "only the lone disagreer is the minority"
    assert not out.loc[out["pre_rating"] == 4, "is_minority"].any()


def test_tie_among_sided_ratings_is_still_a_tie_when_a_neutral_is_present():
    """SM 4.1.2.1's own example: 2 agree, 2 disagree, 1 neutral."""
    out = assign_minority(round_of(6, 6, 2, 2, 4), neutral="as_majority", ties="majority_agree")

    assert out["n_div"].unique().tolist() == [5]
    assert out["k_min"].unique().tolist() == [2]
    assert out.loc[out["pre_rating"] < 4, "is_minority"].all()


def test_unanimous_round_is_dropped_under_every_tie_rule():
    """No dissent means no minority to weight, whatever the tie rule says."""
    for ties in ("majority_agree", "majority_disagree", "exclude"):
        assert len(assign_minority(round_of(6, 6, 5, 7), neutral="as_majority", ties=ties)) == 0, ties


def test_unknown_tie_rule_is_rejected():
    with pytest.raises(ValueError):
        assign_minority(round_of(6, 6, 2, 2), neutral="as_majority", ties="agree")
