"""Port of the pre-registered preprocessing in google-deepmind/habermas_machine/analysis/live_loading.py:
  * keep participant-iteration rows with a human opinion and at least one non-mock rating/ranking;
  * drop groups (launch_ids) containing participants who did the task more than once (keep their richest instance);
  * keep only the first `num_groups` groups that have >= min_num_rounds rounds with >= min_num_iterations iterations
    of >= min_num_citizens participants (pre-registered sample sizes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREREG = {  # GroupMinSizeParameters in live_loading.py
    "EVAL_COHORT1_ABLATION_IID_V1": dict(min_num_citizens=4, min_num_iterations=2, min_num_rounds=3, num_groups=100),
    "EVAL_COHORT2_ABLATION_IID_V2": dict(min_num_citizens=4, min_num_iterations=2, min_num_rounds=3, num_groups=150),
    "EVAL_COHORT3_ABLATION_OOD_V1": dict(min_num_citizens=4, min_num_iterations=2, min_num_rounds=3, num_groups=100),
    "EVAL_COHORT4_CRITIQUE_EXCLUSION": dict(min_num_citizens=4, min_num_iterations=2, min_num_rounds=3, num_groups=50),
}
REMOVE_REPEATS = {"EVAL_COHORT1_ABLATION_IID_V1", "EVAL_COHORT2_ABLATION_IID_V2", "EVAL_COHORT3_ABLATION_OOD_V1",
                  "EVAL_COHORT4_CRITIQUE_EXCLUSION"}


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    ok_opinion = df["own_opinion.metadata.provenance"] == "HUMAN_CITIZEN"
    ok_rating = df["ratings.agreement"].apply(lambda l: l is not None and any(x != "MOCK" for x in l))
    ok_rank = df["rankings.numerical_ranks"].apply(lambda l: l is not None and any(x != -1 for x in l))
    return df[ok_opinion & ok_rating & ok_rank]


def filter_groups_with_repeat_participants(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("worker_id")["metadata.participant_id"].nunique()
    repeat_workers = counts[counts > 1].index
    rep = df[df["worker_id"].isin(repeat_workers)]
    inst = rep.groupby(["worker_id", "metadata.participant_id"]).agg(n_rows=("launch_id", "size"), ts=("monotonic_timestamp", "min")).reset_index()
    inst = inst[inst["n_rows"] > 0].sort_values(["worker_id", "n_rows", "ts"], ascending=[False, False, True])
    keep = inst.groupby("worker_id").first()["metadata.participant_id"]
    remove_instances = rep[~rep["metadata.participant_id"].isin(keep)]["metadata.participant_id"]
    bad_launches = df[df["metadata.participant_id"].isin(remove_instances)]["launch_id"].unique()
    return df[~df["launch_id"].isin(bad_launches)]


def filter_by_number_of_groups_of_min_size(df: pd.DataFrame, *, min_num_citizens=4, min_num_iterations=2, min_num_rounds=3, num_groups=100) -> pd.Series:
    """Return the launch_ids kept (mirrors live_loading, including the `head(num_groups)` after value_counts)."""
    c = df[["launch_id", "round_id", "iteration_index"]].value_counts().rename("count").reset_index()
    c = c[c["count"] >= min_num_citizens][["launch_id", "round_id"]].value_counts().rename("count").reset_index()
    c = c[c["count"] >= min_num_iterations]["launch_id"].value_counts().reset_index().set_axis(["launch_id", "count"], axis=1)
    groups = c[c["count"] >= min_num_rounds]["launch_id"].head(num_groups)
    return groups


def preregistered_launches(comps: pd.DataFrame) -> dict[str, np.ndarray]:
    """version -> array of launch_ids surviving the pre-registered preprocessing."""
    out = {}
    for version, params in PREREG.items():
        df = comps[comps["metadata.version"] == version]
        if not len(df):
            continue
        df = _valid_rows(df)
        if version in REMOVE_REPEATS:
            df = filter_groups_with_repeat_participants(df)
        out[version] = filter_by_number_of_groups_of_min_size(df, **params).values
    return out
