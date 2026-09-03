"""Load the public Habermas Machine dataset and build the tables needed for the Fig. 4C analysis.

Tables produced (all keyed on version/launch_id/round_id):
  opinions   : one row per participant-round (opinion text + pre-deliberation position rating)
  statements : one row per round (initial group statement = critiqued top candidate,
               revised group statement = candidate shown in the end-of-round survey)
  questions  : one row per question (text, affirming/negating position statements)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

LIKERT = {
    "STRONGLY_DISAGREE": 1, "DISAGREE": 2, "SOMEWHAT_DISAGREE": 3, "NEUTRAL": 4,
    "SOMEWHAT_AGREE": 5, "AGREE": 6, "STRONGLY_AGREE": 7,
}

COHORTS = {
    "cohort1": ["EVAL_COHORT1_ABLATION_IID_V1"],
    "cohort2": ["EVAL_COHORT2_ABLATION_IID_V2"],
    "cohort3": ["EVAL_COHORT3_ABLATION_OOD_V1"],
    "cohort4": ["EVAL_COHORT4_CRITIQUE_EXCLUSION"],
    "training": [f"TRAINING_DATA_V{i}" for i in range(1, 6)],
    "vca": [f"EVAL_VIRTUAL_CITIZENS_ASSEMBLY_WEEK{i}" for i in (3, 4, 5)],
}
COHORTS["cohorts_1_3"] = COHORTS["cohort1"] + COHORTS["cohort2"] + COHORTS["cohort3"]
# Priority order for embedding (main-task cohorts first).
EMBED_PRIORITY = ["cohort1", "cohort2", "cohort3", "cohort4", "training", "vca"]


def text_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def _version_to_cohort(version: str) -> str:
    for name in EMBED_PRIORITY:
        if version in COHORTS[name]:
            return name
    return "other"


def load_comparisons(data_dir: Path) -> pd.DataFrame:
    cols = [
        "metadata.version", "launch_id", "round_id", "iteration_index", "metadata.participant_id",
        "worker_id", "question.id", "question.text", "question.affirming_statement",
        "question.negating_statement", "question.topic", "question.split",
        "own_opinion.text", "own_opinion.metadata.id", "own_opinion.metadata.provenance",
        "top_candidate.text", "top_candidate.metadata.id", "top_candidate.metadata.provenance",
        "top_candidate.metadata.generative_model.api_version",
        "top_candidate.metadata.reward_model.api_version",
        "candidates.metadata.id", "candidates.text", "candidates.metadata.provenance",
        "critique.text", "critique.metadata.provenance",
    ]
    return pd.read_parquet(data_dir / "hm_all_candidate_comparisons.parquet", columns=cols)


def load_position_ratings(data_dir: Path) -> pd.DataFrame:
    ps = pd.read_parquet(
        data_dir / "hm_all_position_statement_ratings.parquet",
        columns=["metadata.version", "launch_id", "metadata.participant_id", "question.id",
                 "rating_index", "ratings.agreement", "metadata.provenance"],
    )
    ps = ps[ps["metadata.provenance"] == "HUMAN_CITIZEN"].copy()
    ps["rating"] = ps["ratings.agreement"].apply(lambda l: LIKERT.get(l[0], np.nan))
    ps = ps.dropna(subset=["rating"])
    ps["rating"] = ps["rating"].astype(int)
    return ps[["launch_id", "metadata.participant_id", "question.id", "rating_index", "rating"]]


def load_survey(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(
        data_dir / "hm_all_round_survey_responses.parquet",
        columns=["metadata.version", "metadata.participant_id", "question.id",
                 "candidate.metadata.id", "candidate.text", "candidate.metadata.provenance"],
    )


def build_tables(data_dir: Path, verbose: bool = True):
    data_dir = Path(data_dir)
    comps = load_comparisons(data_dir)
    comps = comps[comps["metadata.version"].apply(_version_to_cohort) != "other"]
    comps["cohort"] = comps["metadata.version"].apply(_version_to_cohort)
    key = ["metadata.version", "launch_id", "round_id"]

    # --- opinions: from iteration-0 rows (one row per human participant per round)
    i0 = comps[(comps["iteration_index"] == 0) & (comps["own_opinion.metadata.provenance"] == "HUMAN_CITIZEN")]
    opinions = i0[key + ["cohort", "metadata.participant_id", "worker_id", "question.id",
                         "own_opinion.metadata.id", "own_opinion.text"]].rename(columns={
        "metadata.participant_id": "participant_id", "question.id": "question_id",
        "own_opinion.metadata.id": "opinion_id", "own_opinion.text": "opinion_text"})
    dup = opinions.duplicated(key + ["participant_id"]).sum()
    assert dup == 0, f"{dup} duplicated participant-rounds"

    ratings = load_position_ratings(data_dir)
    pre = ratings[ratings["rating_index"] == 0].rename(columns={"rating": "pre_rating"})
    post = ratings[ratings["rating_index"] == 1].rename(columns={"rating": "post_rating"})
    on = ["launch_id", "metadata.participant_id", "question.id"]
    opinions = opinions.merge(pre[on + ["pre_rating"]].rename(columns={"metadata.participant_id": "participant_id", "question.id": "question_id"}),
                              on=["launch_id", "participant_id", "question_id"], how="left")
    opinions = opinions.merge(post[on + ["post_rating"]].rename(columns={"metadata.participant_id": "participant_id", "question.id": "question_id"}),
                              on=["launch_id", "participant_id", "question_id"], how="left")

    # --- initial group statement: the critiqued top candidate in iteration-1 rows
    i1 = comps[(comps["iteration_index"] == 1) & (comps["top_candidate.metadata.provenance"] == "MODEL_MEDIATOR")]
    n_unique = i1.groupby(key)["top_candidate.metadata.id"].nunique()
    assert (n_unique == 1).all(), "initial statement not unique within round"
    initial = i1.drop_duplicates(key)[key + ["question.id", "top_candidate.metadata.id", "top_candidate.text",
                                             "top_candidate.metadata.generative_model.api_version",
                                             "top_candidate.metadata.reward_model.api_version"]].rename(columns={
        "question.id": "question_id", "top_candidate.metadata.id": "initial_id", "top_candidate.text": "initial_text",
        "top_candidate.metadata.generative_model.api_version": "initial_gen_api",
        "top_candidate.metadata.reward_model.api_version": "initial_rm_api"})

    # --- revised group statement: the candidate shown in the end-of-round survey (unique per round)
    survey = load_survey(data_dir)
    survey = survey[survey["candidate.metadata.provenance"] == "MODEL_MEDIATOR"]
    pk = i0.drop_duplicates(["metadata.participant_id", "question.id"]).set_index(["metadata.participant_id", "question.id"])[key]
    sv = survey.join(pk, on=["metadata.participant_id", "question.id"], how="inner", rsuffix="_cmp")
    # sanity: the survey candidate must be among the iteration-1 candidates of that round
    c1 = comps[comps["iteration_index"] == 1].groupby(key)["candidates.metadata.id"].apply(lambda s: set(x for l in s for x in l))
    sv["in_iter1"] = [cid in c1.get(k, set()) for k, cid in zip(zip(sv["metadata.version"], sv["launch_id"], sv["round_id"]), sv["candidate.metadata.id"])]
    sv = sv[sv["in_iter1"]]
    n_unique = sv.groupby(key)["candidate.metadata.id"].nunique()
    assert (n_unique == 1).all(), "revised statement not unique within round"
    revised = sv.drop_duplicates(key)[key + ["candidate.metadata.id", "candidate.text"]].rename(columns={
        "candidate.metadata.id": "revised_id", "candidate.text": "revised_text"})

    statements = initial.merge(revised, on=key, how="outer")
    statements["cohort"] = statements["metadata.version"].apply(_version_to_cohort)

    questions = comps.drop_duplicates("question.id")[["question.id", "question.text", "question.affirming_statement",
                                                       "question.negating_statement", "question.topic", "question.split"]]
    questions.columns = ["question_id", "question_text", "affirming", "negating", "topic", "split"]

    if verbose:
        print("opinions:", opinions.shape, "| pre-rating coverage:", opinions["pre_rating"].notna().mean().round(4))
        print(opinions.groupby("cohort").size())
        print("statements:", statements.shape)
        print(statements.groupby("cohort")[["initial_id", "revised_id"]].apply(lambda g: g.notna().mean()).round(3))
        print("questions:", questions.shape)
    return opinions, statements, questions


def build_text_table(opinions: pd.DataFrame, statements: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """Unique texts to embed, with a priority (lower = embed first)."""
    prio = {c: i for i, c in enumerate(EMBED_PRIORITY)}
    rows = []
    for q in questions.itertuples():
        rows.append((q.affirming, "position", -1))
        rows.append((q.negating, "position", -1))
    for r in opinions.itertuples():
        rows.append((r.opinion_text, "opinion", prio.get(r.cohort, 99)))
    for r in statements.itertuples():
        if isinstance(r.initial_text, str):
            rows.append((r.initial_text, "initial", prio.get(r.cohort, 99)))
        if isinstance(r.revised_text, str):
            rows.append((r.revised_text, "revised", prio.get(r.cohort, 99)))
    t = pd.DataFrame(rows, columns=["text", "kind", "priority"])
    t["text_id"] = t["text"].apply(text_id)
    t = t.sort_values("priority").drop_duplicates("text_id")
    t["n_words"] = t["text"].str.split().str.len()
    return t[["text_id", "text", "kind", "priority", "n_words"]].reset_index(drop=True)


def prepare(data_dir: Path, out_dir: Path):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    opinions, statements, questions = build_tables(data_dir)
    texts = build_text_table(opinions, statements, questions)
    opinions.to_parquet(out_dir / "opinions.parquet", index=False)
    statements.to_parquet(out_dir / "statements.parquet", index=False)
    questions.to_parquet(out_dir / "questions.parquet", index=False)
    texts.to_parquet(out_dir / "texts.parquet", index=False)
    print("texts to embed:", len(texts)); print(texts.groupby(["priority", "kind"]).size())
    print("word-count quantiles:\n", texts.groupby("kind")["n_words"].describe(percentiles=[.5, .9, .99]).round(0))
    return opinions, statements, questions, texts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="prepared")
    a = ap.parse_args()
    prepare(Path(a.data_dir), Path(a.out_dir))
