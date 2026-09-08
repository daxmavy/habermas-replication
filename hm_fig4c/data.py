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


AFFIRM_PREFIX, NEGATE_PREFIX = "Yes, I agree. ", "No, I disagree. "

# Readings of SM 5.1.2's chosen endpoints, "a generic + question-specific combination (i.e. 'Yes, I agree. It is the
# government's role to [...]')".  A style maps a question's affirming/negating statements to the text whose embedding
# is the endpoint; the first two are consistent with that description, the last two are SM options it did not choose.
ENDPOINT_STYLES = {
    "prefixed": "generic phrase and released statement as one text, as in the SM's example (pinned)",
    "prefixed_not_lower": "as pinned, with the released statements' capitalised 'NOT' lowercased, as in the SM's example",
    "plain": "released statement only",
    "generic": "generic phrase only",
}


def endpoint_texts(affirming: str, negating: str, style: str = "prefixed") -> tuple[str, str]:
    """Texts whose embeddings are the affirming and negating endpoints under `style` (see ENDPOINT_STYLES)."""
    a, n = affirming.strip(), negating.strip()
    if style == "prefixed":
        return AFFIRM_PREFIX + a, NEGATE_PREFIX + n
    if style == "prefixed_not_lower":
        return AFFIRM_PREFIX + a.replace("NOT", "not"), NEGATE_PREFIX + n.replace("NOT", "not")
    if style == "plain":
        return a, n
    if style == "generic":
        return AFFIRM_PREFIX.strip(), NEGATE_PREFIX.strip()
    raise ValueError(style)


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
        "critique.text", "critique.metadata.provenance", "monotonic_timestamp", "ratings.agreement", "rankings.numerical_ranks",
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
    from .preprocess import preregistered_launches
    prereg = preregistered_launches(comps)
    prereg_set = {(v, l) for v, ls in prereg.items() for l in ls}
    comps["prereg"] = [(v, l) in prereg_set for v, l in zip(comps["metadata.version"], comps["launch_id"])]

    # --- opinions: from iteration-0 rows (one row per human participant per round)
    i0 = comps[(comps["iteration_index"] == 0) & (comps["own_opinion.metadata.provenance"] == "HUMAN_CITIZEN")]
    opinions = i0[key + ["cohort", "prereg", "metadata.participant_id", "worker_id", "question.id",
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

    # --- all candidate statements shown at each iteration (same list for every participant in a round-iteration)
    cand_rows = []
    ci = comps[comps["iteration_index"].isin([0, 1])].drop_duplicates(key + ["iteration_index"])
    for v, l, rd, it, ids, texts_, provs in zip(ci["metadata.version"], ci["launch_id"], ci["round_id"], ci["iteration_index"],
                                                ci["candidates.metadata.id"], ci["candidates.text"], ci["candidates.metadata.provenance"]):
        for cid, ctext, cprov in zip(ids, texts_, provs):
            if cprov == "MODEL_MEDIATOR":
                cand_rows.append((v, l, rd, int(it), cid, ctext))
    candidates = pd.DataFrame(cand_rows, columns=key + ["iteration_index", "candidate_id", "candidate_text"]).drop_duplicates(key + ["iteration_index", "candidate_id"])
    # check the candidate list is identical across participants of a round-iteration
    n_lists = comps[comps["iteration_index"].isin([0, 1])].groupby(key + ["iteration_index"])["candidates.metadata.id"].apply(lambda s: len({tuple(sorted(x)) for x in s}))
    assert (n_lists == 1).mean() > 0.99, "candidate lists differ across participants"

    statements = initial.merge(revised, on=key, how="outer")
    statements["cohort"] = statements["metadata.version"].apply(_version_to_cohort)
    statements["prereg"] = [(v, l) in prereg_set for v, l in zip(statements["metadata.version"], statements["launch_id"])]

    candidates = candidates.merge(statements[key + ["initial_id", "revised_id", "cohort", "prereg"]], on=key, how="inner")
    candidates["phase"] = np.where(candidates["iteration_index"] == 0, "initial", "revised")
    candidates["is_winner"] = np.where(candidates["iteration_index"] == 0, candidates["candidate_id"] == candidates["initial_id"],
                                       candidates["candidate_id"] == candidates["revised_id"])
    candidates = candidates.drop(columns=["initial_id", "revised_id"])

    questions = comps.drop_duplicates("question.id")[["question.id", "question.text", "question.affirming_statement",
                                                       "question.negating_statement", "question.topic", "question.split"]]
    questions.columns = ["question_id", "question_text", "affirming", "negating", "topic", "split"]

    if verbose:
        print("opinions:", opinions.shape, "| pre-rating coverage:", opinions["pre_rating"].notna().mean().round(4))
        print(opinions.groupby("cohort").size())
        print("statements:", statements.shape)
        print("pre-registered groups / rounds per version:\n", statements[statements["prereg"]].groupby("metadata.version").agg(groups=("launch_id", "nunique"), rounds=("round_id", "size")))
        print(statements.groupby("cohort")[["initial_id", "revised_id"]].apply(lambda g: g.notna().mean()).round(3))
        print("questions:", questions.shape)
        print("candidates:", candidates.shape, "| per round-iteration:", candidates.groupby(key + ["iteration_index"]).size().value_counts().to_dict())
        print("winner found among candidates:", candidates.groupby(key + ["phase"])["is_winner"].any().mean().round(4))
    return opinions, statements, questions, candidates


def build_text_table(opinions: pd.DataFrame, statements: pd.DataFrame, questions: pd.DataFrame, candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Unique texts to embed, with a priority (lower = embed first)."""
    prio = {c: i for i, c in enumerate(EMBED_PRIORITY)}
    prereg_q = set(statements.loc[statements["prereg"], "question_id"])  # questions used by pre-registered rounds
    rows = []
    for q in questions.itertuples():
        pq = q.question_id in prereg_q
        for style, kind, prio_ in (("prefixed", "position_prefixed", -2), ("prefixed_not_lower", "position_prefixed_not_lower", -2),
                                   ("plain", "position", -1)):
            for t in endpoint_texts(q.affirming, q.negating, style):
                rows.append((t, kind, prio_, pq))
    rows.append((AFFIRM_PREFIX.strip(), "position_generic", -2, True)); rows.append((NEGATE_PREFIX.strip(), "position_generic", -2, True))
    for r in opinions.itertuples():
        rows.append((r.opinion_text, "opinion", prio.get(r.cohort, 99), bool(r.prereg)))
    for r in statements.itertuples():
        if isinstance(r.initial_text, str):
            rows.append((r.initial_text, "initial", prio.get(r.cohort, 99), bool(r.prereg)))
        if isinstance(r.revised_text, str):
            rows.append((r.revised_text, "revised", prio.get(r.cohort, 99), bool(r.prereg)))
    if candidates is not None:  # non-winning candidates: right after the main cohorts' winners, or last for other cohorts
        for r in candidates[~candidates["is_winner"]].itertuples():
            base = prio.get(r.cohort, 99)
            rows.append((r.candidate_text, "candidate", (3 if r.prereg else 6) if base <= 2 else base + 3, bool(r.prereg)))
    t = pd.DataFrame(rows, columns=["text", "kind", "priority", "prereg"])
    t["text_id"] = t["text"].apply(text_id)
    t = t.sort_values(["priority", "prereg"], ascending=[True, False]).drop_duplicates("text_id")
    t["n_words"] = t["text"].str.split().str.len()
    return t[["text_id", "text", "kind", "priority", "prereg", "n_words"]].reset_index(drop=True)


def prepare(data_dir: Path, out_dir: Path):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    opinions, statements, questions, candidates = build_tables(data_dir)
    texts = build_text_table(opinions, statements, questions, candidates)
    opinions.to_parquet(out_dir / "opinions.parquet", index=False)
    candidates.to_parquet(out_dir / "candidates.parquet", index=False)
    statements.to_parquet(out_dir / "statements.parquet", index=False)
    questions.to_parquet(out_dir / "questions.parquet", index=False)
    texts.to_parquet(out_dir / "texts.parquet", index=False)
    print("texts to embed:", len(texts)); print(texts.groupby(["priority", "kind"]).size())
    print("word-count quantiles:\n", texts.groupby("kind")["n_words"].describe(percentiles=[.5, .9, .99]).round(0))
    return opinions, statements, questions, candidates, texts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="prepared")
    a = ap.parse_args()
    prepare(Path(a.data_dir), Path(a.out_dir))
