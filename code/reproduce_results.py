#!/usr/bin/env python3
"""Reproduce quantitative results in the revised manuscript and Supplementary Information.

Run from anywhere: python open_data/code/reproduce_results.py
Only the six CSV files in open_data/data are read. No credentials, network
requests, raw Discord exports, or identifiable questionnaire files are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.metrics import cohen_kappa_score
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
RNG_SEED = 20261008
BOOTSTRAPS = 5000

ITEMS = ["id_with_group", "member_selfview", "identity_importance", "criticism_as_self"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    people = pd.read_csv(DATA / "participants.csv")
    messages = pd.read_csv(DATA / "human_message_times.csv")
    validation = pd.read_csv(DATA / "team_impression_validation.csv")
    agents = pd.read_csv(DATA / "agent_activity.csv")
    words = pd.read_csv(DATA / "neologism_ratings.csv")
    legacy_events = pd.read_csv(DATA / "legacy_user_events.csv")
    assert len(people) == people.participant_id.nunique() == 100
    assert people.proposal_wave.value_counts().to_dict() == {0: 50, 1: 50}
    assert len(messages) == 2094 and set(messages.participant_id).issubset(set(people.participant_id))
    assert len(validation) == validation.case_id.nunique() == 50
    assert len(agents) == 400
    assert len(words) == words.word_id.nunique() == 50
    assert len(legacy_events) == 2438
    return people, messages, validation, agents, words, legacy_events


def verify_validation_sample(people: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Recreate the 25-per-wave draw and blinded case assignment."""
    rng = np.random.default_rng(RNG_SEED)
    selected = []
    for wave in (1, 0):  # condition labels sorted as neologism, no_neologism
        part = people.loc[people.proposal_wave.eq(wave)].sort_values("validation_sampling_order")
        assert part.validation_sampling_order.tolist() == list(range(1, 51))
        chosen = rng.choice(part.index.to_numpy(), size=25, replace=False)
        selected.extend(chosen.tolist())
    selected = np.asarray(selected)[rng.permutation(50)]
    expected = pd.Series([f"HV{i:03d}" for i in range(1, 51)], index=selected)
    actual = people.validation_case_id.dropna()
    assert len(actual) == 50 and expected.sort_index().equals(actual.sort_index())
    linked = people.loc[selected, ["validation_case_id", "team_impression_end_score", "proposal_wave"]].merge(
        validation[["case_id", "original_llm_score", "condition"]],
        left_on="validation_case_id", right_on="case_id", validate="one_to_one")
    assert np.allclose(linked.team_impression_end_score, linked.original_llm_score)
    assert (linked.proposal_wave.eq(1) == linked.condition.eq("neologism")).all()
    return pd.DataFrame([{"sample_n": len(linked), "proposal_n": int(linked.proposal_wave.sum()),
                          "comparison_n": int(linked.proposal_wave.eq(0).sum()),
                          "case_and_score_links_verified": True,
                          "sampling_seed": RNG_SEED}])


def verify_input_consistency(people: pd.DataFrame, messages: pd.DataFrame,
                             agents: pd.DataFrame, words: pd.DataFrame,
                             legacy_events: pd.DataFrame) -> None:
    item_columns = [f"{item}_{occasion}" for item in ITEMS for occasion in ("middle", "end")]
    assert people[item_columns].notna().all().all()
    assert people[item_columns].apply(lambda col: col.between(1, 5) & col.eq(col.round())).all().all()
    assert people.team_impression_end_score.between(0, 1).all()
    proposal = people.loc[people.proposal_wave.eq(1)].copy()
    comparison = people.loc[people.proposal_wave.eq(0)]
    assert proposal[["word_use_intention", "word_enjoyment"]].notna().all().all()
    assert comparison[["word_use_intention", "word_enjoyment"]].isna().all().all()
    assert (people.within_discussion_message_count + people.post_end_message_count
            == people.extended_window_message_count).all()
    counts = messages.groupby("participant_id").size()
    assert (people.set_index("participant_id").within_discussion_message_count
            == counts.reindex(people.participant_id, fill_value=0).to_numpy()).all()
    durations = people.set_index("participant_id").discussion_elapsed_minutes
    assert messages.minutes_from_start.ge(0).all()
    assert messages.minutes_from_start.le(messages.participant_id.map(durations)).all()
    proposal_messages = messages.merge(
        proposal[["participant_id", "proposal_minutes_from_start"]], on="participant_id", how="inner"
    )
    before = proposal_messages.loc[proposal_messages.minutes_from_start.lt(
        proposal_messages.proposal_minutes_from_start)].groupby("participant_id").size()
    after = proposal_messages.loc[proposal_messages.minutes_from_start.gt(
        proposal_messages.proposal_minutes_from_start)].groupby("participant_id").size()
    assert (proposal.set_index("participant_id").messages_pre_proposal
            == before.reindex(proposal.participant_id, fill_value=0).to_numpy()).all()
    assert (proposal.set_index("participant_id").messages_post_proposal
            == after.reindex(proposal.participant_id, fill_value=0).to_numpy()).all()
    assert agents.groupby("participant_id").size().reindex(people.participant_id).eq(4).all()
    participant_ratings = proposal.groupby(["word_use_intention", "word_enjoyment"]).size()
    word_ratings = words.groupby(["word_use_intention", "word_enjoyment"]).size()
    assert participant_ratings.equals(word_ratings)
    assert set(legacy_events.author_is_reply_marker.unique()) == {0, 1}
    counted = legacy_events.loc[legacy_events.author_is_reply_marker.eq(0)].groupby("participant_id").size()
    indexed = people.set_index("participant_id")
    assert (indexed.loc[counted.index].total_message_count == counted).all()
    assert indexed.loc[counted.index].total_count_source.eq("json_user_nonreply_events").all()
    recovered = indexed.loc[~indexed.index.isin(counted.index)]
    assert len(counted) == 99 and len(recovered) == 1
    assert recovered.total_count_source.eq("recovered_workbook").all()
    assert recovered.total_message_count.iloc[0] == 34


def prepare(people: pd.DataFrame) -> pd.DataFrame:
    people = people.copy()
    for count in (3, 4):
        for occasion in ("middle", "end"):
            columns = [f"{item}_{occasion}" for item in ITEMS[:count]]
            people[f"identity{count}_{occasion}_sum"] = people[columns].sum(axis=1)
            people[f"identity{count}_{occasion}_mean"] = people[columns].mean(axis=1)
        for scale in ("sum", "mean"):
            people[f"identity{count}_change_{scale}"] = (
                people[f"identity{count}_end_{scale}"] - people[f"identity{count}_middle_{scale}"]
            )
    people["neologism_evaluation_sum"] = people[["word_use_intention", "word_enjoyment"]].sum(
        axis=1, min_count=2
    )
    people["age_collapsed"] = people.age_group.map({
        "20s": "20–39", "30s": "20–39", "40s": "40–49", "50s": "50+", "60plus": "50+",
    })
    people["gender_collapsed"] = people.gender.replace({"not_reported": "other_or_no_answer"})
    people["education_collapsed"] = np.where(
        people.university_or_higher.eq(1), "university_or_higher", "below_university_or_other"
    )
    for column in (
        "total_message_count", "within_discussion_message_count", "extended_window_message_count",
        "team_impression_end_score", "identity4_middle_mean",
    ):
        people[f"{column}_c"] = people[column] - people[column].mean()
    assert people[[f"identity4_{x}_mean" for x in ("middle", "end")]].notna().all().all()
    assert people[["total_message_count", "team_impression_end_score"]].notna().all().all()
    return people


def model_rows(label: str, formula: str, data: pd.DataFrame) -> tuple[object, list[dict]]:
    fit = ols(formula, data=data).fit(cov_type="HC3")
    ci = fit.conf_int()
    y_sd = np.std(fit.model.endog, ddof=1)
    rows = []
    for j, term in enumerate(fit.params.index):
        x_sd = np.std(fit.model.exog[:, j], ddof=1)
        factor = x_sd / y_sd if term != "Intercept" and y_sd else np.nan
        rows.append({
            "model": label, "formula": formula, "term": term, "n": int(fit.nobs),
            "b": float(fit.params.iloc[j]), "beta_std": float(fit.params.iloc[j] * factor)
            if term != "Intercept" else np.nan,
            "beta_std_se": float(fit.bse.iloc[j] * factor) if term != "Intercept" else np.nan,
            "beta_std_ci_low": float(ci.iloc[j, 0] * factor) if term != "Intercept" else np.nan,
            "beta_std_ci_high": float(ci.iloc[j, 1] * factor) if term != "Intercept" else np.nan,
            "hc3_se": float(fit.bse.iloc[j]), "ci_low": float(ci.iloc[j, 0]),
            "ci_high": float(ci.iloc[j, 1]), "p": float(fit.pvalues.iloc[j]),
            "r_squared": float(fit.rsquared), "adjusted_r_squared": float(fit.rsquared_adj),
        })
    return fit, rows


def fit_all(people: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = "proposal_wave"
    m = "total_message_count_c"
    s = "team_impression_end_score_c"
    b = "identity4_middle_mean"
    specs = [
        ("primary_ancova", "primary", f"identity4_end_mean ~ {w} + {b}", [w], None),
        ("unadjusted_post", "secondary", f"identity4_end_mean ~ {w}", [w], None),
        ("change_wave", "secondary", f"identity4_change_mean ~ {w}", [w], None),
        ("wave_message_change", "interaction", f"identity4_change_sum ~ {w} * {m}", [f"{w}:{m}"], None),
        ("wave_impression_change", "interaction", f"identity4_change_sum ~ {w} * {s}", [f"{w}:{s}"], None),
        ("wave_message_impression_change", "interaction", f"identity4_change_sum ~ {w} * {m} * {s}", [f"{w}:{m}:{s}"], None),
        ("three_item_ancova", "sensitivity", f"identity3_end_mean ~ {w} + identity3_middle_mean", [w], None),
        ("wave_message_adjusted", "sensitivity", f"identity4_end_mean ~ {b} + {w} * {m}", [f"{w}:{m}"], None),
        ("wave_impression_adjusted", "sensitivity", f"identity4_end_mean ~ {b} + {w} * {s}", [f"{w}:{s}"], None),
        ("wave_message_impression_adjusted", "sensitivity", f"identity4_end_mean ~ {b} + {w} * {m} * {s}", [f"{w}:{m}:{s}"], None),
        ("word_evaluation", "other_exploratory", "identity4_change_sum ~ neologism_evaluation_sum", ["neologism_evaluation_sum"], "proposal"),
        ("wave_midpoint_identity", "other_exploratory", f"identity4_end_mean ~ {w} * identity4_middle_mean_c", [f"{w}:identity4_middle_mean_c"], None),
    ]
    for variant in ("within_discussion_message_count_c", "extended_window_message_count_c"):
        tag = "within" if variant.startswith("within") else "extended"
        specs.extend([
            (f"wave_{tag}_count", "sensitivity", f"identity4_change_sum ~ {w} * {variant}", [f"{w}:{variant}"], None),
            (f"wave_{tag}_three_way", "sensitivity", f"identity4_change_sum ~ {w} * {variant} * {s}", [f"{w}:{variant}:{s}"], None),
        ])
    specs.append(("ancova_with_team_impression", "sensitivity",
                  f"identity4_end_mean ~ {w} + {b} + {s}", [w, s], None))
    for variable, expression in (
        ("age", "C(age_collapsed)"),
        ("gender", "C(gender_collapsed)"),
        ("education", "C(education_collapsed)"),
    ):
        specs.append((f"wave_{variable}", "other_exploratory",
                      f"identity4_end_mean ~ {b} + {w} * {expression}", ["interaction_terms"], None))
    for label, variable in (
        ("broad_uptake_any", "manual_uptake_any"),
        ("substantive_use_any", "manual_substantive_uptake_any"),
        ("broad_uptake_count", "manual_uptake_count"),
        ("substantive_use_count", "manual_substantive_uptake_count"),
    ):
        specs.append((label, "other_exploratory", f"identity4_change_mean ~ {variable}", [variable], "proposal"))

    rows, focal = [], []
    for label, family, formula, terms, subset in specs:
        frame = people.loc[people.proposal_wave.eq(1)] if subset == "proposal" else people
        fit, model = model_rows(label, formula, frame)
        rows.extend(model)
        if terms == ["interaction_terms"]:
            terms = [t for t in fit.params.index if t.startswith(f"{w}:")]
        for term in terms:
            focal.append(next({"test": label + ": " + term, "family": family, **r}
                              for r in model if r["term"] == term))

    # Within-wave interaction estimates are descriptive, outside the 29 focal tests.
    for wave in (0, 1):
        frame = people.loc[people.proposal_wave.eq(wave)].copy()
        frame["message_within_wave_c"] = frame.total_message_count - frame.total_message_count.mean()
        frame["impression_within_wave_c"] = frame.team_impression_end_score - frame.team_impression_end_score.mean()
        _, model = model_rows(f"within_wave_{wave}_message_impression",
                              "identity4_change_sum ~ message_within_wave_c * impression_within_wave_c", frame)
        rows.extend(model)
    return pd.DataFrame(rows), pd.DataFrame(focal)


def timing(people: pd.DataFrame, messages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    proposal = people.loc[people.proposal_wave.eq(1)].copy().sort_values("participant_id")
    control = people.loc[people.proposal_wave.eq(0)].copy().sort_values("participant_id")
    pre = proposal.messages_pre_proposal / proposal.proposal_minutes_from_start
    post = proposal.messages_post_proposal / (
        proposal.discussion_elapsed_minutes - proposal.proposal_minutes_from_start
    )
    changes = (post - pre).to_numpy(float)
    paired = stats.ttest_rel(post, pre)
    wilcoxon = stats.wilcoxon(post, pre)
    assert len(changes) == 50
    fractions = (proposal.proposal_minutes_from_start / proposal.discussion_elapsed_minutes).clip(.001, .999).to_numpy()
    human_times = {pid: part.minutes_from_start.to_numpy(float)
                   for pid, part in messages.groupby("participant_id")}
    rng = np.random.default_rng(RNG_SEED)
    pseudo = []
    for replicate in range(1, 1001):
        assigned = rng.permutation(fractions)
        control_changes = []
        for fraction, participant in zip(assigned, control.itertuples(index=False)):
            elapsed = float(participant.discussion_elapsed_minutes)
            cutoff = fraction * elapsed
            times = human_times.get(participant.participant_id, np.array([], dtype=float))
            before = int((times < cutoff).sum())
            after = int((times > cutoff).sum())
            control_changes.append(after / (elapsed - cutoff) - before / cutoff)
        control_changes = np.asarray(control_changes)
        pseudo.append({"replicate": replicate, "control_mean_rate_change": control_changes.mean(),
                       "difference_proposal_minus_control": changes.mean() - control_changes.mean(),
                       "welch_p": float(stats.ttest_ind(changes, control_changes, equal_var=False).pvalue)})
    pseudo = pd.DataFrame(pseudo)
    summary = pd.DataFrame([{
        "n": 50, "pre_rate_mean": pre.mean(), "post_rate_mean": post.mean(),
        "mean_change": changes.mean(), "sd_change": changes.std(ddof=1),
        "paired_t": float(paired.statistic), "paired_t_p": float(paired.pvalue),
        "wilcoxon_w": float(wilcoxon.statistic), "wilcoxon_p": float(wilcoxon.pvalue),
        "pseudo_control_mean": pseudo.control_mean_rate_change.mean(),
        "pseudo_difference_mean": pseudo.difference_proposal_minus_control.mean(),
        "pseudo_difference_p025": pseudo.difference_proposal_minus_control.quantile(.025),
        "pseudo_difference_p975": pseudo.difference_proposal_minus_control.quantile(.975),
        "median_welch_p": pseudo.welch_p.median(),
        "fraction_welch_p_below_05": pseudo.welch_p.lt(.05).mean(),
    }])
    extra_tests = {"message_rate_paired_t": float(paired.pvalue),
                   "message_rate_wilcoxon": float(wilcoxon.pvalue)}
    return summary, pseudo, extra_tests


def adjust_tests(focal: pd.DataFrame, extra_tests: dict) -> pd.DataFrame:
    tests = focal[["test", "family", "model", "term", "n", "b", "hc3_se", "ci_low", "ci_high", "p"]].copy()
    tests = pd.concat([tests, pd.DataFrame([
        {"test": "message_rate_paired_t", "family": "other_exploratory", "p": extra_tests["message_rate_paired_t"]},
        {"test": "message_rate_wilcoxon", "family": "sensitivity", "p": extra_tests["message_rate_wilcoxon"]},
    ])], ignore_index=True)
    assert len(tests) == 29, tests.groupby("family").size().to_dict()
    assert tests.family.value_counts().to_dict() == {
        "primary": 1, "secondary": 2, "interaction": 3,
        "other_exploratory": 12, "sensitivity": 11,
    }
    tests["p_adjusted"] = np.nan
    for family, method in (("secondary", "holm"), ("interaction", "fdr_bh"),
                           ("other_exploratory", "fdr_bh"), ("sensitivity", "holm")):
        mask = tests.family.eq(family)
        tests.loc[mask, "p_adjusted"] = multipletests(tests.loc[mask, "p"], method=method)[1]
    mask = tests.family.eq("primary")
    tests.loc[mask, "p_adjusted"] = tests.loc[mask, "p"]
    return tests


def reliability(people: pd.DataFrame) -> pd.DataFrame:
    results = []
    for occasion in ("middle", "end"):
        for n_items in (4, 3):
            data = people[[f"{x}_{occasion}" for x in ITEMS[:n_items]]].astype(float)
            k = data.shape[1]
            alpha = k / (k - 1) * (1 - data.var(ddof=1).sum() / data.sum(axis=1).var(ddof=1))
            z = (data - data.mean()) / data.std(ddof=1)
            pca = PCA(n_components=1).fit(z)
            factor = FactorAnalysis(n_components=1, random_state=42).fit(z)
            loadings = factor.components_[0]
            omega = loadings.sum() ** 2 / (loadings.sum() ** 2 + factor.noise_variance_.sum())
            results.append({"occasion": occasion, "n_items": n_items, "n": len(data),
                            "cronbach_alpha": alpha, "omega_one_factor": omega,
                            "pca_variance_explained": pca.explained_variance_ratio_[0],
                            **{f"loading_{name}": value for name, value in zip(ITEMS[:n_items], loadings)}})
    return pd.DataFrame(results)


def validation_results(validation: pd.DataFrame) -> pd.DataFrame:
    v = validation.copy()
    v["human_mean"] = v[["author_a", "author_b"]].mean(axis=1)
    v["llm_category"] = np.floor(1 + 4 * v.original_llm_score + .5).clip(1, 5).astype(int)
    def bootstrap_pair(x: np.ndarray, y: np.ndarray, statistic, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(BOOTSTRAPS):
            idx = rng.integers(0, len(x), len(x))
            if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
                continue
            estimate = float(statistic(x[idx], y[idx]))
            if np.isfinite(estimate):
                values.append(estimate)
        return tuple(np.percentile(values, [2.5, 97.5]))

    rows = []
    correlations = (
        ("author_a_vs_author_b", "author_a", "author_b", 20261010),
        ("author_a_vs_original_llm", "author_a", "original_llm_score", 20261020),
        ("author_b_vs_original_llm", "author_b", "original_llm_score", 20261030),
        ("human_mean_vs_original_llm", "human_mean", "original_llm_score", 20261040),
        ("repeat_1_vs_repeat_2", "repeat_1", "repeat_2", 20261010),
        ("original_vs_repeat_1", "original_llm_score", "repeat_1", 20261011),
        ("original_vs_repeat_2", "original_llm_score", "repeat_2", 20261012),
    )
    for label, first, second, seed in correlations:
        x, y = v[first].to_numpy(float), v[second].to_numpy(float)
        record = {"comparison": label, "n": len(v),
                     "spearman_rho": stats.spearmanr(x, y).statistic,
                     "spearman_p": stats.spearmanr(x, y).pvalue,
                     "pearson_r": stats.pearsonr(x, y).statistic,
                     "pearson_p": stats.pearsonr(x, y).pvalue,
                     "mean_absolute_difference": np.mean(np.abs(x - y))
                     if first in ("repeat_1", "original_llm_score") else np.nan}
        record["spearman_ci_low"], record["spearman_ci_high"] = bootstrap_pair(
            x, y, lambda a, b: stats.spearmanr(a, b).statistic, seed
        )
        if label in ("human_mean_vs_original_llm", "author_a_vs_author_b",
                     "author_a_vs_original_llm", "author_b_vs_original_llm"):
            record["pearson_ci_low"], record["pearson_ci_high"] = bootstrap_pair(
                x, y, lambda a, b: stats.pearsonr(a, b).statistic, seed + 1
            )
        if first in ("repeat_1", "original_llm_score"):
            record["within_0_1"] = int((np.abs(x - y) <= .1 + 1e-12).sum())
            record["exact_matches"] = int(np.isclose(x, y, atol=1e-12).sum())
            record["max_absolute_difference"] = float(np.max(np.abs(x - y)))
        rows.append(record)
    for label, first, second, seed in (
        ("author_a_vs_author_b", "author_a", "author_b", 20261110),
        ("author_a_vs_categorized_llm", "author_a", "llm_category", 20261120),
        ("author_b_vs_categorized_llm", "author_b", "llm_category", 20261130),
    ):
        x, y = v[first].to_numpy(int), v[second].to_numpy(int)
        record = {"comparison": label + "_agreement", "n": len(v),
                  "exact_matches": int((x == y).sum()),
                  "mean_first": float(x.mean()), "mean_second": float(y.mean())}
        for j, (suffix, weights) in enumerate((("unweighted", None), ("linear", "linear"),
                                                ("quadratic", "quadratic"))):
            record["kappa_" + suffix] = cohen_kappa_score(x, y, weights=weights, labels=[1, 2, 3, 4, 5])
            if suffix == "quadratic":
                record["kappa_ci_low"], record["kappa_ci_high"] = bootstrap_pair(
                    x, y, lambda a, b: cohen_kappa_score(a, b, weights=weights,
                                                         labels=[1, 2, 3, 4, 5]), seed + j
                )
        rows.append(record)
    for condition, group in v.groupby("condition"):
        for first in ("author_a", "author_b", "human_mean"):
            rows.append({"comparison": f"{condition}_{first}_vs_original_llm", "n": len(group),
                         "spearman_rho": stats.spearmanr(group[first], group.original_llm_score).statistic})
    def icc_absolute(matrix: np.ndarray) -> tuple[float, float]:
        n, k = matrix.shape
        grand = matrix.mean()
        row_means = matrix.mean(axis=1)
        column_means = matrix.mean(axis=0)
        ms_rows = k * np.square(row_means - grand).sum() / (n - 1)
        ms_columns = n * np.square(column_means - grand).sum() / (k - 1)
        ms_error = np.square(matrix - row_means[:, None] - column_means[None, :] + grand).sum() / ((n - 1) * (k - 1))
        numerator = ms_rows - ms_error
        single = numerator / (ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n)
        mean = numerator / (ms_rows + (ms_columns - ms_error) / n)
        return float(single), float(mean)

    repeat_matrix = v[["repeat_1", "repeat_2"]].to_numpy(float)
    rng = np.random.default_rng(20261020)
    boot = [icc_absolute(repeat_matrix[rng.integers(0, len(v), len(v))])[0]
            for _ in range(BOOTSTRAPS)]
    human_icc_single, human_icc_mean = icc_absolute(v[["author_a", "author_b"]].to_numpy(float))
    rows.append({"comparison": "author_a_vs_author_b_absolute_agreement_icc",
                 "n": len(v), "icc_a1": human_icc_single, "icc_a2": human_icc_mean})
    repeat_icc_single, repeat_icc_mean = icc_absolute(repeat_matrix)
    rows.append({"comparison": "repeat_1_vs_repeat_2_absolute_agreement_icc_a1",
                 "n": len(v), "icc_a1": repeat_icc_single, "icc_a2": repeat_icc_mean,
                 "icc_ci_low": np.percentile(boot, 2.5), "icc_ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def descriptives(people: pd.DataFrame, agents: pd.DataFrame, words: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_wave = people.groupby("proposal_wave").agg(
        n=("participant_id", "size"), middle_identity_mean=("identity4_middle_mean", "mean"),
        middle_identity_sd=("identity4_middle_mean", "std"),
        post_identity_mean=("identity4_end_mean", "mean"),
        post_identity_sd=("identity4_end_mean", "std"),
        identity_change_mean=("identity4_change_mean", "mean"),
        identity_change_sd=("identity4_change_mean", "std"),
    ).reset_index()
    demographic = (people.groupby(["proposal_wave", "age_group", "gender", "education_category", "university_or_higher"])
                   .size().reset_index(name="n"))
    demographic_change = []
    for variable in ("age_collapsed", "gender_collapsed", "education_collapsed"):
        summary = (people.groupby(variable).identity4_change_mean
                   .agg(["size", "mean", "std"]).reset_index()
                   .rename(columns={variable: "category", "size": "n", "mean": "change_mean", "std": "change_sd"}))
        summary.insert(0, "variable", variable)
        demographic_change.append(summary)
    demographic_change = pd.concat(demographic_change, ignore_index=True)
    word_by_gender = (people.loc[people.proposal_wave.eq(1) & people.gender.isin(["female", "male"])]
                      .groupby("gender").agg(n=("participant_id", "size"),
                                            willingness_mean=("word_use_intention", "mean"),
                                            interestingness_mean=("word_enjoyment", "mean")).reset_index())
    uptake = people.loc[people.proposal_wave.eq(1), [
        "manual_uptake_any", "manual_uptake_count", "manual_substantive_uptake_any",
        "manual_substantive_uptake_count", "manual_first_uptake_minutes_after_proposal",
        "manual_first_substantive_minutes_after_proposal",
    ]]
    uptake_summary = pd.DataFrame([{
        "participants_broad": int(uptake.manual_uptake_any.sum()),
        "messages_broad": int(uptake.manual_uptake_count.sum()),
        "participants_substantive": int(uptake.manual_substantive_uptake_any.sum()),
        "messages_substantive": int(uptake.manual_substantive_uptake_count.sum()),
        "median_first_broad_minutes": uptake.manual_first_uptake_minutes_after_proposal.median(),
        "median_first_substantive_minutes": uptake.manual_first_substantive_minutes_after_proposal.median(),
    }])
    agent_summary = (agents.merge(people[["participant_id", "proposal_wave"]], on="participant_id")
                     .groupby(["proposal_wave", "persona"]).agg(
                         mean_messages=("persona_message_count", "mean"),
                         mean_participant_replies=("participant_replies_to_persona", "mean"),
                         mean_term_mentions=("persona_term_mentions_exact", "mean"),
                     ).reset_index())
    r = people[["identity4_middle_mean", "identity4_end_mean"]].corr().iloc[0, 1]
    d = TTestIndPower().solve_power(effect_size=None, nobs1=50, alpha=.05, power=.8,
                                    ratio=1, alternative="two-sided")
    mde = pd.DataFrame([{"midpoint_post_r": r, "two_group_mde_d": d,
                         "ancova_approximation_mde_d": d * np.sqrt(1 - r**2)}])
    word_rank = words.copy()
    word_rank["evaluation_sum"] = word_rank.word_use_intention + word_rank.word_enjoyment
    word_rank = word_rank.sort_values(["evaluation_sum", "word_id"], ascending=[False, True])
    return {"identity_by_wave.csv": by_wave, "demographic_counts.csv": demographic,
            "demographic_identity_change.csv": demographic_change,
            "word_ratings_by_gender.csv": word_by_gender,
            "neologism_ranking.csv": word_rank, "uptake_summary.csv": uptake_summary,
            "agent_activity_summary.csv": agent_summary, "minimum_detectable_effect.csv": mde}


def main() -> None:
    people, messages, validation, agents, words, legacy_events = load_data()
    sample_audit = verify_validation_sample(people, validation)
    verify_input_consistency(people, messages, agents, words, legacy_events)
    people = prepare(people)
    coefficients, focal = fit_all(people)
    timing_summary, pseudo, extra_tests = timing(people, messages)
    tests = adjust_tests(focal, extra_tests)
    OUT.mkdir(exist_ok=True)
    output = {
        "model_coefficients.csv": coefficients,
        "focal_tests.csv": tests,
        "message_timing.csv": timing_summary,
        "pseudo_cutoff_replicates.csv": pseudo,
        "scale_diagnostics.csv": reliability(people),
        "team_impression_validation.csv": validation_results(validation),
        "validation_sample_audit.csv": sample_audit,
        **descriptives(people, agents, words),
    }
    for name, frame in output.items():
        frame.to_csv(OUT / name, index=False)
    from plot_figure2 import build as build_figure2
    figure = build_figure2()
    print("Reproduced 29 focal tests from 100 participants (50 per wave).")
    print(tests.loc[tests.test.str.contains("primary_ancova|wave_message_impression_change|word_evaluation"),
                    ["test", "b", "p", "p_adjusted", "n"]].to_string(index=False))
    print(f"Results and {figure.name} written to {OUT}")


if __name__ == "__main__":
    main()
