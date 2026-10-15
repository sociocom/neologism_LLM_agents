# Reproduction code

`reproduce_results.py` reads only the six CSV files in `../data/`. It writes analysis tables and Figure 2 to `../results/` and prints selected focal estimates. It does not access the private project directories or call an external model API.

## Requirements and execution

Python 3.11 or later is recommended. From the parent directory containing `data/` and `code/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/reproduce_results.py
```

The script was checked against the current manuscript using Python 3.12 and the versions allowed by `requirements.txt`. Small last-digit differences can occur with other supported library versions.

## Generated tables

| Output in `../results/` | Reproduces |
|---|---|
| `model_coefficients.csv` | Full HC3 regression coefficients, standardized coefficients, confidence intervals, sample sizes, and fit statistics; includes descriptive within-wave interactions |
| `focal_tests.csv` | All 29 focal tests and their nominal and family-adjusted *p*-values |
| `identity_by_wave.csv`, `demographic_counts.csv`, `demographic_identity_change.csv`, `word_ratings_by_gender.csv`, `neologism_ranking.csv` | Main and supplementary descriptive results, including all education categories and the proposed-word score ranking |
| `uptake_summary.csv`, `agent_activity_summary.csv` | Broad/substantive participant use and descriptive agent-persona activity |
| `message_timing.csv`, `pseudo_cutoff_replicates.csv` | Pre/post proposal rates and 1,000 comparison-wave pseudo-cutoff assignments |
| `scale_diagnostics.csv` | Alpha, one-factor omega, first-component variance, and factor loadings for four- and three-item identity scores |
| `team_impression_validation.csv`, `validation_sample_audit.csv` | Human–LLM Spearman/Pearson correspondence, within-wave correlations, κ agreement and ICC, repeat-score statistics, and verification of the 25-per-wave sample and case-to-score links |
| `minimum_detectable_effect.csv` | Two-group and midpoint-adjusted sample-sensitivity calculations |
| `figure2.pdf` | Figure 2 forest plot generated from the regression and multiplicity tables by `plot_figure2.py` |

## Analysis decisions

- Wave is coded 1 for the proposal wave and 0 for the comparison wave. It is perfectly confounded with recruitment period.
- The main between-wave analysis is an ANCOVA of post-discussion four-item **mean** identity, adjusting for midpoint mean identity.
- The focal message-count, team-impression, and pooled three-way interaction models use the midpoint-to-post-discussion four-item **sum change**. Their midpoint-adjusted counterparts use post-discussion four-item **mean** identity.
- Total message count and post-discussion team impression are centered using the all-participant mean; within-wave descriptive models center within their own wave.
- Each regression uses HC3 heteroskedasticity-robust standard errors. Reported standardized coefficients multiply the unstandardized coefficient by the design-matrix term's sample SD divided by the outcome's sample SD.
- The 29 focal tests comprise one unadjusted primary test; two Holm-adjusted secondary tests; three BH-FDR-adjusted change-score interactions; 12 other BH-FDR-adjusted exploratory tests; and 11 Holm-adjusted sensitivity tests. These families were defined during manuscript revision, not preregistered.
- The 1,000 pseudo-cutoff assignments use NumPy generator seed `20261008` and the preserved public analysis order. The across-assignment percentile interval describes cutoff allocation variability, not a sampling confidence interval.
- The human-validation bootstrap calculations use 5,000 case resamples. The two new LLM scores are fixed released measurements; this script does not issue API requests.

The code reproduces statistical results from analysis-ready inputs. It does not recreate the original bot dialogue, perform sentiment scoring on private text, or independently recode word use from private transcripts.
The audited within-discussion message counts are checked against released relative message times. For 99 participants, the original total message count is recomputed from anonymized `role=user` event classes, excluding `author=reply`. The remaining participant's count of 34 was recovered from the original aggregate workbook because its source JSON was unavailable; its source-event derivation cannot be checked from this package.
