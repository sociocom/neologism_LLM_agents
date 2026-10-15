# Data files

All files are UTF-8 CSVs. Missing cells represent measures that were not available for that wave or participant; in particular, the proposed-word ratings, proposal timing, and uptake measures apply only to the proposal wave. The public `participant_id` is a new pseudonym; no name-to-ID mapping is provided.

| File | Rows | Contents |
|---|---:|---|
| `participants.csv` | 100 | One human per discussion: condition, identity items, demographics, scores, message counts, proposal timing, and coded word uptake |
| `human_message_times.csv` | 2,094 | Human message times in minutes relative to discussion start, without text or absolute timestamps; used for the 1,000 pseudo-cutoff assignments |
| `team_impression_validation.csv` | 50 | Two independent author ratings, the original LLM score, and two later LLM scores for the sampled responses |
| `agent_activity.csv` | 400 | Four agent-persona activity counts per discussion, without utterance content |
| `neologism_ratings.csv` | 50 | Agent-proposed katakana words and the two questionnaire ratings, without participant IDs |
| `legacy_user_events.csv` | 2,438 | Original conversation records with `role=user`, classified only by whether `author=reply`; no text or timestamps |

## `participants.csv`

| Columns | Meaning |
|---|---|
| `participant_id` | Public, arbitrary ID. `P001`–`P050` belong to the comparison wave and `P051`–`P100` to the proposal wave. The order is retained only to reproduce the seeded pseudo-cutoff assignment. |
| `proposal_wave` | 1 = first recruitment wave, with an agent-proposed neologism; 0 = second recruitment wave, without a proposal. This also encodes condition. |
| `id_with_group_*`, `member_selfview_*`, `identity_importance_*`, `criticism_as_self_*` | The four 1–5 identity items at `middle` and `end`. Four-item sums/means and the three-item sensitivity scores are calculated by the script. |
| `total_message_count`, `total_count_source` | Human total message count used in the principal message-count regressions and its source: JSON user/non-reply event count for 99 participants, recovered workbook value for one. |
| `within_discussion_message_count` | Audited human content-message count inside the discussion window, excluding commands. |
| `extended_window_message_count` | Count that additionally retains participant content messages after the end announcement; a sensitivity definition. |
| `post_end_message_count` | Number of participant messages after the end announcement. |
| `team_impression_end_score` | Original Gemini 2.5 Flash score of the participant's post-discussion team description, on a 0–1 scale. It is concurrent with the post-discussion identity score. |
| `word_use_intention`, `word_enjoyment` | Two 1–7 ratings of the proposed word. Their sum is the neologism-evaluation variable. |
| `discussion_elapsed_minutes`, `proposal_minutes_from_start` | Relative session duration and time of the first proposal; no calendar date or time is released. |
| `messages_pre_proposal`, `messages_post_proposal` | Human message counts strictly before and after the first proposal, used for message rates. |
| `age_group`, `gender`, `education_category`, `university_or_higher` | Demographic categories used in descriptive and exploratory analyses. `education_category` preserves the five levels in Supplementary Table S1; `university_or_higher` is 1/0. |
| `validation_sampling_order`, `validation_case_id` | The source row order used for the stratified 25-per-wave validation draw and the assigned case ID for selected responses. Neither reveals the source account name. The code replays the draw and checks the case-to-score links. |
| `manual_uptake_any`, `manual_uptake_count` | Whether the human used the proposed expression broadly and number of such messages. Repetition/evaluation can qualify as broad uptake. |
| `manual_substantive_uptake_any`, `manual_substantive_uptake_count` | Presence and count of spontaneous substantive use beyond repetition/evaluation. |
| `manual_first_*_minutes_*` | Relative first-use times used for uptake summaries; blank when no qualifying use occurred. |

The within-discussion and extended-window counts are alternative definitions; the original total count remains the focal message-count variable. The relative-time file reproduces the audited within-discussion counts, **not** the original total count. `legacy_user_events.csv` allows the original JSON event filter to be reapplied for 99 participants. One participant's count of 34 was recovered from the original aggregate workbook because the relevant source JSON was unavailable. Its derivation cannot be checked further from this package. Missing word and proposal data in the comparison wave are **not zeroes**.

## Other files

`human_message_times.csv` has `participant_id` and `minutes_from_start`. A participant with no retained content message has no row. The public IDs preserve the private analysis ordering needed for the seeded pseudo-cutoff replication; they do not disclose the original account names.

`legacy_user_events.csv` has `participant_id` and `author_is_reply_marker`. Each row represents one record with `role=user` in a saved conversation JSON. The original count included rows with `author_is_reply_marker=0`; rows with 1 were excluded. The file has no content, absolute time, or original account identifiers. It contains events for 99 of 100 participants; the remaining count came from the original workbook.

`team_impression_validation.csv` has `case_id`, `condition`, `original_llm_score`, `author_a`, `author_b`, `repeat_1`, and `repeat_2`. The selected cases are linked to `participants.csv` through `validation_case_id`, allowing verification of the stratified draw and original-score matching. The two authors used a 1–5 rubric; LLM scores use 0–1. No response text is included.

`agent_activity.csv` has `participant_id`, `persona`, `persona_message_count`, `persona_replies_to_participant`, `participant_replies_to_persona`, and `persona_term_mentions_exact`. Persona names identify agent roles, not human participants. These are descriptive measures and do not isolate persona-specific causal effects.

`neologism_ratings.csv` has `word_id`, `proposed_word_katakana`, `word_use_intention`, and `word_enjoyment`. These short terms were generated by agents. The rows are independently sorted and have no explicit participant ID; however, a unique pair of rating values may permit some rows to be matched indirectly to `participants.csv`. The script sums the two ratings to reproduce the descriptive ranking; English glosses and proposal explanations are printed in the manuscript rather than included as analysis data.

## Data minimization

This package includes derived numeric records rather than raw logs or free-text answers. It permits recomputation of reported statistics while withholding the text needed to rejudge semantic coding, reassess the original LLM sentiment scores, or reconstruct the Discord sessions. Uptake classifications and the original LLM sentiment scores should be understood as analysis inputs.
