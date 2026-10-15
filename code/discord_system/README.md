# Discord discussion system

This directory contains the two condition-specific Discord bot programs that match the study procedure. `proposal_wave.py` is derived from the private `discord_bot/bot.py`; `comparison_wave.py` is derived from `discord_bot/bot_control.py`. The source files' SHA-256 hashes are recorded below so the release can be checked against the retained originals.

| Released file | Retained source | Source SHA-256 | Experimental distinction |
|---|---|---|---|
| `proposal_wave.py` | `discord_bot/bot.py` | `65b52f427159578873bd4b83259f9077cd45e6767f28b076f1a69ea25a948361` | `new_turn = 8`, before `end_turn = 11`; after proposal, the new term could be reused with probability `0.5` in the configured opportunities. |
| `comparison_wave.py` | `discord_bot/bot_control.py` | `8c9aef5503b6e3d154ecc5a6eb03f5a1b0d53d57d459dd2dcbf54a9227fea3a6` | `new_turn = 100`, after `end_turn = 11`; no proposal occurs during a completed session. |

The two retained programs differ only in the proposal-turn setting and the wave-specific midpoint and end questionnaire URLs. The private `bot.log` also contains a traceback naming `discord_bot/bot.py`. The available files do not contain a deployment manifest for every session, so the exact executed file version for each individual discussion cannot be independently certified. The two programs above are the closest retained versions matching the reported conditions; the older `bot_save.py`, `bot_test.py`, and `bot_ratetest.py` differ materially and are not presented as the study programs.

## Public-release changes

Only the four wave-specific questionnaire URLs were removed from the retained programs. The released files read `MIDPOINT_SURVEY_URL` and `END_SURVEY_URL` from environment variables instead. The bot logic, agent prompts, model selection (`gemini-2.0-flash`), timing parameters, and proposal/reuse logic are otherwise retained. Credentials, webhook URLs, and optional avatar URLs were already read from environment variables in the retained files. No `.env` file, logs, participant messages, memory files, or avatar images are included.

## Requirements and configuration

The code uses Python 3.10+ and the packages in `requirements.txt`. The original deployed package versions are unavailable; the requirements list package names without claiming exact environment reproduction. To configure a separate Discord server, set the following environment variables before running one of the programs:

- `DISCORD_BOT_TOKEN`, `GOOGLE_API_KEY`, `DISCORD_CHANNEL_ID`, and `DISCORD_GUILD_ID`
- `AYU_WEBHOOK_URL`, `KAZUKI_WEBHOOK_URL`, `KANAKO_WEBHOOK_URL`, and `KEN_WEBHOOK_URL`
- `MIDPOINT_SURVEY_URL` and `END_SURVEY_URL`
- Optional avatar URLs: `AYU_AVATAR_URL`, `KAZUKI_AVATAR_URL`, `KANAKO_AVATAR_URL`, and `KEN_AVATAR_URL`

Run exactly one condition program per configured bot instance. Running it connects to external Discord and Gemini services and creates local logs. The released programs are provided to document the experimental implementation; the statistical results are reproduced separately by `../reproduce_results.py` using de-identified CSV inputs.
