# Paper & Slide Alignment Checklist

Tracks whether each claim for the CITISMA 2026 presentation is backed by code and evidence.

- **Implemented** — the code exists.
- **Validated** — a run produced an inspectable artifact. Static review does not count.

## Required claims

| ID | Claim | Implementation | Impl. | Valid. | Notes |
|---|---|---|---|---|---|
| A1 | 8-bit quantization | faster-whisper `int8` (C4, C5) | Yes | No | Needs full-run review |
| A2 | Encoder vs decoder timing | Forward hooks + token count | Partial | No | openai-whisper only |
| A3 | 44.1/48 kHz → 16 kHz | `resample_poly`, source rate stored | Yes | No | Full run read 39 × 16 kHz WAVs |
| A4 | Z-score normalization | `zscore` mode, config C3z | Yes | No | Needs runtime comparison |
| A5 | Privacy evidence | `bytes_sent`, `audio_bytes` | Partial | No | Offline test still missing |
| A6 | Hardware floor | `--threads N`, env metadata | Yes | No | Needs runtime check |
| A7 | Noise floor in dBFS | `20·log10(RMS)` in `session.json` | Yes | Yes | −54.20 dBFS |

## Bonus claims (not started)

| ID | Claim | Notes |
|---|---|---|
| B8 | Local LLM via LM Studio | Needs running LM Studio server |
| B9 | Automatic tier selection | `choose_tier()`, `--mode auto` |
| B10 | Noise robustness (SNR 20/10 dB) | Reuses existing dataset |
| B11 | Indonesian mini-test | Separate from English benchmark |

## Pending validation

Implemented, awaiting review of the full run: subprocess isolation, P-core affinity, CPU columns, greedy decoding, per-backend memory baseline, `--test` output isolation, `validate_run.py`.

## Validation log

| Date | Item | Artifact | Result |
|---|---|---|---|
| 2026-10-17 | Setup check (`check_setup.py`) | Console output | Passed; noise warning → recalibrate |
| 2026-10-19 | Shared endpointing smoke run | `runs_20261019_135440.csv` | Passed; identical endpoints across configs |
| 2026-10-19 | Resource measurement smoke run | `runs_20261019_145824.csv` | Passed; CPU/RSS columns populated |
| 2026-10-19 | A7 noise dBFS | `dataset/session.json` | Passed |
| 2026-10-19 | Full benchmark (7 configs × 2 modes × 3 repeats × 39 clips) | `runs_20261019_175744.csv`, `report_…`, `validation_…` | 1638/1638 rows; 0 FAIL, 2 WARN (system CPU > 90% on 11.8% of rows; 10 LLM HTTP 429) — pending review |

Artifacts are under `benchmark/results/`.
