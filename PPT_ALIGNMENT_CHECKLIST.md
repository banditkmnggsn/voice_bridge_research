# PPT and Paper Alignment Checklist

This document tracks whether the implementation supports the claims planned for the CITISMA 2026 presentation. `Implemented` means the code exists. `Validated` means the behavior has been exercised with the required runtime, dataset, and measurements. A code-only change is not experimental evidence.

## A. Required claims

| ID | Requirement | Implementation target | Implemented | Validated | Notes |
|---|---|---|---|---|---|
| A1 | 8-bit quantization | `faster-whisper` with `compute_type="int8"`; C4 tiny int8 and C5 base int8 | Yes | No | Backend/config/CSV support added; requires dependency installation and model runs |
| A2 | Encoder vs decoder timing | Separate encoder/decoder timing and generated token count | Partial | No | Breakdown encoder/decoder hanya untuk openai-whisper; faster-whisper internals still need a supported timing method |
| A3 | 44.1/48 kHz to 16 kHz downsampling | Record at device rate, resample with `scipy.signal.resample_poly`, store source rate | Yes | No | `session.json` mencatat sumber 44.1 kHz; tervalidasi saat full run membaca 39 WAV 16 kHz (`read_wav` menolak selain 16 kHz) |
| A4 | Z-score normalization | `none`, `peak`, `zscore`; add C3z | Yes | No | Implemented in `voicebridge_core.py` and `settings.py`; requires runtime comparison |
| A5 | Privacy evidence | `bytes_sent`, `audio_bytes`, offline rule test | Partial | No | Byte accounting is implemented; offline behavior still needs a focused test |
| A6 | Hardware floor | `--threads N`, Torch/faster-whisper thread settings, env metadata | Yes | No | Thread override is passed to Torch and faster-whisper; requires runtime verification |
| A7 | Noise floor in dBFS | Store `noise_dbfs = 20 * log10(RMS)` in `session.json` | Yes | Yes | `noise_rms 0.0019495 -> 20*log10 = -54.20 dBFS`, sama dengan `noise_dbfs` |

## B. Bonus claims

| ID | Requirement | Implementation target | Implemented | Validated | Notes |
|---|---|---|---|---|---|
| B8 | Local LLM via LM Studio | Third intent mode using OpenAI-compatible local endpoint | No | No | Requires a running LM Studio server |
| B9 | Automatic tier selection | `choose_tier()` and `demo_live.py --mode auto` | No | No | Requires network/RAM/core detection tests |
| B10 | Noise robustness | SNR 20 dB and 10 dB replay plus WER graph | No | No | Can be done without re-recording after the base dataset exists |
| B11 | Indonesian mini-test | Indonesian commands, `language="id"`, wake word `halo` | No | No | Must be isolated from the English benchmark |

## Existing baseline already in the repository

- Rule-based intent parsing, DeepSeek intent parsing, Whisper CPU inference, endpointing, WER, task success, wake-word miss, truncation, latency, process CPU, and model/process memory are implemented in benchmark v2.
- C1, C2, C3, C3n, C3z, C4, and C5 are declared in `benchmark/settings.py`.
- The benchmark writes raw CSV and the analysis script produces summary/report files.
- No v2 result CSV is currently evidence until the dataset and runtime benchmark have been executed.

## Implemented, pending full-run validation

The following implementation changes are complete but remain `Validated = No` until a full benchmark produces inspectable output:

- subprocess isolation per model group;
- P-core affinity detection and application;
- process CPU average and normalized CPU columns;
- greedy decoding for both backends;
- fair per-backend memory baseline and effective compute type;
- `--test` output isolation;
- read-only `validate_run.py`.

## Validation policy

- Do not mark `Validated` from static inspection alone.
- A requirement is validated only after the focused test or benchmark produces an inspectable artifact.
- PPT numbers remain `TBA` until the corresponding result artifact exists.
- After each validated item, record the command, date, and artifact path below.

## Validation log

| Date | Item | Command/test | Artifact | Result |
|---|---|---|---|---|
| 2026-10-16 | Baseline code diagnostics | Editor diagnostics on core benchmark files | No diagnostics reported | Passed static diagnostics; runtime not available |
| 2026-10-16 | Repository synchronization | Git push to `origin/master` | Commit `d7bb615` | Passed |
| 2026-10-16 | Static implementation update | Diagnostics for settings, core, recording, benchmark | No diagnostics on changed files | A4 implemented; A6 partial; A7 implemented; runtime not available |
| 2026-10-16 | Audio and privacy implementation update | Diagnostics for core, recorder, demo, benchmark | No new code diagnostics; external imports unresolved | A3 implemented; A5 partial; runtime not available |
| 2026-10-16 | Quantization and timing implementation update | Diagnostics plus `git diff --check` | Changed files clean; runtime not available | A1 implemented; A2 partial; C4/C5 and timing columns added |
| 2026-10-16 | Report and thread-control update | Diagnostics on report and benchmark | No code diagnostics; runtime not available | Precision-aware summaries and faster-whisper thread propagation added |
| 2026-10-17 | Full setup validation | `benchmark/check_setup.py` | Python 3.11, dependencies, Whisper tiny/base, microphone, and OpenRouter passed; noise p95 was 1.8630 and triggered threshold warning | Environment works; recalibrate in a quieter/lower-gain setup before recording |
| 2026-10-19 | Shared benchmark endpointing | Smoke run `C1 C3 C4`, rule-only, 3 clips | `benchmark/results/runs_20261019_135440.csv` and `env_20261019_135440.json` | 0 clips had different endpoint metadata; shared threshold 0.1933678 from noise p95 0.0773471 | Passed on real 44.1 kHz -> 16 kHz recordings; full run still pending |
| 2026-10-19 | Resource measurement | Smoke run `C1 C4 --modes rule --repeats 1 --limit 3 --threads 2` | `benchmark/results/runs_20261019_145824.csv` and `env_20261019_145824.json` | Process raw/normalized CPU, system CPU, peak RSS, fixed threads, and per-config warm-up populated | Passed; full benchmark still pending |
| 2026-10-19 | A7 noise dBFS | review session.json | `dataset/session.json` | `noise_rms 0.0019495 -> 20*log10 = -54.20 dBFS`, sama dengan `noise_dbfs` | Passed |
| 2026-10-19 | Final full benchmark (7 configs x rule+llm x 3 repeats, 39 clips) | `benchmark/run_all.ps1` (smoke `--test --modes rule --repeats 1 --limit 2` + `validate_run.py --test`, then `run_benchmark.py`, `analyze_results.py`, `validate_run.py`) | `benchmark/results/runs_20261019_175744.csv`, `env_20261019_175744.json`, `report_20261019_175744.md`, `validation_20261019_175744.md`, `_logs/RUN_SUMMARY_20261019_175423.md` | Smoke PASS=17 WARN=1 FAIL=0; full PASS=16 WARN=2 FAIL=0 (1638/1638 rows; WARN: system CPU >90% on 11.84% of rows, mostly C4/C5; 10 LLM errors, all HTTP 429 from provider in C2/C3 at 18:00-18:01); power throttling disabled in parent and all workers | Pending review |
