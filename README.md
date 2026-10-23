# VoiceBridge Research

VoiceBridge is a fully local voice-command pipeline:

```text
microphone -> endpointing -> Whisper (ASR) -> intent parser -> browser action
```

This repository benchmarks how model size, quantization, and audio preprocessing affect accuracy, latency, and resource use on a CPU-only laptop.

## Repository layout

| Path | Purpose |
|---|---|
| `benchmark/` | Current experiment pipeline (used for the paper and slides) |
| `benchmark/guide.md` | Step-by-step setup, recording, and benchmark guide |
| `benchmark/settings.py` | Configurations C1–C5 |
| `benchmark/commands.csv` | Ground-truth test commands |
| `PPT_ALIGNMENT_CHECKLIST.md` | Status of each paper/slide claim |
| `deepseek/`, `local_rule/` | Legacy prototypes (archived) |

## Quick start (Windows)

```powershell
cd benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # add an OpenRouter key for LLM mode; never commit .env

.\1_cek_setup.bat         # check dependencies, mic, models
.\2_rekam.bat             # record the 39-clip dataset
.\3_benchmark.bat         # run all configurations
.\4_analisis.bat          # build report and charts
.\5_demo_live.bat         # live demo (C3, rule-based)
```

Full details: [benchmark/guide.md](benchmark/guide.md).

## Configurations

| Config | Backend | Model | Precision | Preprocessing |
|---|---|---|---|---|
| C1 | openai-whisper | base | FP32 | none |
| C2 | openai-whisper | tiny | FP32 | none |
| C3 | openai-whisper | tiny | FP32 | peak norm + vocabulary prompt |
| C3n | openai-whisper | tiny | FP32 | peak norm (ablation, no prompt) |
| C3z | openai-whisper | tiny | FP32 | z-score + vocabulary prompt |
| C4 | faster-whisper | tiny | INT8 | peak norm + vocabulary prompt |
| C5 | faster-whisper | base | INT8 | peak norm + vocabulary prompt |

Each configuration runs with two intent parsers: **rule-based** (offline) and **LLM** (OpenRouter).

## Metrics

| Metric | Definition |
|---|---|
| WER | `(substitutions + deletions + insertions) / reference words`, after text normalization |
| Task success | Correct intent **and** query contains all expected content words (≤ 2 extra) |
| Transcription latency | Wall time of `core.transcribe()` (preprocessing + inference) |
| Intent latency | Wall time of `intent_rule()` / `intent_llm()` |
| End-to-end latency | `endpoint_wait + transcribe + intent + action` (browser not opened in benchmark) |
| Encoder / decoder time | Forward hooks on Whisper encoder/decoder (openai-whisper only) |
| CPU (raw) | `Δ process CPU time / transcription wall time × 100`; can exceed 100% on multiple cores |
| CPU (normalized) | `raw / logical cores`, clamped to 0–100% |
| Memory | Peak RSS of the process tree during transcription; `model_rss_mb` = RSS increase on model load |
| Privacy | `bytes_sent` (0 for rule-based); `audio_bytes` = local audio never sent |

## Methodology notes

- **Identical input:** every clip is endpointed once (threshold `2.5 × noise p95`, silence `0.6 s`) before the config loop, so all configs see the same audio.
- **Model isolation:** each backend/model group runs in its own Python subprocess, so memory readings do not mix.
- **Warm-up:** one unrecorded warm-up per config; model loading is excluded from latency.
- **Decoding:** greedy for both backends (`beam_size=1`, `best_of=1`, VAD off for faster-whisper).
- **CPU pinning:** on Windows hybrid CPUs, P-cores are detected via the CPU Set API and affinity is applied; 2 threads are used. Normalization still divides by all logical cores.
- **Audio:** recorded at the device rate (44.1/48 kHz) and downsampled to 16 kHz with `scipy.signal.resample_poly`.
