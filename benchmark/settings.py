"""
VoiceBridge benchmark v2 - shared settings.

All experiment parameters live here so the paper, the slides and the code
can point to one place.
"""

SAMPLE_RATE = 16000          # Whisper's native rate; the mic is opened at this rate
CHANNELS = 1

# The original scripts used sounddevice's default block size, which depends on
# the audio driver. The silence threshold is compared against the L2 norm of
# one block, so the block size changes what the threshold means. check_setup.py
# measures the block size the driver actually delivers on this PC (what the
# original scripts got) and stores it in audio_device.json; v2 then uses that
# fixed value everywhere.
import json as _json
import os as _os

_dev_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "audio_device.json")
BLOCKSIZE = 1024             # fallback: 64 ms at 16 kHz
if _os.path.exists(_dev_file):
    with open(_dev_file, encoding="utf-8") as _f:
        BLOCKSIZE = int(_json.load(_f)["blocksize"])

WAKE_WORD = "hello"

# Segment start used when replaying a recorded clip through the endpointing
# logic: the live loop would have started the segment shortly before speech.
PRE_ROLL_SEC = 0.20

# Recording session
CLIP_SECONDS = 6.0           # fixed window per take; speak after the "GO" cue
TAKES_PER_COMMAND = {"original": 5, "paraphrase": 3}   # takes per command, by set
CALIBRATION_SECONDS = 3.0

# Benchmark
REPEATS = 3                  # latency repeats per clip per configuration
RANDOM_SEED = 2026

# Configurations. C1-C3 reproduce the paper's parameters.
# NOTE on C3's vocabulary prompt: the original code passed it as `prompt=`,
# which openai-whisper's transcribe() silently overwrites (transcribe.py,
# decode_options["prompt"] = all_tokens[...]). v2 passes it as
# `initial_prompt=`, which is the supported way, so the prompt is actually used.
VOCAB_PROMPT = "play youtube despacito hello music video search google"

CONFIGS = {
    "C1": {
        "label": "C1 base",
        "backend": "openai_whisper",
        "precision": "fp32",
        "model": "base",
        "silence_threshold": 0.05,
        "silence_duration": 0.6,
        "normalize": "none",
        "vocab_prompt": None,
    },
    "C2": {
        "label": "C2 tiny",
        "backend": "openai_whisper",
        "precision": "fp32",
        "model": "tiny",
        "silence_threshold": 0.15,
        "silence_duration": 0.4,
        "normalize": "none",
        "vocab_prompt": None,
    },
    "C3": {
        "label": "C3 tiny+norm+prompt",
        "backend": "openai_whisper",
        "precision": "fp32",
        "model": "tiny",
        "silence_threshold": 0.15,
        "silence_duration": 0.4,
        "normalize": "peak",
        "vocab_prompt": VOCAB_PROMPT,
    },
    # Ablation: separates the effect of normalization from the prompt.
    "C3n": {
        "label": "C3n tiny+norm (ablation)",
        "backend": "openai_whisper",
        "precision": "fp32",
        "model": "tiny",
        "silence_threshold": 0.15,
        "silence_duration": 0.4,
        "normalize": "peak",
        "vocab_prompt": None,
    },
    "C3z": {
        "label": "C3z tiny+zscore+prompt",
        "backend": "openai_whisper",
        "precision": "fp32",
        "model": "tiny",
        "silence_threshold": 0.15,
        "silence_duration": 0.4,
        "normalize": "zscore",
        "vocab_prompt": VOCAB_PROMPT,
    },
    "C4": {
        "label": "C4 tiny int8",
        "backend": "faster_whisper",
        "precision": "int8",
        "model": "tiny",
        "silence_threshold": 0.15,
        "silence_duration": 0.4,
        "normalize": "peak",
        "vocab_prompt": VOCAB_PROMPT,
    },
    "C5": {
        "label": "C5 base int8",
        "backend": "faster_whisper",
        "precision": "int8",
        "model": "base",
        "silence_threshold": 0.15,
        "silence_duration": 0.4,
        "normalize": "peak",
        "vocab_prompt": VOCAB_PROMPT,
    },
}

INTENT_MODES = ["rule", "llm"]

# DeepSeek (OpenAI-compatible chat completions API)
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT_SEC = 20
