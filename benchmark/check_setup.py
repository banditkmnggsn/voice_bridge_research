"""Step 1 - check that everything needed for the benchmark is in place."""

import importlib
import os
import sys

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"  [OK]   {name}" + (f" - {msg}" if msg else ""))
    except Exception as e:
        ok = False
        print(f"  [GAGAL] {name} - {type(e).__name__}: {e}")


print(f"Python {sys.version.split()[0]} ({sys.executable})")
for mod in ("numpy", "scipy", "sounddevice", "psutil", "requests", "pandas",
            "matplotlib", "torch", "whisper"):
    check(f"import {mod}", lambda m=mod: getattr(importlib.import_module(m), "__version__", ""))


def mic():
    import sounddevice as sd
    return sd.query_devices(kind="input")["name"]


def blocksize():
    """Open the mic exactly like the original scripts (no blocksize given) and
    record which block size the driver delivers. Stay silent for 3 seconds."""
    import collections
    import json
    import time
    import numpy as np
    import sounddevice as sd
    sizes, norms = [], []

    def cb(indata, frames, t, status):
        sizes.append(frames)
        norms.append(float(np.linalg.norm(indata)))

    print("         (DIAM 3 detik - mengukur mic...)", flush=True)
    with sd.InputStream(samplerate=16000, channels=1, callback=cb):
        time.sleep(3.0)
    if not sizes:
        raise RuntimeError("tidak ada audio dari mic")
    bs = collections.Counter(sizes).most_common(1)[0][0]
    info = {"device": sd.query_devices(kind="input")["name"], "blocksize": int(bs),
            "block_sizes_seen": dict(collections.Counter(sizes)),
            "silence_block_norm_p95": float(np.percentile(norms[5:] or norms, 95))}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "audio_device.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    msg = f"block {bs} sampel ({bs / 16:.0f} ms), noise p95={info['silence_block_norm_p95']:.4f}"
    if info["silence_block_norm_p95"] >= 0.05:
        msg += "  !! noise >= threshold C1 (0.05): ruangan/mic terlalu bising"
    return msg


def models():
    import whisper
    for name in ("tiny", "base"):
        whisper.load_model(name, device="cpu")
    return "tiny + base siap (diunduh sekali kalau belum ada)"


def deepseek():
    import voicebridge_core as core
    key = core.load_api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY atau DEEPSEEK_API_KEY belum diisi di benchmark/.env")
    intent, query, info = core.intent_llm("play some jazz music", key)
    if intent == "error":
        raise RuntimeError(info["error"])
    return f"respon: {intent} / {query!r}"


check("mikrofon", mic)
check("ukuran blok audio + noise", blocksize)
check("model Whisper", models)
check("DeepSeek API", deepseek)
print("\nSEMUA SIAP" if ok else "\nAda yang gagal - kirim screenshot ini ke Claude.")
