"""
Live demo: talk to VoiceBridge with the microphone and it opens the browser.

  python demo_live.py                      # C3 + rule-based
  python demo_live.py --config C3 --mode llm

Prints the timing of every stage, including opening the browser, so the demo
can be recorded for the video and the presentation.
"""

import argparse
import queue
import time
import webbrowser

import numpy as np
import sounddevice as sd

import settings as S
import voicebridge_core as core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C3", choices=list(S.CONFIGS))
    ap.add_argument("--mode", default="rule", choices=S.INTENT_MODES)
    args = ap.parse_args()
    cfg = S.CONFIGS[args.config]

    api_key = None
    if args.mode == "llm":
        api_key = core.load_api_key()
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY atau DEEPSEEK_API_KEY belum diisi di benchmark/.env")
        import requests
        http = requests.Session()
    else:
        http = None

    print(f"Memuat Whisper-{cfg['model']} ...")
    model = core.load_model(cfg)
    core.transcribe(model, np.zeros(S.SAMPLE_RATE, dtype=np.float32), cfg)  # warm-up

    q = queue.Queue()
    source_rate = int(round(sd.query_devices(kind="input")["default_samplerate"]))

    def callback(indata, frames, t, status):
        q.put(core.resample_audio(indata[:, 0], source_rate))

    print(f"\nVoiceBridge siap  | {cfg['label']} | intent: {args.mode}")
    print(f"Ucapkan: \"{S.WAKE_WORD} ...\"   (Ctrl+C untuk berhenti)\n")
    with sd.InputStream(samplerate=source_rate, channels=1, dtype="float32",
                        callback=callback):
        while True:
            frames, silence_start, heard = [], None, False
            while True:
                block = q.get()
                frames.append(block)
                if core.block_volume(block) < cfg["silence_threshold"]:
                    if silence_start is None:
                        silence_start = time.perf_counter()
                    elif time.perf_counter() - silence_start > cfg["silence_duration"]:
                        break
                else:
                    silence_start, heard = None, True
            if not heard:
                continue
            t_stop = time.perf_counter()
            audio = np.concatenate(frames)
            text = core.transcribe(model, audio, cfg)
            t_tr = time.perf_counter()
            wake, command = core.split_wake_word(text)
            print(f"Dengar: {text!r}")
            if not wake or not command:
                continue
            if args.mode == "rule":
                intent, query = core.intent_rule(command)
            else:
                intent, query, info = core.intent_llm(command, api_key, http)
                if intent == "error":
                    print("  DeepSeek error:", info["error"])
                    continue
            t_int = time.perf_counter()
            url = core.action_url(intent, query)
            webbrowser.open(url)
            t_act = time.perf_counter()
            print(f"  -> {intent} | {query!r}")
            print(f"  transkripsi {t_tr - t_stop:.2f}s | intent {t_int - t_tr:.3f}s | "
                  f"buka browser {t_act - t_int:.2f}s | total sejak diam "
                  f"{t_act - t_stop + cfg['silence_duration']:.2f}s\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSelesai.")
