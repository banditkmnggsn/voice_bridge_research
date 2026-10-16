"""
Step 3 - run every configuration on the recorded dataset, fully automatic.

Every clip goes through the same pipeline the live system uses:
endpointing -> Whisper -> wake word -> intent (rule AND DeepSeek, on the same
transcript) -> action URL. All numbers are measured and written to
results/runs_<timestamp>.csv. Nothing is typed in by hand.

What is (and is not) in "latency_after_speech_s":
  endpoint wait (how long the system waits in silence before it stops recording)
  + transcription + intent classification + building the action.
  Opening the browser is NOT included (it depends on the browser's state); the
  live demo measures that separately.

Usage:
  python run_benchmark.py                 # everything
  python run_benchmark.py --modes rule    # skip DeepSeek
  python run_benchmark.py --repeats 1 --limit 3   # quick test
"""

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time

import numpy as np
import psutil
from scipy.io import wavfile

import settings as S
import voicebridge_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
RESULTS = os.path.join(HERE, "results")

FIELDS = [
    "run_id", "timestamp", "repeat", "config", "backend", "precision", "model", "intent_mode",
    "clip_id", "command_id", "split", "take", "ground_truth",
    "speech_start_sec", "speech_end_sec", "segment_start_sec", "cut_sec",
    "endpointed", "truncated", "endpoint_wait_s",
    "transcript", "wer", "wake_detected", "command_text",
    "exp_intent", "exp_query", "pred_intent", "pred_query",
    "intent_correct", "query_correct", "task_success",
    "bytes_sent", "audio_bytes",
    "t_encoder_s", "t_decoder_s", "n_tokens",
    "t_transcribe_s", "cores_transcribe", "t_intent_s", "t_action_s",
    "latency_after_speech_s", "llm_error", "llm_prompt_tokens",
    "llm_completion_tokens", "model_rss_mb", "process_rss_mb",
]


def read_wav(path):
    sr, pcm = wavfile.read(path)
    if sr != S.SAMPLE_RATE:
        raise ValueError(f"{path}: expected {S.SAMPLE_RATE} Hz, got {sr}")
    if pcm.dtype == np.int16:
        audio = pcm.astype(np.float32) / 32767.0
    else:
        audio = pcm.astype(np.float32)
    return audio.reshape(-1)


def cpu_seconds(proc):
    t = proc.cpu_times()
    return t.user + t.system


def rss_mb(proc):
    return proc.memory_info().rss / 2 ** 20


def environment(requested_threads=None):
    import torch
    import whisper
    try:
        import importlib.metadata as md
        whisper_ver = md.version("openai-whisper")
    except Exception:
        whisper_ver = getattr(whisper, "__version__", "?")
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    with open(os.path.join(HERE, "voicebridge_core.py"), "rb") as f:
        core_sha = hashlib.sha1(f.read()).hexdigest()[:10]
    return {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": cpu_name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "requested_threads": requested_threads,
        "whisper": whisper_ver,
        "device": "cpu",
        "core_sha1": core_sha,
        "settings": {k: getattr(S, k) for k in (
            "SAMPLE_RATE", "BLOCKSIZE", "PRE_ROLL_SEC", "REPEATS", "RANDOM_SEED",
            "DEEPSEEK_MODEL")},
        "configs": S.CONFIGS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(S.CONFIGS))
    ap.add_argument("--modes", nargs="+", default=S.INTENT_MODES)
    ap.add_argument("--repeats", type=int, default=S.REPEATS)
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (testing)")
    ap.add_argument("--threads", type=int, default=None,
                    help="set Torch CPU threads for the benchmark")
    args = ap.parse_args()

    if args.threads is not None:
        if args.threads < 1:
            ap.error("--threads must be at least 1")
        import torch
        torch.set_num_threads(args.threads)
        for config in S.CONFIGS.values():
            config["cpu_threads"] = args.threads

    with open(os.path.join(DATA, "manifest.csv"), encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    with open(os.path.join(HERE, "commands.csv"), encoding="utf-8") as f:
        commands = {r["command_id"]: r for r in csv.DictReader(f)}
    with open(os.path.join(DATA, "session.json"), encoding="utf-8") as f:
        session = json.load(f)
    noise_rms = session["calibration"]["noise_rms"]
    if args.limit:
        clips = clips[: args.limit]

    api_key = None
    if "llm" in args.modes:
        api_key = core.load_api_key()
        if not api_key:
            print("!! OPENROUTER_API_KEY/DEEPSEEK_API_KEY tidak ditemukan -> mode LLM dilewati.")
            args.modes = [m for m in args.modes if m != "llm"]
    http = None
    if "llm" in args.modes:
        import requests
        http = requests.Session()
        i, q, info = core.intent_llm("open youtube", api_key, http)  # connectivity check + warm TLS
        if i == "error":
            print(f"!! DeepSeek gagal: {info['error']}")
            print("   Cek API key / saldo / internet. Mode LLM dilewati.")
            args.modes = [m for m in args.modes if m != "llm"]

    os.makedirs(RESULTS, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    env = environment(args.threads)
    env["dataset"] = {"clips": len(clips), "mic": session.get("mic"),
                      "calibration": session["calibration"]}
    env["run"] = {"configs": args.configs, "modes": args.modes, "repeats": args.repeats}
    with open(os.path.join(RESULTS, f"env_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    out_path = os.path.join(RESULTS, f"runs_{stamp}.csv")
    out = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()

    print(f"CPU: {env['cpu']} | {env['logical_cores']} threads | RAM {env['ram_gb']} GB")
    print(f"{len(clips)} klip x {len(args.configs)} config x {args.repeats} ulangan, mode: {args.modes}")

    audio_cache = {c["clip_id"]: read_wav(os.path.join(DATA, "audio", c["file"])) for c in clips}
    proc = psutil.Process()
    rng = random.Random(S.RANDOM_SEED)
    run_id = 0
    t_start = time.time()

    by_model = {}
    for cfg_name in args.configs:
        cfg = S.CONFIGS[cfg_name]
        key = (cfg.get("backend", "openai_whisper"), cfg["model"], cfg.get("precision", "fp32"))
        by_model.setdefault(key, []).append(cfg_name)

    for model_key, cfg_names in by_model.items():
        backend, model_name, precision = model_key
        gc.collect()
        rss_before = rss_mb(proc)
        print(f"\nMemuat {backend} {model_name} ({precision}) ...")
        model = core.load_model(S.CONFIGS[cfg_names[0]])
        model_rss = rss_mb(proc) - rss_before
        # warm-up so the first measured clip is not penalised by lazy init
        core.transcribe(model, np.zeros(S.SAMPLE_RATE, dtype=np.float32), S.CONFIGS[cfg_names[0]])
        core.transcribe(model, audio_cache[clips[0]["clip_id"]], S.CONFIGS[cfg_names[0]])

        for cfg_name in cfg_names:
            cfg = S.CONFIGS[cfg_name]
            for rep in range(1, args.repeats + 1):
                order = clips[:]
                rng.shuffle(order)
                for n, clip in enumerate(order, 1):
                    audio = audio_cache[clip["clip_id"]]
                    cmd = commands[clip["command_id"]]
                    s_start, s_end = core.speech_bounds(audio, noise_rms)
                    start = 0 if s_start is None else max(0, int((s_start - S.PRE_ROLL_SEC) * S.SAMPLE_RATE))
                    seg, ep = core.endpoint_segment(audio, cfg["silence_threshold"],
                                                    cfg["silence_duration"], start_sample=start)
                    truncated = bool(ep["endpointed"] and s_end is not None
                                     and ep["cut_sec"] < s_end - 0.05)

                    c0, t0 = cpu_seconds(proc), time.perf_counter()
                    transcript, transcribe_info = core.transcribe(model, seg, cfg, return_info=True)
                    t1, c1 = time.perf_counter(), cpu_seconds(proc)
                    t_tr = t1 - t0
                    cores = (c1 - c0) / t_tr if t_tr > 0 else 0.0
                    wake, command_text = core.split_wake_word(transcript)

                    for mode in args.modes:
                        info = {"error": "", "prompt_tokens": None, "completion_tokens": None,
                                "request_bytes": 0}
                        if not wake:
                            pred_i, pred_q, t_int = "no_wake_word", "", 0.0
                        elif not command_text:
                            pred_i, pred_q, t_int = "no_command", "", 0.0
                        elif mode == "rule":
                            a = time.perf_counter_ns()
                            pred_i, pred_q = core.intent_rule(command_text)
                            t_int = (time.perf_counter_ns() - a) / 1e9
                        else:
                            a = time.perf_counter()
                            pred_i, pred_q, info = core.intent_llm(command_text, api_key, http)
                            t_int = time.perf_counter() - a
                        a = time.perf_counter_ns()
                        core.action_url(pred_i, pred_q)
                        t_act = (time.perf_counter_ns() - a) / 1e9

                        intent_ok = pred_i == cmd["intent"]
                        query_ok = intent_ok and (cmd["intent"] == "open_youtube_home"
                                                  or core.query_matches(cmd["query"], pred_q))
                        wait = ep["endpoint_wait_s"]
                        total = (wait + t_tr + t_int + t_act) if wait is not None else None
                        run_id += 1
                        writer.writerow({
                            "run_id": run_id,
                            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                            "repeat": rep, "config": cfg_name, "backend": backend,
                            "precision": precision, "model": model_name,
                            "intent_mode": mode, "clip_id": clip["clip_id"],
                            "command_id": clip["command_id"], "split": clip["split"],
                            "take": clip["take"], "ground_truth": cmd["text"],
                            "speech_start_sec": None if s_start is None else round(s_start, 3),
                            "speech_end_sec": None if s_end is None else round(s_end, 3),
                            "segment_start_sec": ep["segment_start_sec"], "cut_sec": ep["cut_sec"],
                            "endpointed": ep["endpointed"], "truncated": truncated,
                            "endpoint_wait_s": wait, "transcript": transcript,
                            "wer": round(core.wer(cmd["text"], transcript), 4),
                            "wake_detected": wake, "command_text": command_text,
                            "exp_intent": cmd["intent"], "exp_query": cmd["query"],
                            "pred_intent": pred_i, "pred_query": pred_q,
                            "intent_correct": intent_ok, "query_correct": query_ok,
                            "task_success": bool(intent_ok and query_ok),
                            "bytes_sent": int(info.get("request_bytes", 0)) if mode == "llm" else 0,
                            "audio_bytes": int(seg.nbytes),
                            "t_encoder_s": transcribe_info["t_encoder_s"],
                            "t_decoder_s": transcribe_info["t_decoder_s"],
                            "n_tokens": transcribe_info["n_tokens"],
                            "t_transcribe_s": round(t_tr, 4), "cores_transcribe": round(cores, 3),
                            "t_intent_s": round(t_int, 9), "t_action_s": round(t_act, 9),
                            "latency_after_speech_s": None if total is None else round(total, 4),
                            "llm_error": info["error"],
                            "llm_prompt_tokens": info["prompt_tokens"],
                            "llm_completion_tokens": info["completion_tokens"],
                            "model_rss_mb": round(model_rss, 1),
                            "process_rss_mb": round(rss_mb(proc), 1),
                        })
                    out.flush()
                    mins = (time.time() - t_start) / 60
                    print(f"\r  {cfg_name} ulangan {rep}/{args.repeats}  klip {n}/{len(order)}"
                          f"  | {mins:.1f} menit", end="", flush=True)
                print()
        core._MODELS.clear()
        del model
        gc.collect()

    out.close()
    print(f"\nSelesai. Data mentah: {out_path}")
    print("Langkah berikutnya: jalankan 4_analisis.bat")


if __name__ == "__main__":
    sys.exit(main())
