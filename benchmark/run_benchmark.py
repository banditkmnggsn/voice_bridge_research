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
import ctypes
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import struct
import subprocess
import threading
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
    "t_transcribe_s", "cores_transcribe", "cpu_process_raw_peak_percent",
    "cpu_process_normalized_peak_percent", "cpu_process_average_percent",
    "cpu_process_average_normalized_percent", "cpu_system_peak_percent", "peak_rss_mb",
    "t_intent_s", "t_action_s",
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


def detect_p_cores():
    """Return Windows logical CPU indices with the best efficiency class."""
    if platform.system() != "Windows":
        raise RuntimeError("P-core detection is only implemented for Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    getter = kernel32.GetSystemCpuSetInformation
    required = ctypes.c_ulong(0)
    getter(None, 0, ctypes.byref(required), None, 0)
    if not required.value:
        raise RuntimeError("Windows CPU Set information is unavailable")
    buffer = ctypes.create_string_buffer(required.value)
    if not getter(buffer, required.value, ctypes.byref(required), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    records = []
    offset = 0
    while offset + 8 <= required.value:
        size, record_type = struct.unpack_from("<II", buffer, offset)
        if size < 8 or offset + size > required.value:
            break
        if record_type == 0 and size >= 20:
            logical_index = buffer[offset + 14]
            efficiency_class = buffer[offset + 18]
            records.append((logical_index, efficiency_class))
        offset += size
    if not records:
        raise RuntimeError("No Windows CPU Set records were returned")
    best_class = min(efficiency for _, efficiency in records)
    return [index for index, efficiency in records if efficiency == best_class], {
        "efficiency_class": best_class,
        "all_cpu_efficiency_classes": {str(index): efficiency for index, efficiency in records},
    }


def apply_p_core_affinity():
    p_cores, details = detect_p_cores()
    process = psutil.Process(os.getpid())
    process.cpu_affinity(p_cores)
    details["p_core_indices"] = p_cores
    details["applied_affinity"] = process.cpu_affinity()
    return details


class ResourceSampler:
    """Sample process-tree CPU/RSS and system CPU only during transcription."""

    def __init__(self, interval_s=0.075):
        self.interval_s = interval_s
        self.root = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = None
        self.cpu_process_peak = 0.0
        self.cpu_system_peak = 0.0
        self.rss_peak = 0
        self.cpu_process_start = 0.0
        self.cpu_process_end = 0.0
        self.wall_start = 0.0
        self.wall_end = 0.0

    def _process_tree(self):
        try:
            return [self.root] + self.root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return [self.root]

    def _sample(self):
        processes = self._process_tree()
        process_cpu = 0.0
        rss = 0
        for process in processes:
            try:
                process_cpu += process.cpu_percent(None)
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.cpu_process_peak = max(self.cpu_process_peak, process_cpu)
        self.cpu_system_peak = max(self.cpu_system_peak, psutil.cpu_percent(None))
        self.rss_peak = max(self.rss_peak, rss)

    def _cpu_seconds(self):
        total = 0.0
        for process in self._process_tree():
            try:
                cpu = process.cpu_times()
                total += cpu.user + cpu.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _run(self):
        while not self.stop_event.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        self.cpu_process_start = self._cpu_seconds()
        self.wall_start = time.perf_counter()
        for process in self._process_tree():
            try:
                process.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.cpu_percent(None)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.thread.join()
        self._sample()
        self.cpu_process_end = self._cpu_seconds()
        self.wall_end = time.perf_counter()

    def results(self):
        logical_cores = psutil.cpu_count(logical=True) or 1
        average = 100.0 * (self.cpu_process_end - self.cpu_process_start) / max(
            self.wall_end - self.wall_start, 1e-9)
        return {
            "cpu_process_raw_peak_percent": round(self.cpu_process_peak, 2),
            "cpu_process_normalized_peak_percent": round(
                min(100.0, self.cpu_process_peak / logical_cores), 2),
            "cpu_process_average_percent": round(average, 2),
            "cpu_process_average_normalized_percent": round(
                min(100.0, average / logical_cores), 2),
            "cpu_system_peak_percent": round(self.cpu_system_peak, 2),
            "peak_rss_mb": round(self.rss_peak / 2 ** 20, 1),
        }


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
            "BENCHMARK_THREADS", "BENCHMARK_NOISE_MULTIPLIER",
            "BENCHMARK_SILENCE_DURATION",
            "DEEPSEEK_MODEL")},
        "configs": S.CONFIGS,
    }


def model_groups(config_names):
    groups = {}
    for config_name in config_names:
        config = S.CONFIGS[config_name]
        key = (config.get("backend", "openai_whisper"), config["model"],
               config.get("precision", "fp32"))
        groups.setdefault(key, []).append(config_name)
    return list(groups.values())


def run_subprocesses(args, clips, session, benchmark_threshold, affinity):
    """Run one fresh Python process per backend/model/precision group."""
    os.makedirs(RESULTS, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    env = environment(args.threads)
    env["affinity"] = affinity
    env["dataset"] = {"clips": len(clips), "mic": session.get("mic"),
                      "calibration": session["calibration"]}
    groups = model_groups(args.configs)
    env["run"] = {
        "configs": args.configs, "modes": args.modes, "repeats": args.repeats,
        "threads": args.threads, "subprocess_groups": groups,
        "endpoint_threshold": benchmark_threshold,
        "endpoint_noise_multiplier": S.BENCHMARK_NOISE_MULTIPLIER,
        "endpoint_silence_duration": S.BENCHMARK_SILENCE_DURATION,
        "shared_endpointing": True,
    }
    env_path = os.path.join(RESULTS, f"env_{stamp}.json")
    with open(env_path, "w", encoding="utf-8") as output:
        json.dump(env, output, indent=2)

    worker_dir = os.path.join(RESULTS, f".workers_{stamp}")
    os.makedirs(worker_dir, exist_ok=True)
    worker_paths = []
    try:
        for index, group in enumerate(groups):
            worker_path = os.path.join(worker_dir, f"group_{index}.csv")
            command = [sys.executable, os.path.abspath(__file__), "--worker",
                       "--configs", *group, "--modes", *args.modes,
                       "--repeats", str(args.repeats), "--limit", str(args.limit),
                       "--threads", str(args.threads), "--worker-output", worker_path]
            subprocess.run(command, cwd=HERE, check=True)
            worker_paths.append(worker_path)

        output_path = os.path.join(RESULTS, f"runs_{stamp}.csv")
        run_id = 0
        with open(output_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            for worker_path in worker_paths:
                with open(worker_path, newline="", encoding="utf-8") as worker_file:
                    for row in csv.DictReader(worker_file):
                        run_id += 1
                        row["run_id"] = run_id
                        writer.writerow(row)
        print(f"\nSelesai. Data mentah: {output_path}")
        print(f"Environment: {env_path}")
        return 0
    finally:
        shutil.rmtree(worker_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(S.CONFIGS))
    ap.add_argument("--modes", nargs="+", default=S.INTENT_MODES)
    ap.add_argument("--repeats", type=int, default=S.REPEATS)
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (testing)")
    ap.add_argument("--threads", type=int, default=S.BENCHMARK_THREADS,
                    help="set the same CPU thread count for every config")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
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
    noise_block_p95 = session["calibration"].get("noise_block_norm_p95")
    if noise_block_p95 is None:
        raise RuntimeError("session.json tidak memiliki noise_block_norm_p95; rekam kalibrasi ulang")
    benchmark_threshold = S.BENCHMARK_NOISE_MULTIPLIER * noise_block_p95
    if args.limit:
        clips = clips[: args.limit]

    affinity = apply_p_core_affinity()
    if not args.worker:
        return run_subprocesses(args, clips, session, benchmark_threshold, affinity)

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
    env["affinity"] = affinity
    env["dataset"] = {"clips": len(clips), "mic": session.get("mic"),
                      "calibration": session["calibration"]}
    env["run"] = {
        "configs": args.configs, "modes": args.modes, "repeats": args.repeats,
        "threads": args.threads,
        "endpoint_threshold": benchmark_threshold,
        "endpoint_noise_multiplier": S.BENCHMARK_NOISE_MULTIPLIER,
        "endpoint_silence_duration": S.BENCHMARK_SILENCE_DURATION,
        "shared_endpointing": True,
    }
    if not args.worker:
        with open(os.path.join(RESULTS, f"env_{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2)
    out_path = args.worker_output or os.path.join(RESULTS, f"runs_{stamp}.csv")
    out = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()

    print(f"CPU: {env['cpu']} | {env['logical_cores']} threads | RAM {env['ram_gb']} GB")
    print(f"{len(clips)} klip x {len(args.configs)} config x {args.repeats} ulangan, mode: {args.modes}")

    audio_cache = {c["clip_id"]: read_wav(os.path.join(DATA, "audio", c["file"])) for c in clips}
    prepared_clips = {}
    for clip in clips:
        audio = audio_cache[clip["clip_id"]]
        command = commands[clip["command_id"]]
        s_start, s_end = core.speech_bounds(audio, noise_rms)
        start = 0 if s_start is None else max(
            0, int((s_start - S.PRE_ROLL_SEC) * S.SAMPLE_RATE))
        segment, endpoint = core.endpoint_segment(
            audio, benchmark_threshold, S.BENCHMARK_SILENCE_DURATION,
            start_sample=start)
        truncated = bool(endpoint["endpointed"] and s_end is not None
                         and endpoint["cut_sec"] < s_end - 0.05)
        prepared_clips[clip["clip_id"]] = {
            "audio": audio,
            "command": command,
            "segment": segment,
            "endpoint": endpoint,
            "speech_start": s_start,
            "speech_end": s_end,
            "truncated": truncated,
        }
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
        # One unrecorded warm-up per config prevents lazy initialization from
        # entering the measured transcription latency.
        for warm_cfg_name in cfg_names:
            core.transcribe(model, np.zeros(S.SAMPLE_RATE, dtype=np.float32),
                            S.CONFIGS[warm_cfg_name])

        for cfg_name in cfg_names:
            cfg = S.CONFIGS[cfg_name]
            for rep in range(1, args.repeats + 1):
                order = clips[:]
                rng.shuffle(order)
                for n, clip in enumerate(order, 1):
                    prepared = prepared_clips[clip["clip_id"]]
                    audio = prepared["audio"]
                    cmd = prepared["command"]
                    seg = prepared["segment"]
                    ep = prepared["endpoint"]
                    s_start = prepared["speech_start"]
                    s_end = prepared["speech_end"]
                    truncated = prepared["truncated"]

                    t0 = time.perf_counter()
                    with ResourceSampler() as resources:
                        transcript, transcribe_info = core.transcribe(
                            model, seg, cfg, return_info=True)
                    t1 = time.perf_counter()
                    t_tr = t1 - t0
                    resource_info = resources.results()
                    cores = resource_info["cpu_process_raw_peak_percent"] / 100.0
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
                            **resource_info,
                            "t_intent_s": round(t_int, 9), "t_action_s": round(t_act, 9),
                            "latency_after_speech_s": None if total is None else round(total, 4),
                            "llm_error": info["error"],
                            "llm_prompt_tokens": info["prompt_tokens"],
                            "llm_completion_tokens": info["completion_tokens"],
                            "model_rss_mb": round(model_rss, 1),
                            "process_rss_mb": resource_info["peak_rss_mb"],
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
