"""Read-only validation for a completed VoiceBridge benchmark run."""

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def median(values):
    return statistics.median(values) if values else None


def add(checks, status, message):
    checks.append((status, message))


def validate(path):
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    stamp = os.path.basename(path)[len("runs_"):-len(".csv")]
    env_path = os.path.join(folder, f"env_{stamp}.json")
    checks = []

    with open(path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with open(env_path, encoding="utf-8") as source:
        env = json.load(source)

    run = env.get("run", {})
    affinity = env.get("affinity", {})
    classes = {str(k): int(v) for k, v in affinity.get(
        "all_cpu_efficiency_classes", {}).items()}
    expected_p = sorted(int(k) for k, v in classes.items() if v == max(classes.values())) if classes else []
    actual_p = sorted(int(v) for v in affinity.get("p_core_indices", []))
    add(checks, "PASS" if expected_p and actual_p == expected_p else "FAIL",
        f"Affinity P-core indices: expected {expected_p}, recorded {actual_p}")

    threads = run.get("threads", env.get("requested_threads"))
    add(checks, "PASS" if threads == 2 and env.get("torch_threads") == 2 else "FAIL",
        f"Threads: run={threads}, torch={env.get('torch_threads')}")
    add(checks, "PASS" if run.get("decoding", env.get("settings", {}).get("DECODING")) == "greedy" else "FAIL",
        f"Decoding: {run.get('decoding', env.get('settings', {}).get('DECODING'))}")

    clips = int(env.get("dataset", {}).get("clips", 0))
    configs = run.get("configs", sorted({row.get("config") for row in rows}))
    repeats = int(run.get("repeats", 0))
    # Use the requested modes, so a silently skipped LLM mode counts as missing.
    modes = list(run.get("modes") or sorted({row.get("intent_mode") for row in rows}))
    expected_rows = clips * len(configs) * repeats * len(modes)
    add(checks, "PASS" if len(rows) == expected_rows else "FAIL",
        f"Completeness: {len(rows)} rows, expected {expected_rows} from {clips} clips x {len(configs)} configs x {repeats} repeats x {len(modes)} modes {modes}")
    present = {(row.get("config"), row.get("intent_mode")) for row in rows}
    missing_pairs = [f"{config}/{mode}" for config in configs for mode in modes
                     if (config, mode) not in present]
    add(checks, "PASS" if not missing_pairs else "FAIL",
        f"Config x mode coverage: missing {', '.join(missing_pairs) or 'none'}")

    missing_transcription = sum(not row.get("t_transcribe_s") for row in rows)
    add(checks, "PASS" if missing_transcription == 0 else "FAIL",
        f"Transcription latency missing: {missing_transcription} rows")

    endpoint_fields = ("segment_start_sec", "cut_sec", "audio_bytes")
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row.get("clip_id")].append(tuple(row.get(field) for field in endpoint_fields))
    different_audio = sum(len(set(values)) > 1 for values in by_clip.values())
    add(checks, "PASS" if different_audio == 0 else "FAIL",
        f"Identical endpointed audio: {different_audio} clips differ across configs")

    by_config_clip = defaultdict(list)
    for row in rows:
        value = as_float(row.get("t_transcribe_s"))
        if value is not None:
            by_config_clip[(row.get("config"), row.get("clip_id"))].append(value)
    cvs = []
    for values in by_config_clip.values():
        if len(values) > 1 and median(values):
            cvs.append(statistics.pstdev(values) / median(values))
    median_cv = median(cvs)
    add(checks, "WARN" if median_cv is not None and median_cv > 0.15 else "PASS",
        f"Transcription stability median CV: {median_cv:.4f}" if median_cv is not None else "Transcription stability: insufficient repeats")

    first_last = []
    by_config_repeat = defaultdict(list)
    for row in rows:
        value = as_float(row.get("t_transcribe_s"))
        repeat = row.get("repeat")
        if value is not None and repeat:
            by_config_repeat[(row.get("config"), int(repeat))].append(value)
    for config in configs:
        repeat_values = {repeat: median(values) for (name, repeat), values in by_config_repeat.items() if name == config}
        if 1 in repeat_values and repeats in repeat_values and repeat_values[1]:
            delta = abs(repeat_values[repeats] - repeat_values[1]) / repeat_values[1]
            first_last.append(delta)
    median_delta = median(first_last)
    add(checks, "WARN" if median_delta is not None and median_delta > 0.10 else "PASS",
        f"First versus last repeat median delta: {median_delta:.4f}" if median_delta is not None else "First versus last repeat: insufficient repeats")

    # Outliers: one transcription per (config, repeat, clip); rule and llm rows share it.
    details = []
    transcriptions = {}
    for row in rows:
        value = as_float(row.get("t_transcribe_s"))
        key = (row.get("config"), row.get("repeat"), row.get("clip_id"))
        if value is not None and key not in transcriptions:
            transcriptions[key] = (value, row.get("timestamp", ""))
    config_medians = {}
    for config in configs:
        values = [value for (name, _, _), (value, _) in transcriptions.items() if name == config]
        config_medians[config] = median(values)
    outliers = []
    for (config, repeat, clip_id), (value, timestamp) in transcriptions.items():
        config_median = config_medians.get(config)
        if config_median and value > 3 * config_median:
            outliers.append((timestamp, config, clip_id, repeat, value, config_median))
    outliers.sort()
    add(checks, "WARN" if outliers else "PASS",
        f"Transcription outliers (> 3x config median): {len(outliers)}"
        + (" -> " + "; ".join(f"{ts[11:19]} {cfg} {clip}" for ts, cfg, clip, *_ in outliers[:10])
           + (" ..." if len(outliers) > 10 else "") if outliers else ""))
    details.extend(f"- OUTLIER {ts[11:19]} {cfg} {clip} repeat={rep}: t_transcribe_s={value:.3f} (median {med:.3f})"
                   for ts, cfg, clip, rep, value, med in outliers)

    # Gaps: a pause > 20 s between consecutive rows of one config suggests sleep/stall.
    gaps = []
    last_time = {}
    for row in rows:
        try:
            moment = dt.datetime.fromisoformat(row.get("timestamp", ""))
        except ValueError:
            continue
        config = row.get("config")
        if config in last_time:
            gap = (moment - last_time[config]).total_seconds()
            if gap > 20:
                gaps.append((last_time[config].strftime("%H:%M:%S"), moment.strftime("%H:%M:%S"),
                             config, row.get("clip_id"), gap))
        last_time[config] = moment
    add(checks, "WARN" if gaps else "PASS",
        f"Gaps > 20 s within a config: {len(gaps)}"
        + (" -> " + "; ".join(f"{start}-{end} {cfg} ({gap:.0f} s)" for start, end, cfg, _, gap in gaps[:10])
           + (" ..." if len(gaps) > 10 else "") if gaps else ""))
    details.extend(f"- GAP {start} -> {end} {cfg} before {clip}: {gap:.0f} s"
                   for start, end, cfg, clip, gap in gaps)

    system_values = [as_float(row.get("cpu_system_peak_percent")) for row in rows]
    system_values = [value for value in system_values if value is not None]
    interference = 100.0 * sum(value > 90 for value in system_values) / len(system_values) if system_values else 0.0
    add(checks, "WARN" if interference > 5 else "PASS",
        f"System CPU peak above 90%: {interference:.2f}% of rows")

    medians = defaultdict(list)
    for row in rows:
        value = as_float(row.get("peak_rss_mb"))
        if value is not None:
            medians[row.get("config")].append(value)
    memory_medians = {config: median(values) for config, values in medians.items()}
    for left, right, label in (("C1", "C2", "C1 > C2"), ("C4", "C2", "C4 < C2"), ("C5", "C1", "C5 < C1")):
        if left in memory_medians and right in memory_medians:
            valid = memory_medians[left] > memory_medians[right] if label == "C1 > C2" else memory_medians[left] < memory_medians[right]
            add(checks, "PASS" if valid else "WARN", f"Memory sanity {label}: {memory_medians[left]:.1f} vs {memory_medians[right]:.1f} MB")

    rule_rows = [row for row in rows if row.get("intent_mode") == "rule"]
    bad_privacy = sum(as_float(row.get("bytes_sent")) != 0 for row in rule_rows)
    add(checks, "PASS" if bad_privacy == 0 else "FAIL",
        f"Rule privacy bytes_sent nonzero: {bad_privacy} rows")

    llm_errors = sum(bool(row.get("llm_error")) for row in rows)
    add(checks, "WARN" if llm_errors else "PASS", f"LLM errors: {llm_errors}")

    encoder_violations = 0
    for row in rows:
        if row.get("backend") == "openai_whisper":
            encoder = as_float(row.get("t_encoder_s"))
            decoder = as_float(row.get("t_decoder_s"))
            transcription = as_float(row.get("t_transcribe_s"))
            if None not in (encoder, decoder, transcription) and encoder + decoder > transcription:
                encoder_violations += 1
    add(checks, "PASS" if encoder_violations == 0 else "FAIL",
        f"OpenAI encoder+decoder exceeds transcription: {encoder_violations} rows")

    info = defaultdict(list)
    for row in rows:
        info[row.get("config")].append(row)
    info_lines = []
    for config, config_rows in sorted(info.items()):
        wers = [as_float(row.get("wer")) for row in config_rows]
        wakes = [str(row.get("wake_detected")).lower() in ("true", "1") for row in config_rows]
        wers = [value for value in wers if value is not None]
        wer_mean = sum(wers) / len(wers) if wers else float("nan")
        wake_miss_rate = 1 - sum(wakes) / len(wakes) if wakes else float("nan")
        info_lines.append(f"INFO {config}: wer_mean={wer_mean:.4f}, wake_miss_rate={wake_miss_rate:.4f}")
    for line in info_lines:
        print(line)

    status_counts = {status: sum(item_status == status for item_status, _ in checks) for status in ("PASS", "WARN", "FAIL")}
    lines = [f"# Validation {stamp}", "", f"Source: `{os.path.basename(path)}`", "", "## Summary", ""]
    lines.append(f"PASS={status_counts['PASS']} WARN={status_counts['WARN']} FAIL={status_counts['FAIL']}")
    lines.extend(["", "## Checks", "", "| Status | Check |", "|---|---|"])
    lines.extend(f"| {status} | {message} |" for status, message in checks)
    lines.extend(["", "## Info", ""])
    lines.extend(f"- {line}" for line in info_lines)
    if details:
        lines.extend(["", "## Details", ""])
        lines.extend(details)
    output_path = os.path.join(folder, f"validation_{stamp}.md")
    with open(output_path, "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    print(f"PASS={status_counts['PASS']} WARN={status_counts['WARN']} FAIL={status_counts['FAIL']}")
    print(f"Validation report: {output_path}")
    return 1 if status_counts["FAIL"] else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="runs CSV path")
    parser.add_argument("--test", action="store_true", help="read latest results/_test/runs_*.csv")
    args = parser.parse_args()
    folder = os.path.join(RESULTS, "_test") if args.test else RESULTS
    path = os.path.abspath(args.path) if args.path else None
    if not path:
        files = sorted(glob.glob(os.path.join(folder, "runs_*.csv")))
        if not files:
            raise SystemExit(f"No runs CSV found in {folder}")
        path = files[-1]
    raise SystemExit(validate(path))


if __name__ == "__main__":
    main()
