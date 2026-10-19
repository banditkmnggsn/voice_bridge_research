"""
Step 4 - turn the raw runs CSV into the numbers we present.

Outputs (in results/):
  summary_<stamp>.csv      one row per config x intent mode x split
  report_<stamp>.md        readable tables + environment + transcription errors
  latency_<stamp>.png      where the waiting time goes, per config and mode
  success_<stamp>.png      task success rate with 95% confidence intervals

Usage: python analyze_results.py [results/runs_XXXX.csv]   (default: latest)
"""

import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def to_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def summarise(df):
    rows = []
    group_columns = ["config", "intent_mode"]
    for optional in ("backend", "precision"):
        if optional in df.columns:
            group_columns.insert(1, optional)
    for group_key, g0 in df.groupby(group_columns, sort=False):
        values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
        cfg = values["config"]
        mode = values["intent_mode"]
        for split in ["all", "original", "paraphrase"]:
            g = g0 if split == "all" else g0[g0["split"] == split]
            if g.empty:
                continue
            per_clip = g.groupby("clip_id").agg(
                success=("task_success", "mean"), intent=("intent_correct", "mean"),
                wer=("wer", "first"), wake=("wake_detected", "first"),
                trunc=("truncated", "first"))
            n = len(per_clip)
            sr = per_clip["success"].mean()
            lo, hi = wilson(sr, n)
            tot = g["latency_after_speech_s"].dropna()
            rows.append({
                **values, "split": split,
                "n_clips": n, "n_runs": len(g),
                "task_success_rate": round(sr, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                "intent_accuracy": round(per_clip["intent"].mean(), 4),
                "wer_mean": round(per_clip["wer"].mean(), 4),
                "wake_miss_rate": round(1 - per_clip["wake"].mean(), 4),
                "truncation_rate": round(per_clip["trunc"].mean(), 4),
                "endpoint_wait_mean_s": round(g["endpoint_wait_s"].mean(), 4),
                "transcribe_mean_s": round(g["t_transcribe_s"].mean(), 4),
                "transcribe_median_s": round(g["t_transcribe_s"].median(), 4),
                "transcribe_p95_s": round(g["t_transcribe_s"].quantile(0.95), 4),
                "transcribe_std_s": round(g["t_transcribe_s"].std(), 4),
                "cpu_process_raw_peak_mean": round(g["cpu_process_raw_peak_percent"].mean(), 2),
                "cpu_process_normalized_peak_mean": round(
                    g["cpu_process_normalized_peak_percent"].mean(), 2),
                "cpu_process_average_mean": round(g["cpu_process_average_percent"].mean(), 2),
                "cpu_process_average_normalized_mean": round(
                    g["cpu_process_average_normalized_percent"].mean(), 2),
                "cpu_system_peak_mean": round(g["cpu_system_peak_percent"].mean(), 2),
                "peak_rss_mean_mb": round(g["peak_rss_mb"].mean(), 1),
                "intent_mean_s": round(g["t_intent_s"].mean(), 6),
                "intent_median_s": round(g["t_intent_s"].median(), 6),
                "intent_p95_s": round(g["t_intent_s"].quantile(0.95), 6),
                "total_mean_s": round(tot.mean(), 4) if len(tot) else None,
                "total_median_s": round(tot.median(), 4) if len(tot) else None,
                "total_p95_s": round(tot.quantile(0.95), 4) if len(tot) else None,
                "no_endpoint_runs": int((~g["endpointed"]).sum()),
                "cores_transcribe_mean": round(g["cores_transcribe"].mean(), 2),
                "model_rss_mb": round(g["model_rss_mb"].iloc[0], 1),
                "llm_errors": int((g["llm_error"].fillna("") != "").sum()),
            })
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_latency(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[summary["split"] == "all"].reset_index(drop=True)
    labels = [f"{r.config} · {r.intent_mode}" for r in s.itertuples()]
    parts = [("Wait for silence", "endpoint_wait_mean_s", BLUE),
             ("Transcription", "transcribe_mean_s", ORANGE),
             ("Intent", "intent_mean_s", AQUA)]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(s) + 1.6), facecolor=SURFACE)
    y = np.arange(len(s))[::-1]
    left = np.zeros(len(s))
    for name, col, color in parts:
        vals = s[col].fillna(0).values
        ax.barh(y, vals, left=left, height=0.55, color=color, label=name,
                edgecolor=SURFACE, linewidth=2)
        left += vals
    for yi, tot, ne in zip(y, left, s["no_endpoint_runs"].values):
        note = "  (silence never detected!)" if ne > 0 else ""
        ax.text(tot + 0.03, yi, f"{tot:.2f} s{note}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel("Mean time after the user stops speaking (s)", color=INK2)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 if len(s) > 4 else -0.3),
              ncol=3, frameon=False, labelcolor=INK)
    ax.set_title("Where the waiting time goes", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_success(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[summary["split"] != "all"]
    cfgs = list(dict.fromkeys(s["config"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SURFACE)
    for ax, split in zip(axes, ["original", "paraphrase"]):
        sub = s[s["split"] == split]
        x = np.arange(len(cfgs))
        for k, (mode, color) in enumerate([("rule", BLUE), ("llm", ORANGE)]):
            m = sub[sub["intent_mode"] == mode].set_index("config").reindex(cfgs)
            if m["task_success_rate"].isna().all():
                continue
            v = m["task_success_rate"].values * 100
            err = np.vstack([v - m["ci95_low"].values * 100, m["ci95_high"].values * 100 - v])
            xs = x + (k - 0.5) * 0.36
            ax.bar(xs, v, width=0.34, color=color, label="Rule-based" if mode == "rule" else "DeepSeek LLM",
                   edgecolor=SURFACE, linewidth=2)
            ax.errorbar(xs, v, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, color=INK)
        ax.set_ylim(0, 105)
        ax.set_title("Paper commands" if split == "original" else "Paraphrased commands",
                     loc="left", color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_ylabel("Task success (%)", color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, labelcolor=INK,
               loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        files = sorted(glob.glob(os.path.join(RESULTS, "runs_*.csv")))
        if not files:
            print("Belum ada results/runs_*.csv. Jalankan 3_benchmark.bat dulu.")
            return 1
        path = files[-1]
    stamp = os.path.basename(path)[len("runs_"):-len(".csv")]
    df = pd.read_csv(path)
    for c in ("endpointed", "truncated", "wake_detected", "intent_correct",
              "query_correct", "task_success"):
        df[c] = to_bool(df[c])
    summary = summarise(df)
    summary.to_csv(os.path.join(RESULTS, f"summary_{stamp}.csv"), index=False)

    env_path = os.path.join(RESULTS, f"env_{stamp}.json")
    env = json.load(open(env_path, encoding="utf-8")) if os.path.exists(env_path) else {}

    key_columns = [c for c in ("config", "backend", "precision", "intent_mode") if c in summary.columns]
    key = summary[summary["split"] == "all"][
        key_columns + ["n_clips", "task_success_rate", "ci95_low", "ci95_high",
        "wer_mean", "wake_miss_rate", "truncation_rate", "transcribe_mean_s",
        "cpu_process_raw_peak_mean", "cpu_process_normalized_peak_mean",
        "cpu_process_average_mean", "cpu_process_average_normalized_mean",
        "cpu_system_peak_mean", "peak_rss_mean_mb",
        "intent_median_s", "total_mean_s", "total_p95_s", "no_endpoint_runs",
        "cores_transcribe_mean",
        "model_rss_mb", "llm_errors"]]
    split_tbl = summary[summary["split"] != "all"][[
        "config", "intent_mode", "split", "n_clips", "task_success_rate", "intent_accuracy"]]
    first = df[(df["repeat"] == 1) & (df["intent_mode"] == df["intent_mode"].iloc[0])]
    errors = first[first["wer"] > 0][["config", "clip_id", "ground_truth", "transcript", "wer"]]
    fails = df[(df["repeat"] == 1) & (~df["task_success"])][[
        "config", "intent_mode", "clip_id", "command_text", "pred_intent", "pred_query",
        "exp_intent", "exp_query"]]

    lines = [f"# VoiceBridge benchmark report ({stamp})", "",
             f"Source: `{os.path.basename(path)}` ({len(df)} rows)", "",
             "## Environment", "", "```json",
             json.dumps({k: env.get(k) for k in ("date", "os", "cpu", "physical_cores",
                                                  "logical_cores", "ram_gb", "python",
                                                  "torch", "torch_threads", "whisper",
                                                  "dataset", "run")}, indent=2), "```", "",
             "## Main results (all commands)", "", md_table(key), "",
             "## By command set", "", md_table(split_tbl), "",
             "## Transcription errors (repeat 1)", "", md_table(errors), "",
             "## Failed tasks (repeat 1)", "", md_table(fails), "",
             "Definitions: task success = correct intent AND correct query. "
             "Latency = wait for silence + transcription + intent + building the URL "
             "(browser launch excluded). CI = Wilson 95% interval over clips."]
    with open(os.path.join(RESULTS, f"report_{stamp}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    plot_latency(summary, os.path.join(RESULTS, f"latency_{stamp}.png"))
    plot_success(summary, os.path.join(RESULTS, f"success_{stamp}.png"))

    pd.set_option("display.width", 200)
    print(key.to_string(index=False))
    print(f"\nLaporan: results/report_{stamp}.md (+ grafik PNG)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
