import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True

CSV_FILE = 'metrics_log.csv'
OUTPUT_DIR = 'graphs'
STATS_FILE = 'statistics_table.txt'

def load_metrics(csv_file=CSV_FILE):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} sessions from {csv_file}")
    return df

def plot_average_latency_breakdown(df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    latency_cols = []
    if 'transcription_latency' in df.columns:
        latency_cols.append('transcription_latency')
    if 'intent_classification_latency' in df.columns:
        latency_cols.append('intent_classification_latency')
    if 'execution_latency' in df.columns:
        latency_cols.append('execution_latency')
    if 'total_latency' in df.columns:
        latency_cols.append('total_latency')

    if not latency_cols:
        print("No relevant latency columns found. Skipping latency graph.")
        return

    means = df[latency_cols].astype(float).mean()

    labels = [c.replace('_', ' ').title().replace('Latency', '').strip() for c in latency_cols]
    values = [means[c] for c in latency_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, alpha=0.8)
    ax.set_title('Average Latency Breakdown')
    ax.set_ylabel('Average Time (seconds)')
    ax.set_ylim(0, max(values)*1.3 if len(values)>0 else 1)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}s", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'latency_breakdown.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_performance_metrics(df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    cpu_col = 'cpu_percent' if 'cpu_percent' in df.columns else None
    mem_col = 'memory_mb' if 'memory_mb' in df.columns else ('memory_percent' if 'memory_percent' in df.columns else None)

    if cpu_col is None and mem_col is None:
        print("No CPU/MEM data in CSV. Skipping performance graph.")
        return

    if cpu_col and mem_col:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes = [axes]

    idx = 0
    if cpu_col:
        df_cpu = df[df[cpu_col].notna()].copy()
        if len(df_cpu) > 0:
            df_cpu = df_cpu.reset_index(drop=True)
            x = range(1, len(df_cpu) + 1)
            axes[idx].plot(x, df_cpu[cpu_col].astype(float), marker='o', linestyle='-')
            mean_cpu = df_cpu[cpu_col].astype(float).mean()
            axes[idx].axhline(y=mean_cpu, color='red', linestyle='--', label=f"Mean: {mean_cpu:.1f}%")
            axes[idx].set_title('CPU Usage (%)')
            axes[idx].set_xlabel('Session Number')
            axes[idx].set_ylabel('CPU (%)')
            axes[idx].legend()
        else:
            axes[idx].text(0.5, 0.5, 'No CPU data', ha='center', va='center')
        idx += 1

    if mem_col:
        df_mem = df[df[mem_col].notna()].copy()
        if len(df_mem) > 0:
            df_mem = df_mem.reset_index(drop=True)
            x = range(1, len(df_mem) + 1)
            axes[idx].plot(x, df_mem[mem_col].astype(float), marker='o', linestyle='-')
            mean_mem = df_mem[mem_col].astype(float).mean()
            axes[idx].axhline(y=mean_mem, color='red', linestyle='--', label=f"Mean: {mean_mem:.1f}")
            axes[idx].set_title('Memory Usage')
            axes[idx].set_xlabel('Session Number')
            axes[idx].set_ylabel(mem_col.replace('_', ' ').title())
            axes[idx].legend()
        else:
            axes[idx].text(0.5, 0.5, 'No Memory data', ha='center', va='center')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'performance_metrics.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def generate_statistics_table(df, output_file=STATS_FILE):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STATISTICS SUMMARY\n")
        f.write("="*60 + "\n\n")

        latency_metrics = [m for m in ['transcription_latency', 'intent_classification_latency', 'execution_latency', 'total_latency'] if m in df.columns]

        if latency_metrics:
            f.write("LATENCY (seconds)\n")
            f.write("-"*60 + "\n")
            for metric in latency_metrics:
                data = df[metric].dropna().astype(float)
                if len(data) == 0:
                    continue
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Count:  {len(data)}\n")
                f.write(f"  Mean:   {data.mean():.4f}s\n")
                f.write(f"  Median: {data.median():.4f}s\n")
                f.write(f"  Std:    {data.std():.4f}s\n")
                f.write(f"  Min:    {data.min():.4f}s\n")
                f.write(f"  Max:    {data.max():.4f}s\n\n")
        else:
            f.write("No relevant latency data (transcription/execution/total)\n\n")

        perf_candidates = []
        if 'cpu_percent' in df.columns:
            perf_candidates.append(('CPU (%)', df['cpu_percent'].dropna().astype(float)))
        if 'memory_mb' in df.columns:
            perf_candidates.append(('Memory (MB)', df['memory_mb'].dropna().astype(float)))
        elif 'memory_percent' in df.columns:
            perf_candidates.append(('Memory (%)', df['memory_percent'].dropna().astype(float)))

        if perf_candidates:
            f.write("PERFORMANCE\n")
            f.write("-"*60 + "\n")
            for name, data in perf_candidates:
                if len(data) == 0:
                    continue
                f.write(f"{name}:\n")
                f.write(f"  Count:  {len(data)}\n")
                f.write(f"  Mean:   {data.mean():.2f}\n")
                f.write(f"  Median: {data.median():.2f}\n")
                f.write(f"  Std:    {data.std():.2f}\n")
                f.write(f"  Min:    {data.min():.2f}\n")
                f.write(f"  Max:    {data.max():.2f}\n\n")
        else:
            f.write("No performance data (CPU/MEM) in CSV\n\n")

        f.write("="*60 + "\n")

    print(f"Statistics table saved: {output_file}")

def main():
    print("\nMETRICS ANALYSIS")
    df = load_metrics()
    if df is None or len(df) == 0:
        print("No data to analyze")
        return

    print("\nGenerating visualizations...")
    plot_average_latency_breakdown(df)
    plot_performance_metrics(df)

    print("\nGenerating statistics...")
    generate_statistics_table(df)

    print("\nDone. Output files in 'graphs' folder and statistics_table.txt\n")

if __name__ == "__main__":
    main()
