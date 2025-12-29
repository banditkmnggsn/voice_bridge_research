import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Get paths
deepseek_log = Path("deepseek/metrics_log.csv")
local_rule_log = Path("local_rule/metrics_log.csv")

# Check if files exist
if not deepseek_log.exists():
    print(f"❌ {deepseek_log} tidak ditemukan")
    exit(1)

if not local_rule_log.exists():
    print(f"❌ {local_rule_log} tidak ditemukan")
    exit(1)

# Read CSV files with encoding
print("[*] Loading metrics...")
try:
    deepseek_df = pd.read_csv(deepseek_log, encoding='utf-8', on_bad_lines='skip')
    local_rule_df = pd.read_csv(local_rule_log, encoding='utf-8', on_bad_lines='skip')
except Exception as e:
    print(f"[ERROR] Error reading CSV files: {e}")
    print("\n[*] Trying to fix CSV files...")
    
    # Clean and fix CSV files
    def clean_csv(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Keep only complete rows (those that start with date)
        cleaned_lines = [lines[0]]  # Keep header
        for line in lines[1:]:
            if line.startswith('2025-'):  # Only keep lines starting with date
                cleaned_lines.append(line)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        print(f"[+] Fixed {filepath}")
    
    clean_csv(deepseek_log)
    clean_csv(local_rule_log)
    
    # Try again
    deepseek_df = pd.read_csv(deepseek_log, encoding='utf-8', on_bad_lines='skip')
    local_rule_df = pd.read_csv(local_rule_log, encoding='utf-8', on_bad_lines='skip')

print(f"[+] DeepSeek: {len(deepseek_df)} records")
print(f"[+] Local Rule: {len(local_rule_df)} records")

# Check if intent_classification_latency column exists, if not add it
if 'intent_classification_latency' not in deepseek_df.columns:
    print("\n[!] Adding missing 'intent_classification_latency' column to deepseek data")
    deepseek_df.insert(8, 'intent_classification_latency', '')

if 'intent_classification_latency' not in local_rule_df.columns:
    print("[!] Adding missing 'intent_classification_latency' column to local_rule data")
    local_rule_df.insert(8, 'intent_classification_latency', '')

# Add model label
deepseek_df['model'] = 'DeepSeek'
local_rule_df['model'] = 'Local Rule'

# Helper function to safely get column
def get_safe_col(df, col_name):
    return df[col_name].dropna() if col_name in df.columns else pd.Series([])

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('DeepSeek vs Local Rule Performance Comparison', fontsize=16, fontweight='bold')

# 1. Intent Classification Latency
ax = axes[0, 0]
deepseek_intent = get_safe_col(deepseek_df, 'intent_classification_latency')
local_rule_intent = get_safe_col(local_rule_df, 'intent_classification_latency')

if len(deepseek_intent) > 0 and len(local_rule_intent) > 0:
    ax.bar(['DeepSeek', 'Local Rule'], 
           [deepseek_intent.mean(), local_rule_intent.mean()],
           color=['#FF6B6B', '#4ECDC4'],
           alpha=0.7,
           edgecolor='black',
           linewidth=1.5)
    ax.set_ylabel('Latency (seconds)', fontweight='bold')
    ax.set_title('Intent Classification Latency\n(Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([deepseek_intent.mean(), local_rule_intent.mean()]):
        ax.text(i, v + 0.01, f'{v:.3f}s', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No intent classification data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=11, style='italic')
    ax.set_title('Intent Classification Latency', fontweight='bold')

# 2. Total Latency
ax = axes[0, 1]
deepseek_total = get_safe_col(deepseek_df, 'total_latency')
local_rule_total = get_safe_col(local_rule_df, 'total_latency')

if len(deepseek_total) > 0 and len(local_rule_total) > 0:
    ax.bar(['DeepSeek', 'Local Rule'],
           [deepseek_total.mean(), local_rule_total.mean()],
           color=['#FF6B6B', '#4ECDC4'],
           alpha=0.7,
           edgecolor='black',
           linewidth=1.5)
    ax.set_ylabel('Latency (seconds)', fontweight='bold')
    ax.set_title('Total End-to-End Latency\n(Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([deepseek_total.mean(), local_rule_total.mean()]):
        ax.text(i, v + 0.01, f'{v:.3f}s', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No total latency data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=11, style='italic')
    ax.set_title('Total End-to-End Latency', fontweight='bold')

# 3. Transcription Latency
ax = axes[0, 2]
deepseek_trans = get_safe_col(deepseek_df, 'transcription_latency')
local_rule_trans = get_safe_col(local_rule_df, 'transcription_latency')

if len(deepseek_trans) > 0 and len(local_rule_trans) > 0:
    ax.bar(['DeepSeek', 'Local Rule'],
           [deepseek_trans.mean(), local_rule_trans.mean()],
           color=['#FF6B6B', '#4ECDC4'],
           alpha=0.7,
           edgecolor='black',
           linewidth=1.5)
    ax.set_ylabel('Latency (seconds)', fontweight='bold')
    ax.set_title('Transcription Latency\n(Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([deepseek_trans.mean(), local_rule_trans.mean()]):
        ax.text(i, v + 0.01, f'{v:.3f}s', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No transcription latency data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=11, style='italic')
    ax.set_title('Transcription Latency', fontweight='bold')

# 4. Recording Duration
ax = axes[1, 0]
deepseek_rec = get_safe_col(deepseek_df, 'recording_latency')
local_rule_rec = get_safe_col(local_rule_df, 'recording_latency')

if len(deepseek_rec) > 0 and len(local_rule_rec) > 0:
    ax.bar(['DeepSeek', 'Local Rule'],
           [deepseek_rec.mean(), local_rule_rec.mean()],
           color=['#FF6B6B', '#4ECDC4'],
           alpha=0.7,
           edgecolor='black',
           linewidth=1.5)
    ax.set_ylabel('Duration (seconds)', fontweight='bold')
    ax.set_title('Recording Duration\n(Actual Audio Length)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([deepseek_rec.mean(), local_rule_rec.mean()]):
        ax.text(i, v + 0.01, f'{v:.3f}s', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No recording duration data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=11, style='italic')
    ax.set_title('Recording Duration', fontweight='bold')

# 5. CPU Usage
ax = axes[1, 1]
deepseek_cpu = get_safe_col(deepseek_df, 'cpu_percent')
local_rule_cpu = get_safe_col(local_rule_df, 'cpu_percent')

if len(deepseek_cpu) > 0 and len(local_rule_cpu) > 0:
    ax.bar(['DeepSeek', 'Local Rule'],
           [deepseek_cpu.mean(), local_rule_cpu.mean()],
           color=['#FF6B6B', '#4ECDC4'],
           alpha=0.7,
           edgecolor='black',
           linewidth=1.5)
    ax.set_ylabel('CPU Usage (%)', fontweight='bold')
    ax.set_title('Average CPU Usage\n(Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([deepseek_cpu.mean(), local_rule_cpu.mean()]):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No CPU data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=11, style='italic')
    ax.set_title('Average CPU Usage', fontweight='bold')

# 6. Memory Usage
ax = axes[1, 2]
deepseek_mem = get_safe_col(deepseek_df, 'memory_mb')
local_rule_mem = get_safe_col(local_rule_df, 'memory_mb')

if len(deepseek_mem) > 0 and len(local_rule_mem) > 0:
    ax.bar(['DeepSeek', 'Local Rule'],
           [deepseek_mem.mean(), local_rule_mem.mean()],
           color=['#FF6B6B', '#4ECDC4'],
           alpha=0.7,
           edgecolor='black',
           linewidth=1.5)
    ax.set_ylabel('Memory (MB)', fontweight='bold')
    ax.set_title('Average Memory Usage\n(Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([deepseek_mem.mean(), local_rule_mem.mean()]):
        ax.text(i, v + 1, f'{v:.1f}MB', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No memory data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=11, style='italic')
    ax.set_title('Average Memory Usage', fontweight='bold')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("\n[+] Grafik perbandingan disimpan ke: metrics_comparison.png")
plt.show()

# Print detailed statistics
print("\n" + "="*70)
print("DETAILED METRICS COMPARISON")
print("="*70)

print("\n🎯 INTENT CLASSIFICATION LATENCY (seconds)")
print("-" * 70)
if len(deepseek_intent) > 0:
    print(f"DeepSeek  | Mean: {deepseek_intent.mean():.4f}s | Median: {deepseek_intent.median():.4f}s | Min: {deepseek_intent.min():.4f}s | Max: {deepseek_intent.max():.4f}s")
else:
    print("DeepSeek  | No data")
    
if len(local_rule_intent) > 0:
    print(f"Local Rule| Mean: {local_rule_intent.mean():.4f}s | Median: {local_rule_intent.median():.4f}s | Min: {local_rule_intent.min():.4f}s | Max: {local_rule_intent.max():.4f}s")
else:
    print("Local Rule| No data")

print("\n⏱️ TOTAL END-TO-END LATENCY (seconds)")
print("-" * 70)
if len(deepseek_total) > 0:
    print(f"DeepSeek  | Mean: {deepseek_total.mean():.4f}s | Median: {deepseek_total.median():.4f}s | Min: {deepseek_total.min():.4f}s | Max: {deepseek_total.max():.4f}s")
else:
    print("DeepSeek  | No data")
    
if len(local_rule_total) > 0:
    print(f"Local Rule| Mean: {local_rule_total.mean():.4f}s | Median: {local_rule_total.median():.4f}s | Min: {local_rule_total.min():.4f}s | Max: {local_rule_total.max():.4f}s")
else:
    print("Local Rule| No data")

print("\n🎤 TRANSCRIPTION LATENCY (seconds)")
print("-" * 70)
if len(deepseek_trans) > 0:
    print(f"DeepSeek  | Mean: {deepseek_trans.mean():.4f}s | Median: {deepseek_trans.median():.4f}s | Min: {deepseek_trans.min():.4f}s | Max: {deepseek_trans.max():.4f}s")
else:
    print("DeepSeek  | No data")
    
if len(local_rule_trans) > 0:
    print(f"Local Rule| Mean: {local_rule_trans.mean():.4f}s | Median: {local_rule_trans.median():.4f}s | Min: {local_rule_trans.min():.4f}s | Max: {local_rule_trans.max():.4f}s")
else:
    print("Local Rule| No data")

print("\n💻 CPU USAGE (%)")
print("-" * 70)
if len(deepseek_cpu) > 0:
    print(f"DeepSeek  | Mean: {deepseek_cpu.mean():.2f}% | Median: {deepseek_cpu.median():.2f}% | Min: {deepseek_cpu.min():.2f}% | Max: {deepseek_cpu.max():.2f}%")
else:
    print("DeepSeek  | No data")
    
if len(local_rule_cpu) > 0:
    print(f"Local Rule| Mean: {local_rule_cpu.mean():.2f}% | Median: {local_rule_cpu.median():.2f}% | Min: {local_rule_cpu.min():.2f}% | Max: {local_rule_cpu.max():.2f}%")
else:
    print("Local Rule| No data")

print("\n🧠 MEMORY USAGE (MB)")
print("-" * 70)
if len(deepseek_mem) > 0:
    print(f"DeepSeek  | Mean: {deepseek_mem.mean():.2f}MB | Median: {deepseek_mem.median():.2f}MB | Min: {deepseek_mem.min():.2f}MB | Max: {deepseek_mem.max():.2f}MB")
else:
    print("DeepSeek  | No data")
    
if len(local_rule_mem) > 0:
    print(f"Local Rule| Mean: {local_rule_mem.mean():.2f}MB | Median: {local_rule_mem.median():.2f}MB | Min: {local_rule_mem.min():.2f}MB | Max: {local_rule_mem.max():.2f}MB")
else:
    print("Local Rule| No data")

# Save detailed comparison to CSV
comparison_summary = pd.DataFrame({
    'Metric': [
        'Intent Classification Latency (s)',
        'Total Latency (s)',
        'Transcription Latency (s)',
        'Recording Duration (s)',
        'CPU Usage (%)',
        'Memory Usage (MB)'
    ],
    'DeepSeek Mean': [
        deepseek_intent.mean() if len(deepseek_intent) > 0 else 'N/A',
        deepseek_total.mean() if len(deepseek_total) > 0 else 'N/A',
        deepseek_trans.mean() if len(deepseek_trans) > 0 else 'N/A',
        deepseek_rec.mean() if len(deepseek_rec) > 0 else 'N/A',
        deepseek_cpu.mean() if len(deepseek_cpu) > 0 else 'N/A',
        deepseek_mem.mean() if len(deepseek_mem) > 0 else 'N/A'
    ],
    'Local Rule Mean': [
        local_rule_intent.mean() if len(local_rule_intent) > 0 else 'N/A',
        local_rule_total.mean() if len(local_rule_total) > 0 else 'N/A',
        local_rule_trans.mean() if len(local_rule_trans) > 0 else 'N/A',
        local_rule_rec.mean() if len(local_rule_rec) > 0 else 'N/A',
        local_rule_cpu.mean() if len(local_rule_cpu) > 0 else 'N/A',
        local_rule_mem.mean() if len(local_rule_mem) > 0 else 'N/A'
    ]
})

comparison_summary.to_csv('metrics_comparison_summary.csv', index=False)
print("\n[+] Ringkasan perbandingan disimpan ke: metrics_comparison_summary.csv")
print("\n" + "="*70)
