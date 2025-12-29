### 1. Jalankan Configuration File

```bash
cd deepseek
python configuration-1.py

cd local_rule
python configuration-1.py
```

### 2. Lakukan beberapa test runs

Setiap kali Anda berbicara dan system mengclassify intent, metrics akan dicatat otomatis ke CSV-nya.

### 3. Compare hasil menggunakan script

```bash
python compare_metrics.py
```

Script ini akan:
- Load metrics dari `deepseek/metrics_log.csv` dan `local_rule/metrics_log.csv`
- Generate comparison visualization (bar charts)
- Print detailed statistics
- Save ringkasan ke `metrics_comparison_summary.csv`

## Metrics Comparison Output

Script akan menghasilkan:

1. **metrics_comparison.png** - Bar chart perbandingan:
   - Intent Classification Latency (utama untuk compare DeepSeek vs Local Rule)
   - Total End-to-End Latency
   - Transcription Latency
   - Recording Duration
   - CPU Usage
   - Memory Usage

2. **metrics_comparison_summary.csv** - Spreadsheet dengan mean values untuk semua metrics

3. **Console output** - Detailed statistics (mean, median, min, max) untuk setiap metric

## Key Metrics untuk Comparison

Fokus utama untuk membandingkan DeepSeek vs Local Rule:

1. **intent_classification_latency** - Kecepatan classify intent
   - DeepSeek: Lebih lambat tapi more accurate (API call)
   - Local Rule: Lebih cepat tapi hanya pattern matching

2. **total_latency** - End-to-end user experience
   - Lebih rendah = lebih responsive

3. **cpu_percent & memory_mb** - Resource efficiency
   - DeepSeek: Butuh API call (network overhead)
   - Local Rule: Pure local processing

## Notes
- Semua metrics **auto-reset** setelah setiap session
- CSV files **append mode** - tidak menghapus data lama
- Gunakan `compare_metrics.py` untuk analyze semua runs sekaligus
- Untuk reset metrics, cukup hapus file CSV-nya (akan recreate dengan headers baru)
