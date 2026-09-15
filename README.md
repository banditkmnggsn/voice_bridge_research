# Voice Bridge Research

Riset kecil untuk membandingkan performa **voice assistant pipeline** dengan dua pendekatan intent classification: **DeepSeek (LLM API)** vs **Local Rule (keyword matching)**. Alur dasarnya: dengar wake word → rekam sampai hening → transkripsi pakai **Whisper** → klasifikasi intent → eksekusi aksi (buka YouTube, putar video, atau search Google).

## Struktur

```
deepseek/     # Konfigurasi intent classification via DeepSeek
local_rule/   # Konfigurasi intent classification via rule-based lokal
compare_metrics.py   # Bandingkan hasil metrics kedua pendekatan
test_wer.py           # Uji akurasi transkripsi (Word Error Rate)
verify_setup.py        # Cek kelengkapan setup sebelum testing
```

Tiap folder (`deepseek/`, `local_rule/`) punya 3 variasi konfigurasi (`configuration-1/2/3.py` — beda parameter seperti ukuran model Whisper dan silence threshold) plus `metrics_monitor.py` untuk mencatat latency, CPU, memory, dan akurasi tiap sesi ke `metrics_log.csv`.

## Cara kerja

1. Jalankan salah satu `configuration-*.py` di `deepseek/` atau `local_rule/` → tiap sesi bicara otomatis tercatat ke CSV.
2. Jalankan `compare_metrics.py` → menghasilkan grafik perbandingan (`metrics_comparison.png`) dan ringkasan (`metrics_comparison_summary.csv`) untuk metrik seperti intent classification latency, total latency, CPU/memory usage.
3. `test_wer.py` dipakai terpisah untuk mengukur akurasi transkripsi dibanding ground truth.

## Dependencies

Whisper (STT), sounddevice, numpy, requests, psutil (monitoring), pandas + matplotlib (analisis & visualisasi) — lihat `requirements.txt` di masing-masing folder.
