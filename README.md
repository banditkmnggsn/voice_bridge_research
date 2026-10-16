# VoiceBridge Research

VoiceBridge adalah pipeline voice command lokal: microphone -> endpointing -> Whisper -> intent parser -> browser action.

Workflow utama untuk penelitian ada di [benchmark/README.md](benchmark/README.md). Folder `deepseek/` dan `local_rule/` adalah implementasi lama yang dipertahankan sebagai arsip.

## Struktur utama

```text
benchmark/                  # workflow eksperimen yang digunakan untuk PPT
benchmark/settings.py       # konfigurasi C1-C5 dan mode preprocessing
benchmark/commands.csv      # ground truth perintah pengujian
benchmark/run_benchmark.py  # menjalankan benchmark
benchmark/analyze_results.py# membuat laporan dan grafik
PPT_ALIGNMENT_CHECKLIST.md  # status kesesuaian code dengan PPT
```

## Mulai cepat

1. Siapkan Python dan dependency dari `benchmark/requirements.txt`.
2. Ikuti tutorial pengujian di [benchmark/README.md](benchmark/README.md).
3. Jangan commit `benchmark/.env`; file itu berisi credential dan sudah di-ignore Git.
