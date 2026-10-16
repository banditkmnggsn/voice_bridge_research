# VoiceBridge Research

VoiceBridge adalah pipeline voice command lokal: microphone -> endpointing -> Whisper -> intent parser -> browser action.

Workflow utama untuk penelitian ada di [benchmark/guide.md](benchmark/guide.md). Folder `deepseek/` dan `local_rule/` adalah implementasi lama yang dipertahankan sebagai arsip.

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
2. Ikuti tutorial pengujian di [benchmark/guide.md](benchmark/guide.md).
3. Jangan commit `benchmark/.env`; file itu berisi credential dan sudah di-ignore Git.

## Definisi Pengukuran

Semua angka utama untuk PPT dibuat oleh benchmark v2. Detail prosedur dan command ada di [benchmark/guide.md](benchmark/guide.md).

- **WER:** reference berasal dari `commands.csv[text]`, hypothesis berasal dari transcript Whisper. Keduanya dinormalisasi lalu dibandingkan dengan dynamic programming pada seluruh kata. `WER = (substitution + deletion + insertion) / jumlah kata reference`.
- **Endpointing:** audio dimulai dari `segment_start_sec` dengan pre-roll sebelum speech. Audio berhenti di `cut_sec` setelah volume berada di bawah threshold selama `silence_duration`. `endpoint_wait_s` adalah waktu dari blok keras terakhir sampai endpoint berhenti.
- **Transcription latency:** dimulai tepat sebelum `core.transcribe()` dan berhenti ketika transcript selesai. Ini mencakup preprocessing dan inferensi Whisper, tetapi tidak mencakup endpointing, intent, atau browser.
- **Encoder/decoder:** forward hook mengukur `model.encoder` dan `model.decoder` OpenAI Whisper selama transkripsi. `n_tokens` berasal dari token segment hasil Whisper.
- **Intent latency:** dimulai sebelum `intent_rule()` atau `intent_llm()` dan berhenti setelah intent/query dikembalikan. Transkripsi dan browser tidak termasuk.
- **Latency setelah bicara:** `endpoint_wait_s + t_transcribe_s + t_intent_s + t_action_s`. Browser tidak dibuka dalam benchmark utama.
- **Task success:** sukses jika intent benar dan query benar. Query harus memuat semua content word expected dan maksimal dua content word tambahan.
- **CPU:** selisih process CPU time dibagi transcription wall time, sehingga hasilnya estimasi jumlah core proses yang digunakan, bukan CPU sistem global.
- **Memory:** `model_rss_mb` adalah kenaikan RSS saat model dimuat; `process_rss_mb` adalah RSS proses ketika hasil ditulis.
- **Privacy:** `bytes_sent=0` untuk rule-based. Mode LLM mencatat ukuran JSON request; `audio_bytes` mencatat ukuran audio lokal yang tidak dikirim ke LLM.
