# VoiceBridge Research

VoiceBridge adalah pipeline voice command lokal: microphone -> endpointing -> Whisper -> intent parser -> browser action.

Workflow utama untuk penelitian ada di [benchmark/guide.md](benchmark/guide.md). Folder `deepseek/` dan `local_rule/` adalah implementasi lama yang dipertahankan sebagai arsip.

## Struktur utama

```text
benchmark/                  # workflow eksperimen yang digunakan untuk PPT
benchmark/settings.py       # konfigurasi C1, C2, C3, C3n, C3z, C4, C5
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
- **Endpointing benchmark:** setiap clip dipotong satu kali sebelum loop konfigurasi. Threshold bersama adalah `2.5 x noise_block_norm_p95` dan silence duration bersama `0.6 s`, sehingga semua konfigurasi menerima segment audio identik. Threshold/silence per-config hanya dipakai demo live. `endpoint_wait_s` adalah waktu dari blok keras terakhir sampai endpoint berhenti.
- **Transcription latency:** dimulai tepat sebelum `core.transcribe()` dan berhenti ketika transcript selesai. Ini mencakup preprocessing dan inferensi Whisper, tetapi tidak mencakup endpointing, intent, atau browser.
- **Encoder/decoder:** forward hook mengukur `model.encoder` dan `model.decoder` OpenAI Whisper selama transkripsi. `n_tokens` berasal dari token segment hasil Whisper.
- **Intent latency:** dimulai sebelum `intent_rule()` atau `intent_llm()` dan berhenti setelah intent/query dikembalikan. Transkripsi dan browser tidak termasuk.
- **Latency setelah bicara:** `endpoint_wait_s + t_transcribe_s + t_intent_s + t_action_s`. Browser tidak dibuka dalam benchmark utama.
- **Task success:** sukses jika intent benar dan query benar. Query harus memuat semua content word expected dan maksimal dua content word tambahan.
- **CPU average:** selama transkripsi, sampler mengambil process CPU time dari process tree (proses utama dan child process). Rumus raw yang disimpan adalah:

	```text
	cpu_process_average_percent =
			((process_cpu_time_after - process_cpu_time_before)
			 / transcription_wall_time) x 100
	```

	`process_cpu_time_*` adalah jumlah detik user + system CPU dari `psutil.Process(os.getpid())` dan child process. `transcription_wall_time` dimulai tepat sebelum `core.transcribe()` dan berhenti setelah transcript selesai. Karena CPU time dapat berjalan pada beberapa core, hasil raw bisa lebih dari 100%. Contoh `250%` berarti kira-kira 2.5 core terpakai rata-rata selama fase transcription.

	`cpu_process_average_normalized_percent` dihitung sebagai `raw / logical_core_count`, dibatasi 0-100%. Contoh pada 12 logical core: `250% / 12 = 20.83%`.

	`cpu_process_raw_peak_percent` dan `cpu_process_normalized_peak_percent` adalah nilai peak dari sampling interval 75 ms. `cpu_system_peak_percent` adalah CPU seluruh sistem dan hanya digunakan untuk mendeteksi gangguan aplikasi lain.
- **Memory:** `peak_rss_mb` adalah peak RSS process tree selama transcription. `model_rss_mb` adalah kenaikan RSS saat model dimuat di subprocess model yang bersih; `process_rss_mb` dipertahankan sebagai alias peak RSS.
- **Model isolation:** setiap grup `openai-whisper base`, `openai-whisper tiny`, `faster-whisper tiny INT8`, dan `faster-whisper base INT8` berjalan di subprocess Python terpisah. Parent hanya menggabungkan worker CSV, sehingga RSS antar-model tidak tercampur. Setiap config mendapat satu warm-up yang tidak dicatat dan waktu loading model tidak masuk latency.
- **CPU affinity:** pada Windows hybrid CPU, `detect_p_cores()` membaca `all_cpu_efficiency_classes` dari Windows CPU Set API. `EfficiencyClass` terbesar dipilih sebagai P-core. Pada i3-1220P, pemetaan yang diharapkan adalah logical index `0-3` sebagai P-core dan `4-11` sebagai E-core. Affinity diterapkan ke parent dan worker, lalu disimpan di `env.json` bersama `p_core_indices` dan `applied_affinity`. Standard benchmark memakai 2 thread untuk Torch dan faster-whisper.
- **Normalized CPU bukan pembagian P-core:** nilai `*_normalized_*` dibagi `logical_core_count` (12 logical core pada mesin ini), bukan 4 P-core. Affinity hanya membatasi CPU yang boleh dipakai; denominator normalisasi tetap jumlah logical core sistem.
- **Decoding:** semua backend memakai mode greedy. OpenAI Whisper membiarkan `beam_size=None`; faster-whisper menerima `beam_size=1`, `best_of=1`, dan `vad_filter=False`.
- **Test output dan validasi:** `run_benchmark.py --test` menulis ke `benchmark/results/_test/`. `analyze_results.py --test` membaca folder itu. `validate_run.py` hanya membaca CSV/env dan menulis `validation_<stamp>.md` tanpa menjalankan model.
- **Privacy:** `bytes_sent=0` untuk rule-based. Mode LLM mencatat ukuran JSON request; `audio_bytes` mencatat ukuran audio lokal yang tidak dikirim ke LLM.
