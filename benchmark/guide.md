# VoiceBridge Benchmark Guide

Panduan resmi untuk menyiapkan, merekam, menjalankan, dan membaca benchmark VoiceBridge. Jalankan semua perintah dari folder `benchmark`.

## 1. Siapkan environment

Gunakan Python 3.11 atau versi yang kompatibel dengan dependency project.

```powershell
cd benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Jika virtual environment sudah dibuat oleh project, cukup aktifkan:

```powershell
.\.venv\Scripts\Activate.ps1
```

Untuk mode LLM OpenRouter:

```powershell
Copy-Item .env.example .env
```

Isi `.env` dengan key OpenRouter. Jangan commit file `.env`.

## 2. Urutan pengujian

### Step 1: pemeriksaan setup

```powershell
.\1_cek_setup.bat
```

Pemeriksaan ini memvalidasi dependency, dua backend Whisper, microphone, sample rate, model, dan koneksi LLM jika mode LLM dikonfigurasi. Jika muncul peringatan noise floor di atas threshold, jangan lanjut merekam; kecilkan gain microphone atau pindah ke ruangan yang lebih tenang lalu ulangi step ini.

### Step 2: rekam dataset

```powershell
.\2_rekam.bat
```

Dataset terdiri dari 11 perintah:

- 3 perintah original, masing-masing 5 take.
- 8 perintah paraphrase, masing-masing 3 take.
- Total: 39 klip.

Microphone direkam pada `default_samplerate` perangkat. Audio kemudian di-downsample ke 16 kHz dengan `scipy.signal.resample_poly`, yang menyediakan filter anti-aliasing. `manifest.csv` menyimpan sample rate sumber dan sample rate output.

Kondisi recording:

- Ruangan tenang, kipas/AC dimatikan jika memungkinkan.
- Jarak mulut ke microphone sekitar 50 cm.
- Ucapkan satu perintah dengan tempo normal.
- Retake hanya untuk salah ucap, batuk, atau gangguan suara.
- Jika proses berhenti, jalankan lagi; klip yang sudah tersimpan akan dilewati.

### Step 3: benchmark

Semua konfigurasi memakai file audio yang sama. Benchmark memotong setiap clip satu kali sebelum loop konfigurasi; threshold endpoint benchmark adalah `2.5 x noise_block_norm_p95` dari kalibrasi dan silence duration bersama adalah `0.6 s`. Threshold/silence per-config pada `settings.py` hanya digunakan oleh demo live.

```powershell
.\3_benchmark.bat
```

Konfigurasi transcription:

| Config | Backend | Model | Precision | Preprocessing |
|---|---|---|---|---|
| C1 | openai-whisper | base | FP32 | none |
| C2 | openai-whisper | tiny | FP32 | none |
| C3 | openai-whisper | tiny | FP32 | peak + vocabulary prompt |
| C3n | openai-whisper | tiny | FP32 | peak, no prompt |
| C3z | openai-whisper | tiny | FP32 | z-score + vocabulary prompt |
| C4 | faster-whisper | tiny | INT8 | peak + vocabulary prompt |
| C5 | faster-whisper | base | INT8 | peak + vocabulary prompt |

Intent parser:

- `rule`: keyword matching lokal, tidak membutuhkan internet.
- `llm`: mengirim teks transkripsi ke OpenRouter; audio tidak dikirim.

Contoh command:

```powershell
python run_benchmark.py --modes rule
python run_benchmark.py --modes llm
python run_benchmark.py --configs C1 C4 --modes rule --repeats 1 --limit 3 --threads 2
```

`--limit` dan `--repeats 1` hanya untuk smoke test. Untuk angka PPT, gunakan dataset penuh dan repeat default.

### Step 4: analisis

```powershell
.\4_analisis.bat
```

Output dibuat di `results/`:

- `runs_*.csv`: data mentah setiap kombinasi clip/config/parser/repeat.
- `summary_*.csv`: rata-rata dan ringkasan per config/backend/precision/parser, termasuk CPU peak/average dan peak RSS.
- `report_*.md`: laporan hasil.
- `latency_*.png`: breakdown latency.
- `success_*.png`: task success dengan Wilson 95% confidence interval.
- `env_*.json`: hardware, software, thread, dataset, dan parameter run.

## 3. Cara code menghitung metrik

### 3.1 Ground truth dan clip

`commands.csv` berisi `text`, `intent`, dan `query` yang benar. `run_benchmark.py` mencocokkan `command_id` dari `dataset/manifest.csv` ke baris tersebut.

Setiap clip diproses tiga kali secara default. Urutan clip diacak dengan seed tetap. Satu baris CSV dibuat untuk setiap:

```text
clip x configuration x repeat x intent_mode
```

### 3.2 WER

WER dihitung di `voicebridge_core.wer(reference, hypothesis)` setelah teks dinormalisasi:

1. lowercase;
2. tanda baca dihapus;
3. tanda `-` diubah menjadi spasi;
4. spasi dirapikan;
5. teks dipecah menjadi kata.

Reference adalah `commands.csv[text]`. Hypothesis adalah teks yang dikembalikan Whisper.

Rumusnya:

```text
WER = (S + D + I) / N
```

- `S`: substitution, kata reference diganti kata lain.
- `D`: deletion, kata reference hilang.
- `I`: insertion, kata tambahan di hypothesis.
- `N`: jumlah kata pada reference.

Code menggunakan dynamic programming Levenshtein pada seluruh range kata reference dan hypothesis, dari indeks pertama sampai kata terakhir. WER dihitung untuk setiap clip/configuration/repeat, lalu diringkas oleh `analyze_results.py`.

### 3.3 Wake-word dan command

Whisper menghasilkan transcript penuh. `split_wake_word()` menormalisasi transcript lalu mencari frasa `hey voice bridge`.

- Sebelum frasa `hey voice bridge`: dianggap bagian wake-word/transcript.
- Setelah frasa `hey voice bridge`: menjadi `command_text`.
- Jika frasa tersebut tidak ditemukan, `wake_detected=False`, intent tidak dijalankan, dan hasil dicatat sebagai `no_wake_word`.

### 3.4 Endpointing dan truncation

`run_benchmark.py` memanggil `endpoint_segment()` satu kali untuk setiap clip, sebelum C1-C5 dijalankan. Segment hasilnya disimpan di `prepared_clips` dan object audio yang sama diberikan ke setiap konfigurasi dan repeat. Ini mencegah C1 dan C2 memiliki panjang input berbeda hanya karena threshold silence masing-masing.

Threshold benchmark dihitung sebagai:

```text
benchmark_threshold = 2.5 x noise_block_norm_p95
benchmark_silence_duration = 0.6 s
```

`endpoint_segment()` memproses audio dari `segment_start_sec` dengan blok audio sebesar `BLOCKSIZE`.

Recording berhenti ketika volume L2 sebuah blok berada di bawah `silence_threshold` selama lebih dari `silence_duration`.

- `segment_start_sec`: awal audio yang dikirim ke Whisper, dengan pre-roll sampai 0.20 detik sebelum speech.
- `cut_sec`: akhir segment yang dikirim ke Whisper.
- `endpoint_wait_s`: `cut_sec - last_loud_sec`, yaitu waktu tunggu setelah blok keras terakhir sampai endpoint berhenti.
- `truncated=True` jika endpoint berhenti lebih dari 0.05 detik sebelum `speech_end_sec` referensi.

Pada `demo_live.py`, threshold dan silence duration per-config tetap digunakan karena demo harus merespons secara langsung. Nilainya tidak digunakan untuk endpointing benchmark utama.

### 3.5 Transcription latency

Timer transcription dimulai tepat sebelum `core.transcribe(model, seg, cfg)` dipanggil dan berhenti setelah fungsi mengembalikan transcript.

```text
t_transcribe_s = t_after_transcribe - t_before_transcribe
```

Range ini mencakup preprocessing konfigurasi dan inferensi Whisper, tetapi tidak mencakup endpointing, intent parser, atau membuka browser.

### 3.6 Encoder dan decoder latency

Untuk OpenAI Whisper, forward hook dipasang pada `model.encoder` dan `model.decoder` selama pemanggilan transcribe.

- `t_encoder_s`: jumlah waktu forward pada encoder.
- `t_decoder_s`: jumlah waktu forward pada decoder, termasuk pemanggilan decoder berulang selama decoding token.
- `n_tokens`: jumlah token pada segment hasil Whisper.

Untuk faster-whisper, CSV tetap menyediakan kolom timing tersebut, tetapi implementasi internal CTranslate2 tidak mengekspos module `model.encoder`/`model.decoder` seperti OpenAI Whisper. Karena itu timing internal faster-whisper harus diperlakukan sebagai belum tersedia sampai ada instrumentation backend yang didukung.

### 3.7 Intent latency

Untuk setiap transcript yang memiliki wake word dan command:

- rule: timer dimulai tepat sebelum `intent_rule(command_text)` dan berhenti setelah return.
- llm: timer dimulai tepat sebelum `intent_llm(command_text, ...)` dan berhenti setelah response diproses menjadi `intent` dan `query`.

Intent latency tidak mencakup transcription dan tidak mencakup membuka browser.

### 3.8 Action latency dan latency setelah bicara

`action_url(pred_intent, pred_query)` hanya membangun URL; tidak membuka browser.

```text
latency_after_speech_s = endpoint_wait_s
                         + t_transcribe_s
                         + t_intent_s
                         + t_action_s
```

Timer ini merepresentasikan waktu sejak user selesai bicara sampai URL siap. Waktu membuka browser tidak termasuk dalam benchmark utama. Demo live mencetak waktu browser secara terpisah.

Jika endpoint tidak mendeteksi hening, `latency_after_speech_s` ditulis kosong karena endpoint wait tidak dapat ditentukan.

### 3.9 Task success

Task dianggap sukses jika dua syarat benar:

```text
intent_correct = predicted_intent == expected_intent
query_correct = intent_correct AND expected_query cocok dengan predicted_query
task_success = intent_correct AND query_correct
```

Query dinilai cocok jika semua content word dari expected query ada pada predicted query dan predicted query menambah maksimal dua content word. Stop words seperti `the`, `to`, `please`, dan `in` tidak dihitung.

`analyze_results.py` menghitung task success per clip terlebih dahulu, lalu rata-rata antar clip. Wilson 95% confidence interval dihitung dari jumlah clip, bukan dari jumlah repeat.

### 3.10 CPU dan memory

Setiap grup model dijalankan dalam subprocess Python baru. Grup yang dipisahkan adalah OpenAI Whisper base, OpenAI Whisper tiny, faster-whisper tiny INT8, dan faster-whisper base INT8. Parent process hanya menggabungkan CSV worker, sehingga RSS antar-model tidak tercampur.

Selama `core.transcribe()` saja, `ResourceSampler` berjalan di background thread setiap 75 ms. Sampler mengagregasi process utama dan child process dari `psutil.Process(os.getpid())`.

CSV menyimpan:

- `cpu_process_raw_peak_percent`: peak CPU process-tree mentah; dapat lebih dari 100% saat memakai beberapa core.
- `cpu_process_normalized_peak_percent`: peak CPU process-tree dibagi jumlah logical core, dibatasi 0-100%.
- `cpu_process_average_percent`: `(process_cpu_time_after - process_cpu_time_before) / transcription_wall_time x 100`.
- `cpu_process_average_normalized_percent`: CPU average raw dibagi 12 logical core pada mesin ini, dibatasi 0-100%. Ini bukan dibagi 4 P-core; affinity P-core dan denominator normalisasi adalah dua hal berbeda.
- `cpu_system_peak_percent`: peak CPU seluruh sistem, hanya indikator gangguan aplikasi lain.
- `peak_rss_mb`: peak RSS process-tree selama transcription.

`cores_transcribe` adalah nilai kompatibilitas yang diturunkan dari CPU process raw peak, bukan pengukuran system-wide. Waktu model loading dan warm-up tidak masuk latency atau resource per clip. Setiap config mendapat satu warm-up yang tidak ditulis ke CSV.

Benchmark standard memakai 2 thread. Nilai `run.threads`, `requested_threads`, dan `torch_threads` ditulis di `env_*.json`; faster-whisper menerima nilai yang sama sebagai `cpu_threads`.

Pada Windows hybrid CPU, benchmark mendeteksi logical index P-core melalui Windows CPU Set API, membatasi affinity setiap parent/worker process ke P-core tersebut, dan menyimpan `p_core_indices`, `efficiency_class`, serta `applied_affinity` di `env_*.json`.

Semua backend memakai decoding greedy. OpenAI Whisper tidak diberi `beam_size`; faster-whisper diberi `beam_size=1`, `best_of=1`, dan `vad_filter=False`. Nilai `DECODING="greedy"` dicatat di environment.

Untuk output terpisah saat validasi:

```powershell
python run_benchmark.py --test --modes rule --repeats 1 --limit 3
python analyze_results.py --test
python validate_run.py --test
```

`validate_run.py` bersifat read-only dan hanya membaca hasil yang sudah ada.

Memory model tetap dicatat sebagai `model_rss_mb`, yaitu kenaikan RSS saat model backend dimuat. `process_rss_mb` adalah alias kompatibilitas yang berisi nilai `peak_rss_mb`.

Spesifikasi CPU, RAM, OS, Python, Torch, Whisper, thread, backend, dan konfigurasi dicatat dalam `env_*.json`.

### 3.11 Privacy bytes

Untuk setiap command:

- `audio_bytes = seg.nbytes`: ukuran audio lokal yang diproses Whisper.
- `bytes_sent = 0` untuk mode rule.
- `bytes_sent = ukuran JSON request` untuk mode LLM, sebelum request dikirim.

Audio tidak dimasukkan ke request LLM. Mode rule dapat diuji offline dengan:

```powershell
python run_benchmark.py --modes rule
```

## 4. Membaca hasil untuk PPT

Jangan mengisi angka PPT dari contoh lama atau `metrics_comparison_summary.csv` arsip. Gunakan `summary_*.csv` dan `report_*.md` dari benchmark v2 terbaru.

- Slide latency: `latency_*.png` dan kolom `endpoint_wait_mean_s`, `transcribe_mean_s`, `intent_mean_s`.
- Slide model/precision: bandingkan `backend`, `precision`, `model_rss_mb`, `transcribe_mean_s`, dan `wer_mean`.
- Slide accuracy: `wer_mean`, `task_success_rate`, `ci95_low`, dan `ci95_high`.
- Slide privacy: `bytes_sent` versus `audio_bytes`.
- Slide hardware: `env_*.json`.

Angka dianggap valid untuk presentasi hanya setelah command selesai, file hasil terbentuk, dan isi hasil diperiksa.
