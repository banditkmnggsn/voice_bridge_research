# VoiceBridge Benchmark v2

Tutorial pengujian yang digunakan untuk melengkapi PPT CITISMA 2026. Semua perintah dijalankan dari folder `benchmark`.

## 1. Siapkan environment

Buat virtual environment dan install dependency dari `requirements.txt` di komputer yang memiliki Python valid. `torch`, Whisper, dan modelnya berukuran besar, jadi lakukan instalasi dengan koneksi yang memadai.

```powershell
cd benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Untuk OpenRouter, salin `.env.example` menjadi `.env` lalu isi key. `.env` sudah di-ignore Git dan tidak boleh di-commit.

```powershell
Copy-Item .env.example .env
```

## 2. Jalankan pengujian

Jalankan berurutan:

```powershell
.\1_cek_setup.bat
.\2_rekam.bat
.\3_benchmark.bat
.\4_analisis.bat
```

Keterangan:

- `1_cek_setup.bat`: memeriksa microphone, dependency, model, dan koneksi LLM.
- `2_rekam.bat`: merekam 39 klip berdasarkan `commands.csv`. Rekaman diambil pada sample rate microphone lalu di-resample ke 16 kHz.
- `3_benchmark.bat`: menjalankan semua konfigurasi dan parser pada dataset yang sama.
- `4_analisis.bat`: membuat tabel ringkasan, laporan Markdown, dan grafik.

Untuk menguji rule-based saja tanpa request jaringan:

```powershell
python run_benchmark.py --modes rule
```

Untuk membatasi pengujian saat debugging:

```powershell
python run_benchmark.py --modes rule --configs C1 --repeats 1 --limit 3 --threads 4
```

## 3. Konfigurasi dan data

Konfigurasi transcription ada di `settings.py`:

- C1-C3/C3n/C3z: OpenAI Whisper FP32.
- C4: faster-whisper tiny INT8.
- C5: faster-whisper base INT8.

Parser intent adalah `rule` dan `llm`. Mode `llm` memakai OpenRouter untuk key `sk-or-...`; mode `rule` tidak mengirim audio atau request jaringan.

Dataset berisi 11 perintah: 3 perintah original dengan 5 take dan 8 parafrase dengan 3 take, total 39 klip. Jangan mengubah `commands.csv` setelah recording dimulai.

## 4. Hasil pengujian

File dibuat di `results/`:

- `runs_*.csv`: data mentah per clip, konfigurasi, parser, WER, task success, latency, memory, precision, dan metadata privasi.
- `summary_*.csv`: ringkasan per konfigurasi, backend, precision, parser, dan split dataset.
- `report_*.md`: laporan yang siap diperiksa.
- `latency_*.png` dan `success_*.png`: grafik hasil.
- `env_*.json`: hardware, software, thread, dan parameter eksperimen.

Jangan mengetik atau mengubah angka hasil secara manual. Isi slide PPT hanya dari file hasil benchmark.

## 5. Kondisi rekaman

Gunakan ruangan tenang, matikan kipas/AC bila memungkinkan, dan jaga jarak mulut ke microphone sekitar 50 cm. Ucapkan setiap perintah sekali dengan tempo normal. Retake hanya jika salah ucap, batuk, atau ada gangguan suara. Jika proses dihentikan, jalankan `2_rekam.bat` lagi untuk melanjutkan klip yang belum tersimpan.

## 6. Demo live

Setelah setup selesai:

```powershell
.\5_demo_live.bat
```

Demo membuka browser setelah intent dikenali. Waktu membuka browser dipisahkan dari latency benchmark utama.

Untuk status kesesuaian implementasi dengan PPT, lihat `../PPT_ALIGNMENT_CHECKLIST.md`.
