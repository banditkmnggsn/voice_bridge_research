# VoiceBridge benchmark v2

Pengukuran ulang VoiceBridge untuk IEEE CITISMA 2026. Folder lama (`deepseek/`, `local_rule/`) **tidak diubah** dan tetap disimpan sebagai arsip.

## Cara pakai (double-click berurutan)

| Langkah | File | Siapa | Waktu |
|---|---|---|---|
| 0 | Salin `.env.example` jadi `.env`, lalu tempel API key DeepSeek | Stephen | 1 menit |
| 1 | `1_cek_setup.bat` (diam 3 detik saat diminta) | otomatis | ±2 menit |
| 2 | `2_rekam.bat` (rekam 39 klip suara) | **Stephen bicara** | ±15 menit |
| 3 | `3_benchmark.bat` | otomatis | ±25 menit |
| 4 | `4_analisis.bat` | otomatis | <1 menit |
| – | `5_demo_live.bat` (demo pakai mic langsung) | Stephen | – |

Tips merekam: ruangan sepi, kipas dan AC mati kalau bisa, jarak mulut ke mic laptop **50 cm** (ukur pakai meteran, sama seperti setup di paper), dan bicara dengan tempo normal. Ulangi take hanya kalau ada salah ucap, batuk, atau gangguan suara. Rekaman yang sudah tersimpan aman: kalau script dihentikan, jalankan lagi dan sesi akan dilanjutkan.

## Desain eksperimen

- **Dataset:** 11 perintah. 3 perintah dari paper diambil masing-masing 5 take, dan 8 parafrase masing-masing 3 take, jadi totalnya 39 klip WAV (16 kHz). Semua konfigurasi memakai file yang sama persis, jadi inputnya setara.
- **Konfigurasi:** C1, C2, dan C3 memakai parameter dari paper. C3n adalah ablation (normalisasi saja, tanpa prompt), untuk memisahkan efek masing-masing komponen.
- **Intent:** rule-based dan DeepSeek dijalankan pada transkrip yang **sama**.
- **Ulangan:** 3× per klip per konfigurasi, dengan urutan diacak (seed tetap).
- **Metrik:**
  - *Task success* = intent benar **dan** query benar. Selang kepercayaan Wilson 95% dihitung per klip.
  - WER terhadap ground truth.
  - *Wake-word miss*.
  - *Truncation*: rekaman terpotong sebelum kalimat selesai.
  - Latency per tahap, diukur dengan `perf_counter`.
  - CPU proses (jumlah core yang dipakai saat transkripsi).
  - Memori model.
  - Spesifikasi perangkat dicatat otomatis di `results/env_*.json`.
- **Latency setelah bicara** = tunggu hening + transkripsi + intent + membangun URL. Waktu membuka browser **tidak** termasuk; demo live mengukurnya terpisah.
- Semua angka ditulis otomatis ke `results/runs_*.csv`. Tidak ada angka yang diketik manual.

## Perbedaan dari kode lama (wajib disebut di paper/presentasi)

1. **Vocabulary prompt Config 3 sebelumnya tidak berfungsi.** Kode lama mengirimnya sebagai `prompt=`, padahal `whisper.transcribe()` menimpa parameter itu (lihat `transcribe.py`: `decode_options["prompt"] = all_tokens[...]`). Di v2 prompt dikirim lewat `initial_prompt=`, sehingga benar-benar dipakai.
2. **Total latency lama tidak memuat transkripsi.** Timer "total" baru dimulai setelah wake word terdeteksi, yaitu setelah transkripsi selesai. Di v2, latency dihitung sejak user berhenti bicara.
3. **CPU lama diukur di level sistem** (`psutil.cpu_percent()`), dan diambil setelah browser dibuka. Di v2, yang diukur adalah CPU proses VoiceBridge saat transkripsi.
4. **Rule-based lama memetakan "find" ke putar YouTube.** Di v2, "find", "search", dan "look up" dipetakan ke Google search, sesuai deskripsi paper.
5. **Ukuran blok audio dikunci** ke nilai default driver di PC ini, yang diukur oleh `check_setup.py` dan disimpan di `audio_device.json`. Dengan begitu, threshold hening punya arti yang tetap.
6. **Folder `deepseek/` lama tidak berisi pemanggilan DeepSeek.** Kodenya identik dengan `local_rule/`. Di v2, integrasi DeepSeek ditulis ulang di `voicebridge_core.py` (`deepseek-chat`, temperature 0, output JSON, 3 contoh few-shot).
7. **Sistem berjalan sebagai aplikasi Python lokal di CPU**, bukan TensorFlow.js di browser seperti yang tertulis di paper.
