from local_rule.metrics_monitor import MetricsMonitor
import numpy as np
import time

# ===================== CONFIG =====================

GROUND_TRUTH_COMMANDS = [
    {
        "ground_truth": "hey voice bridge open youtube home",
        "intent": "open_youtube_home",
        "query": ""
    },
    {
        "ground_truth": "hey voice bridge play justin bieber",
        "intent": "play_youtube",
        "query": "justin bieber"
    },
    {
        "ground_truth": "hey voice bridge search binusmaya",
        "intent": "search_google",
        "query": "binusmaya"
    }
]

REPEAT_PER_COMMAND = 2
LOG_FILE = "metrics_log.csv"

# ==================================================

def run_wer_experiment():
    metrics = MetricsMonitor(log_file=LOG_FILE)

    session_id = 1

    for cmd in GROUND_TRUTH_COMMANDS:
        for run in range(REPEAT_PER_COMMAND):
            print(f"\n▶️ Session {session_id} | Command: {cmd['ground_truth']} | Run {run+1}")

            # ================== RECORDING ==================
            metrics.mark_wake_word_detected()
            metrics.start_recording()

            print("🎤 Silakan ucapkan atau putar rekaman sekarang...")
            time.sleep(3)  # waktu kamu ngomong / muter audio

            metrics.end_recording()

            # ================= TRANSCRIPTION =================
            metrics.start_transcription()

            # ⚠️ DI SINI kamu ambil OUTPUT ASLI AI
            # Contoh:
            # transcribed_text = run_voicebridge_stt()

            transcribed_text = input("📝 Masukkan hasil transkripsi AI: ").strip()

            metrics.end_transcription()

            # ================= EXECUTION =================
            metrics.start_execution()
            time.sleep(0.05)  # simulasi eksekusi
            metrics.end_execution()

            # Dummy audio (opsional, hanya untuk metadata)
            audio_data = np.random.randn(16000, 1).astype("float32")

            # ================= LOGGING =================
            metrics.log_session(
                wake_word="hey voice bridge",
                transcribed_text=transcribed_text,
                command=cmd["ground_truth"],
                intent=cmd["intent"],
                query=cmd["query"],
                audio_data=audio_data,
                ground_truth=cmd["ground_truth"],
                notes=f"WER test run {run+1}"
            )

            session_id += 1

    print("\n✅ WER experiment selesai.")
    print(f"📄 Data tersimpan di: {LOG_FILE}")
    print("📊 Jalankan analyze_metrics.py untuk grafik & statistik")

if __name__ == "__main__":
    run_wer_experiment()
