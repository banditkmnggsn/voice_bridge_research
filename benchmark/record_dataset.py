"""
Step 2 - record the test dataset (the only step that needs a human voice).

For every command in commands.csv you record TAKES_PER_COMMAND takes. Each take
is a fixed CLIP_SECONDS window: wait for "GO", say the command once at a normal
pace, then stay quiet until the window closes.

Retake rule (write it in the paper): a take may be redone only for speaker
error (wrong words, cough, background noise). No model runs during recording,
so retakes cannot be chosen based on results.

Output: dataset/audio/*.wav (16 kHz mono int16), dataset/manifest.csv,
        dataset/session.json (mic name, noise floor, date)
"""

import csv
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

import settings as S
import voicebridge_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
AUDIO = os.path.join(DATA, "audio")


def load_commands():
    with open(os.path.join(HERE, "commands.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record(seconds, samplerate):
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate,
                   channels=S.CHANNELS, dtype="float32")
    sd.wait()
    return audio.flatten()


def save_wav(path, audio):
    pcm = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, S.SAMPLE_RATE, (pcm * 32767).astype(np.int16))
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def calibrate(source_rate):
    print(f"\nKalibrasi: DIAM selama {S.CALIBRATION_SECONDS:.0f} detik (jangan bicara)...")
    time.sleep(1.0)
    audio = core.resample_audio(record(S.CALIBRATION_SECONDS, source_rate), source_rate)
    n = len(audio) // S.BLOCKSIZE
    norms = [float(np.linalg.norm(audio[i * S.BLOCKSIZE:(i + 1) * S.BLOCKSIZE])) for i in range(n)]
    rms = float(np.sqrt(np.mean(audio ** 2)))
    noise_dbfs = float(20.0 * np.log10(rms)) if rms > 0 else float("-inf")
    p95 = float(np.percentile(norms, 95))
    print(f"  Noise floor: RMS={rms:.5f} ({noise_dbfs:.2f} dBFS) | block-norm p95={p95:.4f}")
    lowest = min(c["silence_threshold"] for c in S.CONFIGS.values())
    if p95 >= lowest:
        print(f"  !! Noise ruangan ({p95:.3f}) >= threshold terendah ({lowest}).")
        print("     Deteksi diam bisa tidak pernah aktif. Cari ruangan lebih sepi,")
        print("     matikan kipas/AC, atau kecilkan gain mic, lalu ulangi.")
        return {"noise_rms": rms, "noise_dbfs": noise_dbfs,
            "noise_block_norm_p95": p95,
            "noise_block_norm_mean": float(np.mean(norms))}


def countdown():
    for c in ("3", "2", "1"):
        print(f"  {c}...", end=" ", flush=True)
        time.sleep(0.6)
    print(">>> GO! <<<", flush=True)


def main():
    os.makedirs(AUDIO, exist_ok=True)
    commands = load_commands()
    manifest_path = os.path.join(DATA, "manifest.csv")
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            done = {r["clip_id"] for r in csv.DictReader(f)}
        print(f"Melanjutkan sesi: {len(done)} klip sudah ada.")

    dev = sd.query_devices(kind="input")
    source_rate = int(round(dev["default_samplerate"]))
    print("=" * 60)
    print("VoiceBridge - sesi rekam dataset")
    print(f"Mic: {dev['name']} | native rate: {source_rate} Hz -> {S.SAMPLE_RATE} Hz")
    total = sum(S.TAKES_PER_COMMAND[c["split"]] for c in commands)
    print(f"{len(commands)} perintah, {total} take total, {S.CLIP_SECONDS:.0f} detik per take (+-15 menit)")
    print("Posisi: mulut tepat 50 cm dari mic laptop (ukur pakai meteran), ruangan sepi.")
    print("=" * 60)

    session_path = os.path.join(DATA, "session.json")
    session = {}
    if os.path.exists(session_path):
        with open(session_path, encoding="utf-8") as f:
            session = json.load(f)
    if "calibration" not in session:
        session["calibration"] = calibrate(source_rate)
        session["mic"] = dev["name"]
        session["recording_sample_rate"] = source_rate
        session["started_at"] = dt.datetime.now().isoformat(timespec="seconds")
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

    new_file = not os.path.exists(manifest_path)
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(mf)
    if new_file:
        writer.writerow(["clip_id", "command_id", "split", "take", "text", "file",
                 "seconds", "source_sample_rate", "output_sample_rate", "peak",
                 "retakes", "sha1", "recorded_at"])

    rng = random.Random(S.RANDOM_SEED)
    max_takes = max(S.TAKES_PER_COMMAND.values())
    for take in range(1, max_takes + 1):
        order = [c for c in commands if take <= S.TAKES_PER_COMMAND[c["split"]]]
        rng.shuffle(order)  # different order each round, so fatigue is spread out
        for cmd in order:
            clip_id = f"{cmd['command_id']}_t{take}"
            if clip_id in done:
                continue
            retakes = 0
            while True:
                print(f"\n[Putaran {take}/{max_takes}] Ucapkan:  \"{cmd['text']}\"")
                input("  Tekan ENTER kalau siap...")
                countdown()
                recorded = record(S.CLIP_SECONDS, source_rate)
                audio = core.resample_audio(recorded, source_rate)
                peak = float(np.max(np.abs(audio)))
                warn = ""
                if peak < 0.05:
                    warn = "  !! Terlalu pelan (peak < 0.05)"
                elif peak > 0.98:
                    warn = "  !! Clipping (terlalu keras)"
                print(f"  Selesai. peak={peak:.2f}{warn}")
                ans = input("  ENTER = simpan | r = ulangi (salah ucap/batuk/noise) | p = putar ulang: ").strip().lower()
                while ans == "p":
                    sd.play(audio, S.SAMPLE_RATE)
                    sd.wait()
                    ans = input("  ENTER = simpan | r = ulangi: ").strip().lower()
                if ans == "r":
                    retakes += 1
                    continue
                break
            fname = f"{clip_id}.wav"
            sha = save_wav(os.path.join(AUDIO, fname), audio)
            writer.writerow([clip_id, cmd["command_id"], cmd["split"], take, cmd["text"],
                             fname, S.CLIP_SECONDS, source_rate, S.SAMPLE_RATE, round(peak, 4),
                             retakes, sha,
                             dt.datetime.now().isoformat(timespec="seconds")])
            mf.flush()
    mf.close()
    print("\nSelesai! Semua klip tersimpan di benchmark/dataset/audio")
    print("Langkah berikutnya: jalankan 3_benchmark.bat")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDihentikan. Klip yang sudah disimpan aman; jalankan lagi untuk melanjutkan.")
        sys.exit(0)
