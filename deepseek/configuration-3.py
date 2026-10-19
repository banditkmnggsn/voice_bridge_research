import sounddevice as sd
import numpy as np
import queue
import time
import whisper
import webbrowser
import urllib.parse
import requests
import re
from metrics_monitor import MetricsMonitor

WAKE_WORD = "hey voice bridge"
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.15
SILENCE_DURATION = 0.4

import os
audio_queue = queue.Queue()
model = whisper.load_model("tiny")

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "metrics_log.csv")
metrics = MetricsMonitor(log_file=log_file_path)

print("Assistant running")
print(f"Wake word: '{WAKE_WORD}'")
print("Listening...\n")

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def record_until_silence():
    metrics.start_recording()
    frames = []
    silence_start = None

    while True:
        data = audio_queue.get()
        frames.append(data)

        volume = np.linalg.norm(data)
        if volume < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_DURATION:
                break
        else:
            silence_start = None

    metrics.end_recording()
    return np.concatenate(frames, axis=0)

def transcribe(audio_np):
    metrics.start_transcription()
    audio_np = audio_np.flatten().astype("float32")
    
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val
    
    result = model.transcribe(
        audio_np,
        language="en",
        temperature=0,
        condition_on_previous_text=False,
        prompt="play youtube despacito hey voice bridge music video search google"
    )
    metrics.end_transcription()
    return result["text"].lower().strip()

def play_youtube_first_video(query):
    print("Finding video for:", query)

    search_url = (
        "https://www.youtube.com/results?search_query="
        + urllib.parse.quote(query)
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(search_url, headers=headers)
    matches = re.findall(r"watch\\?v=([a-zA-Z0-9_-]{11})", r.text)

    if matches:
        video_url = "https://www.youtube.com/watch?v=" + matches[0]
        print("Playing:", video_url)
        webbrowser.open(video_url)
    else:
        print("No direct video found, opening YouTube search")
        webbrowser.open(search_url)

def categorize_intent(command):
    cmd = command.lower()

    youtube_home_keywords = ["open youtube", "youtube home", "youtube main", "go to youtube"]
    if any(kw in cmd for kw in youtube_home_keywords):
        return "open_youtube_home", ""

    play_keywords = ["play", "put on", "listen to", "search for", "find", "music", "song", "video"]
    if any(kw in cmd for kw in play_keywords):
        query = cmd
        for kw in play_keywords:
            if kw in query:
                query = query.split(kw, 1)[-1].strip()
                break
        return "play_youtube", query or cmd

    return "search_google", cmd

def execute(intent, query):
    metrics.start_execution()
    print(f"Executing -> {intent} | {query}")

    if intent == "open_youtube_home":
        webbrowser.open("https://www.youtube.com")
        metrics.end_execution()
        return

    if intent == "play_youtube":
        play_youtube_first_video(query)
        metrics.end_execution()
        return

    if intent == "search_google":
        webbrowser.open(
            "https://www.google.com/search?q="
            + urllib.parse.quote(query)
        )
        metrics.end_execution()
        return

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    callback=audio_callback
):
    while True:
        audio = record_until_silence()
        text = transcribe(audio)

        if not text:
            continue

        print("Heard:", text)

        if WAKE_WORD not in text:
            continue

        metrics.mark_wake_word_detected()

        command = text.replace(WAKE_WORD, "").strip()
        if not command:
            continue

        try:
            metrics.start_intent_classification()
            intent, query = categorize_intent(command)
            metrics.end_intent_classification()
            
            execute(intent, query)
            metrics.log_session(
                wake_word=WAKE_WORD,
                transcribed_text=text,
                command=command,
                intent=intent,
                query=query,
                audio_data=audio
            )
        except Exception as e:
            print("Error:", e)
            metrics.reset_session()
