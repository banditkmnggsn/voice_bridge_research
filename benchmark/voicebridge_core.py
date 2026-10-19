"""
VoiceBridge core pipeline (v2).

Used by both the benchmark (replaying recorded clips) and the live demo, so the
numbers we report come from the same code that runs on stage.

Pipeline: audio -> endpointing (silence detection) -> Whisper transcription
          -> wake word check -> intent (rule or DeepSeek) -> browser action
"""

import json
import os
import re
import time
import urllib.parse
from math import gcd

import numpy as np
from scipy.signal import resample_poly

import settings as S


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text):
    """Lowercase, drop punctuation, collapse spaces."""
    text = (text or "").lower().replace("-", " ")
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def wer(reference, hypothesis):
    """Word error rate = (substitutions + deletions + insertions) / reference words."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(ref), len(hyp)]) / len(ref)


_STOP = {"a", "an", "the", "some", "for", "to", "me", "please", "of", "in",
         "on", "what", "is", "can", "you", "i", "want"}


def content_tokens(text):
    return [t for t in normalize_text(text).split() if t not in _STOP]


def query_matches(expected, predicted):
    """A query counts as correct if it contains every content word of the
    expected query and adds at most two extra content words.
    'weather in jakarta' == 'jakarta weather'; 'some coldplay' == 'coldplay'."""
    exp = content_tokens(expected)
    pred = content_tokens(predicted)
    if not exp:
        return True
    if not set(exp).issubset(pred):
        return False
    return len([t for t in pred if t not in exp]) <= 2


# --------------------------------------------------------------------------
# Audio: endpointing and normalization
# --------------------------------------------------------------------------

def block_volume(block):
    """Same volume measure as the original scripts: L2 norm of one block."""
    return float(np.linalg.norm(block))


def resample_audio(audio, source_rate, target_rate=S.SAMPLE_RATE):
    """Downsample microphone audio with scipy's anti-aliasing polyphase filter."""
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def endpoint_segment(audio, silence_threshold, silence_duration,
                     blocksize=S.BLOCKSIZE, sr=S.SAMPLE_RATE, start_sample=0):
    """Replay the original record_until_silence() logic on a recorded clip,
    using audio time instead of wall-clock time.

    Returns (segment, info). The segment ends at the block where silence has
    lasted longer than silence_duration, exactly like the live loop.
    """
    silence_start = None
    end = len(audio)
    endpointed = False
    last_loud_end = None
    pos = start_sample
    while pos < len(audio):
        block = audio[pos:pos + blocksize]
        t_block_end = (pos + len(block)) / sr
        if block_volume(block) < silence_threshold:
            if silence_start is None:
                silence_start = pos / sr
            elif t_block_end - silence_start > silence_duration:
                end = pos + len(block)
                endpointed = True
                break
        else:
            silence_start = None
            last_loud_end = t_block_end
        pos += blocksize
    seg = audio[start_sample:end]
    cut_sec = end / sr
    info = {
        "segment_start_sec": round(start_sample / sr, 4),
        "cut_sec": round(cut_sec, 4),
        "endpointed": endpointed,
        "last_loud_sec": None if last_loud_end is None else round(last_loud_end, 4),
        # time the user waits after their last loud block before recording stops
        "endpoint_wait_s": None if (last_loud_end is None or not endpointed)
        else round(cut_sec - last_loud_end, 4),
    }
    return seg, info


def speech_bounds(audio, noise_floor_rms, sr=S.SAMPLE_RATE, frame=320):
    """Config-independent reference for where speech starts and ends
    (20 ms RMS frames above 4x the measured noise floor)."""
    n = len(audio) // frame
    if n == 0:
        return None, None
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    thr = max(4.0 * noise_floor_rms, 0.01)
    loud = np.where(rms > thr)[0]
    if len(loud) == 0:
        return None, None
    return loud[0] * frame / sr, (loud[-1] + 1) * frame / sr


def peak_normalize(audio):
    """Config 3 normalization, as in the original code: divide by peak."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio / peak if peak > 0 else audio


def zscore_normalize(audio):
    """Normalize audio using the paper's z-score formula."""
    if not len(audio):
        return audio
    mean = float(np.mean(audio))
    std = float(np.std(audio))
    return (audio - mean) / std if std > 0 else audio - mean


def normalize_audio(audio, mode):
    """Apply the configured preprocessing mode, including legacy booleans."""
    if mode is True:
        mode = "peak"
    if mode in (False, None, "none"):
        return audio
    if mode == "peak":
        return peak_normalize(audio)
    if mode == "zscore":
        return zscore_normalize(audio)
    raise ValueError(f"unknown normalization mode: {mode}")


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

_MODELS = {}


def load_model(cfg):
    """Load and cache the configured Whisper backend and precision."""
    backend = cfg.get("backend", "openai_whisper")
    name = cfg["model"]
    precision = cfg.get("precision", "fp32")
    key = (backend, name, precision)
    if key in _MODELS:
        return _MODELS[key]
    if backend == "openai_whisper":
        import whisper
        model = whisper.load_model(name, device="cpu")
    elif backend == "faster_whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(name, device="cpu", compute_type=precision,
                             cpu_threads=cfg.get("cpu_threads", 0))
    else:
        raise ValueError(f"unknown transcription backend: {backend}")
    _MODELS[key] = model
    return model


def transcribe(model, audio, cfg, return_info=False):
    audio = audio.flatten().astype("float32")
    audio = normalize_audio(audio, cfg["normalize"]).astype("float32")
    timing = {"t_encoder_s": None, "t_decoder_s": None, "n_tokens": None}
    if cfg.get("backend", "openai_whisper") == "faster_whisper":
        segments, _ = model.transcribe(
            audio, language=cfg.get("language", "en"), temperature=0,
            condition_on_previous_text=False,
            initial_prompt=cfg.get("vocab_prompt"),
        )
        segments = list(segments)
        text = "".join(segment.text for segment in segments).strip()
        if return_info:
            timing["n_tokens"] = sum(len(getattr(segment, "tokens", ()) or ())
                                     for segment in segments)
            return text, timing
        return text
    kwargs = dict(language="en", temperature=0, condition_on_previous_text=False,
                  fp16=False)
    if cfg["vocab_prompt"]:
        kwargs["initial_prompt"] = cfg["vocab_prompt"]
    hooks = []
    encoder_elapsed = [0.0]
    decoder_elapsed = [0.0]

    if return_info and hasattr(model, "encoder") and hasattr(model, "decoder"):
        # Forward hooks cannot observe the work already completed, so each hook
        # uses a paired pre-hook/post-hook timer around the module forward pass.
        def add_timer(module, total):
            starts = []

            def before(_module, _inputs):
                starts.append(time.perf_counter())

            def after(_module, _inputs, _output):
                total[0] += time.perf_counter() - starts.pop()

            hooks.extend([module.register_forward_pre_hook(before),
                          module.register_forward_hook(after)])

        add_timer(model.encoder, encoder_elapsed)
        add_timer(model.decoder, decoder_elapsed)
    try:
        result = model.transcribe(audio, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()
    text = result["text"].strip()
    if return_info:
        timing["t_encoder_s"] = round(encoder_elapsed[0], 6)
        timing["t_decoder_s"] = round(decoder_elapsed[0], 6)
        timing["n_tokens"] = sum(len(segment.get("tokens", ()))
                                  for segment in result.get("segments", ()))
        return text, timing
    return text


def split_wake_word(transcript, wake=S.WAKE_WORD):
    """Returns (wake_detected, command_text)."""
    norm = normalize_text(transcript)
    wake_tokens = normalize_text(wake).split()
    transcript_tokens = norm.split()
    if not wake_tokens:
        return False, ""
    for index in range(len(transcript_tokens) - len(wake_tokens) + 1):
        if transcript_tokens[index:index + len(wake_tokens)] == wake_tokens:
            return True, " ".join(transcript_tokens[index + len(wake_tokens):])
    return False, ""


# --------------------------------------------------------------------------
# Intent: rule-based
# --------------------------------------------------------------------------

HOME_KW = ["open youtube", "youtube home", "youtube main", "go to youtube"]
SEARCH_KW = ["search for", "search", "find", "look up"]
PLAY_KW = ["play", "put on", "listen to", "music", "song", "video"]


def _after(cmd, kw):
    return cmd.split(kw, 1)[-1].strip()


def intent_rule(command):
    """Keyword rules from the original code. One fix: in the original, 'find'
    and 'search for' sat in the PLAY list, so 'find github' opened YouTube.
    v2 routes them to Google search, which is what the paper describes."""
    cmd = normalize_text(command)
    if any(kw in cmd for kw in HOME_KW):
        return "open_youtube_home", ""
    for kw in SEARCH_KW:
        if re.search(r"\b" + re.escape(kw) + r"\b", cmd):
            return "search_google", _after(cmd, kw) or cmd
    for kw in PLAY_KW:
        if re.search(r"\b" + re.escape(kw) + r"\b", cmd):
            return "play_youtube", _after(cmd, kw) or cmd
    return "search_google", cmd


# --------------------------------------------------------------------------
# Intent: DeepSeek LLM
# --------------------------------------------------------------------------

LLM_SYSTEM = """You turn a short voice command into a browser action.
Reply with JSON only: {"intent": "...", "query": "..."}.
Allowed intents:
- "open_youtube_home": open the YouTube home page (query is "").
- "play_youtube": play a song, artist or video on YouTube (query = what to play).
- "search_google": search the web on Google (query = what to search).
The text comes from speech recognition and may contain recognition errors;
infer the most likely meaning. Keep the query short and drop filler words."""

LLM_EXAMPLES = [
    ("bring up youtube for me", {"intent": "open_youtube_home", "query": ""}),
    ("i feel like hearing some adele", {"intent": "play_youtube", "query": "adele"}),
    ("how tall is mount everest", {"intent": "search_google", "query": "height of mount everest"}),
]


def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(here, "..", ".env")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(("OPENROUTER_API_KEY=", "DEEPSEEK_API_KEY=")):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and not val.startswith("PASTE"):
                            return val
    return None


def load_env_value(name, default=None):
    value = os.environ.get(name)
    if value:
        return value.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(here, "..", ".env")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(name + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


def intent_llm(command, api_key, session=None):
    """Returns (intent, query, info). info has tokens / error for logging."""
    import requests
    messages = [{"role": "system", "content": LLM_SYSTEM}]
    for text, out in LLM_EXAMPLES:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps(out)})
    messages.append({"role": "user", "content": command})
    is_openrouter = api_key.startswith("sk-or-")
    endpoint = load_env_value(
        "VOICE_BRIDGE_LLM_URL",
        "https://openrouter.ai/api/v1/chat/completions" if is_openrouter else S.DEEPSEEK_URL,
    )
    model = load_env_value(
        "VOICE_BRIDGE_LLM_MODEL",
        "deepseek/deepseek-chat" if is_openrouter else S.DEEPSEEK_MODEL,
    )
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 60,
        "response_format": {"type": "json_object"},
    }
    http = session or requests
    info = {"error": "", "prompt_tokens": None, "completion_tokens": None,
            "request_bytes": 0}
    try:
        request_bytes = len(json.dumps(body, separators=(",", ":")).encode("utf-8"))
        info["request_bytes"] = request_bytes
        headers = {"Authorization": f"Bearer {api_key}"}
        if is_openrouter:
            headers["HTTP-Referer"] = "https://github.com/banditkmnggsn/voice_bridge_research"
            headers["X-Title"] = "VoiceBridge Research"
        r = http.post(endpoint, json=body, timeout=S.DEEPSEEK_TIMEOUT_SEC,
                      headers=headers)
        if r.status_code != 200:
            info["error"] = f"HTTP {r.status_code}: {r.text[:120]}"
            return "error", "", info
        data = r.json()
        usage = data.get("usage", {})
        info["prompt_tokens"] = usage.get("prompt_tokens")
        info["completion_tokens"] = usage.get("completion_tokens")
        out = json.loads(data["choices"][0]["message"]["content"])
        intent = out.get("intent", "")
        if intent not in ("open_youtube_home", "play_youtube", "search_google"):
            info["error"] = f"invalid intent: {intent}"
            return "error", "", info
        return intent, normalize_text(out.get("query", "")), info
    except Exception as e:  # network, JSON, timeout
        info["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return "error", "", info


# --------------------------------------------------------------------------
# Action
# --------------------------------------------------------------------------

def action_url(intent, query):
    if intent == "open_youtube_home":
        return "https://www.youtube.com"
    if intent == "play_youtube":
        return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    if intent == "search_google":
        return "https://www.google.com/search?q=" + urllib.parse.quote(query)
    return ""


def now():
    return time.perf_counter()
