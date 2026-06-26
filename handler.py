import base64
import re
import tempfile
import uuid
from pathlib import Path

import runpod
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

SIMPLE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_model = None


def safe_filename(name: str) -> str:
    name = (name or "").strip()

    if not name:
        name = "qwen3_tts_output.wav"

    if not name.lower().endswith(".wav"):
        name += ".wav"

    if name.startswith("/") or name.startswith("\\"):
        raise ValueError("Absolute paths are not allowed")

    if "/" in name or "\\" in name or ".." in name:
        raise ValueError("Path traversal is not allowed")

    if not SIMPLE_FILENAME_RE.fullmatch(name):
        raise ValueError("Filename contains invalid characters")

    return name


def get_model():
    global _model

    if _model is None:
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

    return _model


def health():
    return {
        "status": "ok",
        "worker": "hermes-qwen3-tts-serverless",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def qwen3_tts(job_input):
    text = (job_input.get("text") or "").strip()
    voice_prompt = (
        job_input.get("voice_prompt")
        or "Speak clearly and naturally, with a warm presenter tone."
    )
    filename = safe_filename(job_input.get("filename") or "qwen3_tts_output.wav")
    speaker = job_input.get("speaker") or "Aiden"
    language = job_input.get("language") or "English"

    if not text:
        raise ValueError("Text is required")

    if len(text) > 1000:
        raise ValueError("Text too long for first Qwen3-TTS test. Keep it under 1000 characters.")

    model = get_model()

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=voice_prompt,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / filename
        sf.write(str(output_path), wavs[0], sr)

        audio_bytes = output_path.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "status": "ok",
        "job_id": str(uuid.uuid4()),
        "engine": "qwen3-tts",
        "model": MODEL_ID,
        "filename": filename,
        "mime_type": "audio/wav",
        "size_bytes": len(audio_bytes),
        "audio_base64": audio_b64,
        "speaker": speaker,
        "language": language,
        "voice_prompt": voice_prompt,
    }


def handler(job):
    job_input = job.get("input", {}) or {}
    action = job_input.get("action")

    try:
        if action == "health":
            return health()

        if action == "tts":
            return qwen3_tts(job_input)

        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "supported_actions": ["health", "tts"],
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "action": action,
        }


runpod.serverless.start({"handler": handler})
