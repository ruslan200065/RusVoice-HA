"""Validated Supervisor options for RusVoice HA."""
import json
import os
from pathlib import Path

DEFAULTS = {
    "tts_engine": "piper",
    "voice": "ru_RU-irina-medium",
    "supertonic_speaker": 0,
    "supertonic_steps": 4,
    "tts_threads": 2,
    "tts_speed": 1.0,
    "stt_model": "vosk-model-small-ru-0.22",
    "stt_streaming": True,
    "tts_streaming": True,
    "length_scale": 0.95,
    "noise_scale": 0.667,
    "noise_w": 0.8,
    "api_token": "",
}
VOICES = {"ru_RU-dmitri-medium", "ru_RU-denis-medium", "ru_RU-irina-medium", "ru_RU-ruslan-medium"}
STT_MODELS = {"vosk-model-small-ru-0.22"}


def load_options(path: str | Path = "/data/options.json") -> dict:
    result = DEFAULTS.copy()
    options_path = Path(os.environ.get("RUSVOICE_OPTIONS", str(path)))
    if options_path.exists():
        raw = json.loads(options_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("options.json must be an object")
        result.update({k: v for k, v in raw.items() if k in DEFAULTS})
    if result["tts_engine"] not in ("piper", "supertonic"):
        raise ValueError("tts_engine must be piper or supertonic")
    if result["voice"] not in VOICES or result["stt_model"] not in STT_MODELS:
        raise ValueError("Unsupported voice or STT model")
    for key in ("stt_streaming", "tts_streaming"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key, lo, hi in (("supertonic_speaker", 0, 9), ("supertonic_steps", 1, 16), ("tts_threads", 1, 4)):
        value = result[key]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"{key} must be integer {lo}..{hi}")
    for key, lo, hi in (("length_scale", .5, 2), ("noise_scale", 0, 2),
                        ("noise_w", 0, 2), ("tts_speed", .6, 1.8)):
        value = result[key]
        if type(value) not in (float, int) or not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
    if not isinstance(result["api_token"], str):
        raise ValueError("api_token must be a string")
    return result
