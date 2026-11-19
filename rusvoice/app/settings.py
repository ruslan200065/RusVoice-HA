"""Safe, typed options loaded from Home Assistant Supervisor."""
import json
import os
from pathlib import Path

DEFAULTS = {
    "voice": "ru_RU-dmitri-medium",
    "stt_model": "vosk-model-small-ru-0.22",
    "stt_streaming": True,
    "tts_streaming": True,
    "length_scale": 1.0,
    "noise_scale": 0.667,
    "noise_w": 0.333,
    "api_token": "",
}
VOICES = {"ru_RU-dmitri-medium", "ru_RU-denis-medium", "ru_RU-irina-medium", "ru_RU-ruslan-medium"}
STT_MODELS = {"vosk-model-small-ru-0.22"}


def load_options(path: str | Path = "/data/options.json") -> dict:
    """Read Supervisor options; reject malformed values before passing to child CLI."""
    result = DEFAULTS.copy()
    options_path = Path(os.environ.get("RUSVOICE_OPTIONS", str(path)))
    if options_path.exists():
        raw = json.loads(options_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("options.json must be an object")
        result.update({k: v for k, v in raw.items() if k in DEFAULTS})
    if result["voice"] not in VOICES or result["stt_model"] not in STT_MODELS:
        raise ValueError("Unsupported voice or STT model")
    for key in ("stt_streaming", "tts_streaming"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key, low, high in (("length_scale", .5, 2), ("noise_scale", 0, 2), ("noise_w", 0, 2)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    if not isinstance(result["api_token"], str):
        raise ValueError("api_token must be a string")
    return result
