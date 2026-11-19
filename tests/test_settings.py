import json
import pytest
from rusvoice.app.settings import DEFAULTS, load_options


def test_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("RUSVOICE_OPTIONS", raising=False)
    assert load_options(tmp_path / "missing.json") == DEFAULTS


def test_override(tmp_path, monkeypatch):
    file = tmp_path / "options.json"
    file.write_text(json.dumps({"voice": "ru_RU-irina-medium", "length_scale": 1.2}))
    monkeypatch.delenv("RUSVOICE_OPTIONS", raising=False)
    assert load_options(file)["voice"] == "ru_RU-irina-medium"


@pytest.mark.parametrize("value", ["; echo invalid", "ru_RU-not-a-voice"])
def test_reject_injected_voice(tmp_path, monkeypatch, value):
    file = tmp_path / "options.json"
    file.write_text(json.dumps({"voice": value}))
    monkeypatch.delenv("RUSVOICE_OPTIONS", raising=False)
    with pytest.raises(ValueError):
        load_options(file)


def test_reject_bad_streaming(tmp_path, monkeypatch):
    file = tmp_path / "options.json"
    file.write_text(json.dumps({"stt_streaming": "false"}))
    monkeypatch.delenv("RUSVOICE_OPTIONS", raising=False)
    with pytest.raises(ValueError):
        load_options(file)
