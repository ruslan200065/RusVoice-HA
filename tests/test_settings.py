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


def test_edge_opt_in_and_validation(tmp_path, monkeypatch):
    file = tmp_path / "options.json"
    file.write_text(json.dumps({"tts_engine": "edge", "edge_voice": "ru-RU-SvetlanaNeural"}))
    monkeypatch.delenv("RUSVOICE_OPTIONS", raising=False)
    options = load_options(file)
    assert options["tts_engine"] == "edge"
    assert options["edge_voice"] == "ru-RU-SvetlanaNeural"

@pytest.mark.parametrize("voice", ["invalid", "../../secret", "en-US-JennyNeural"])
def test_edge_rejects_unapproved_voice(tmp_path, monkeypatch, voice):
    file = tmp_path / "options.json"
    file.write_text(json.dumps({"tts_engine": "edge", "edge_voice": voice}))
    monkeypatch.delenv("RUSVOICE_OPTIONS", raising=False)
    with pytest.raises(ValueError):
        load_options(file)

def test_supertonic_valid_options(tmp_path, monkeypatch):
    file = tmp_path / 'options.json'
    file.write_text(json.dumps({'tts_engine': 'supertonic', 'supertonic_steps': 2, 'supertonic_speaker': 9}))
    monkeypatch.delenv('RUSVOICE_OPTIONS', raising=False)
    data = load_options(file)
    assert data['tts_engine'] == 'supertonic'
    assert data['supertonic_steps'] == 2


@pytest.mark.parametrize('key,value', [
    ('supertonic_speaker', 12), ('supertonic_speaker', True),
    ('supertonic_steps', 0), ('tts_engine', 'some-command'),
    ('tts_threads', 7), ('tts_speed', 0),
])
def test_reject_invalid_new_options(tmp_path, monkeypatch, key, value):
    file = tmp_path / 'options.json'
    file.write_text(json.dumps({key: value}))
    monkeypatch.delenv('RUSVOICE_OPTIONS', raising=False)
    with pytest.raises(ValueError):
        load_options(file)
