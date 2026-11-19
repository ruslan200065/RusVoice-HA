import io
import zipfile

import pytest

from rusvoice.app import models


def sample_archive():
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w') as file:
        for suffix in ('am/final.mdl', 'conf/model.conf', 'graph/HCLG.fst'):
            file.writestr(f'{models.VOSK_NAME}/{suffix}', 'fake-for-unit-test')
    return mem.getvalue()


def test_vosk_install_and_cache(tmp_path, monkeypatch):
    calls = []
    progress = []
    def fake_fetch(url, dest, callback, **kwargs):
        calls.append(url)
        dest.write_bytes(sample_archive())
    monkeypatch.setattr(models, 'fetch', fake_fetch)
    models.ensure_vosk(tmp_path, lambda **state: progress.append(state))
    assert models.vosk_installed(tmp_path / models.VOSK_NAME)
    assert progress[-1]['phase'] == 'installed'
    models.ensure_vosk(tmp_path, lambda **state: progress.append(state))
    assert len(calls) == 1  # No re-download after restart


def test_vosk_mirror_fallback(tmp_path, monkeypatch):
    attempts = []
    def fake_fetch(url, dest, callback, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            raise OSError('primary unavailable')
        dest.write_bytes(sample_archive())
    monkeypatch.setattr(models, 'fetch', fake_fetch)
    models.ensure_vosk(tmp_path)
    assert len(attempts) == 2
    assert models.vosk_installed(tmp_path / models.VOSK_NAME)


def test_reject_zip_traversal():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as archive:
        archive.writestr(f'{models.VOSK_NAME}/../../evil.txt', 'owned')
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as archive:
        with pytest.raises(ValueError):
            models._safe_members(archive.infolist(), models.VOSK_NAME)


def test_supertonic_requires_all_weights(tmp_path):
    folder = tmp_path / models.SUPERTONIC_NAME
    folder.mkdir()
    (folder / 'tts.json').write_text('{}')
    assert not models.supertonic_installed(folder)


def test_piper_rejects_malformed_voice(tmp_path):
    with pytest.raises(ValueError):
        models.ensure_piper(tmp_path, '../../secret')
