"""Low-latency Wyoming TTS. Piper streams synthesis output directly, without WAV tempfile.

Supertonic 3 INT8 is experimental. Edge Neural is an explicit cloud-only option.
"""
from __future__ import annotations

import asyncio
import logging
import re
from functools import partial
from pathlib import Path

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped

from .models import SUPERTONIC_NAME
from .settings import EDGE_VOICES, load_options

LOG = logging.getLogger("rusvoice.tts")
OPTS = load_options()
DATA = Path(__import__("os").environ.get("RUSVOICE_DATA_DIR", "/data"))
LOCK = asyncio.Lock()


def split_sentences(value: str, flush: bool = False) -> tuple[list[str], str]:
    """Extract complete sentences from a stream without losing unflushed text."""
    pieces = []
    while True:
        # Require whitespace or end of input after punctuation to avoid breaking decimals.
        m = re.search(r"[.!?…]+(?=\s|$)", value)
        if not m:
            break
        pieces.append(value[:m.end()].strip())
        value = value[m.end():].lstrip()
    if flush and value.strip():
        pieces.append(value.strip())
        value = ""
    return [p for p in pieces if p], value


class Piper:
    def __init__(self):
        from piper import PiperVoice, SynthesisConfig
        self.model = PiperVoice.load(str(DATA / "piper" / (OPTS["voice"] + ".onnx")))
        self.config = SynthesisConfig(length_scale=float(OPTS["length_scale"]),
                                      noise_scale=float(OPTS["noise_scale"]),
                                      noise_w_scale=float(OPTS["noise_w"]))

    def synthesize(self, text: str, speaker: str | None = None):
        from piper import SynthesisConfig
        cfg = self.config
        # One model/speaker is loaded once, keeping latency consistent.
        if speaker is not None:
            cfg = SynthesisConfig(length_scale=float(OPTS["length_scale"]),
                                  noise_scale=float(OPTS["noise_scale"]),
                                  noise_w_scale=float(OPTS["noise_w"]))
            cfg.speaker_id = int(speaker) if speaker.isdigit() else self.model.config.speaker_id_map.get(speaker)
        for chunk in self.model.synthesize(text, syn_config=cfg):
            yield (chunk.sample_rate, chunk.sample_width, chunk.sample_channels, chunk.audio_int16_bytes)


class Supertonic:
    def __init__(self):
        import sherpa_onnx
        root = DATA / "supertonic" / SUPERTONIC_NAME
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(root / "duration_predictor.int8.onnx"),
                text_encoder=str(root / "text_encoder.int8.onnx"),
                vector_estimator=str(root / "vector_estimator.int8.onnx"),
                vocoder=str(root / "vocoder.int8.onnx"),
                tts_json=str(root / "tts.json"),
                unicode_indexer=str(root / "unicode_indexer.bin"),
                voice_style=str(root / "voice.bin"),
            ), num_threads=int(OPTS["tts_threads"]), provider="cpu"))
        if not cfg.validate():
            raise ValueError("Invalid Supertonic model files")
        self.model = sherpa_onnx.OfflineTts(cfg)
        self.sherpa_onnx = sherpa_onnx

    def synthesize(self, text: str, speaker: str | None = None):
        import numpy as np
        cfg = self.sherpa_onnx.GenerationConfig()
        cfg.sid = int(speaker) if speaker is not None and speaker.isdigit() else OPTS["supertonic_speaker"]
        if not 0 <= cfg.sid <= 9:
            raise ValueError("Supertonic speaker must be 0..9")
        cfg.num_steps = OPTS["supertonic_steps"]
        cfg.speed = OPTS["tts_speed"]
        cfg.extra["lang"] = "ru"
        audio = self.model.generate(text, cfg)
        if len(audio.samples) == 0:
            raise RuntimeError("Supertonic returned empty audio")
        pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), 16384):
            yield (audio.sample_rate, 2, 1, pcm[i:i+16384])


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, synth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = info
        self.synth = synth
        self._buffer = ""
        self._stream = False
        self._format = None
        self._speaker = None
        self._voice_name = None

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(self.info.event())
            elif SynthesizeStart.is_type(event.type):
                request = SynthesizeStart.from_event(event)
                self._stream = True
                self._buffer = ""
                self._format = None
                self._speaker = request.voice.speaker if request.voice else None
                self._voice_name = request.voice.name if request.voice else None
            elif SynthesizeChunk.is_type(event.type):
                if not self._stream:
                    raise ValueError("Received synthesize-chunk without synthesize-start")
                self._buffer += SynthesizeChunk.from_event(event).text
                sentences, self._buffer = split_sentences(self._buffer)
                for sentence in sentences:
                    await self.emit(sentence)
            elif SynthesizeStop.is_type(event.type):
                if not self._stream:
                    raise ValueError("Received synthesize-stop without synthesize-start")
                sentences, self._buffer = split_sentences(self._buffer, flush=True)
                for sentence in sentences:
                    await self.emit(sentence)
                await self.write_event(SynthesizeStopped().event())
                self._stream = False
                self._format = None
            elif Synthesize.is_type(event.type):
                if self._stream:
                    return True  # Compatibility event; streaming text arrives in chunks.
                request = Synthesize.from_event(event)
                self._format = None
                self._speaker = request.voice.speaker if request.voice else None
                self._voice_name = request.voice.name if request.voice else None
                sentences, _ = split_sentences(request.text, flush=True)
                for sentence in sentences:
                    await self.emit(sentence)
                if self._format is None:
                    # Empty request still gets a terminal event.
                    await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())
                await self.write_event(AudioStop().event())
            return True
        except Exception as exc:
            LOG.exception("TTS synthesis error")
            await self.write_event(Event(type="error", data={"text": str(exc), "code": exc.__class__.__name__}))
            return True


    async def emit(self, sentence: str):
        # Serialize a shared synthesizer per process; TTS streaming is preserved.
        async with LOCK:
            if OPTS["tts_engine"] == "edge":
                async for item in self.synth.synthesize_async(sentence, self._voice_name):
                    await self.write_pcm(item)
                return

            generator = self.synth.synthesize(sentence, self._speaker)
            sentinel = object()

            def next_chunk():
                return next(generator, sentinel)

            while True:
                item = await asyncio.to_thread(next_chunk)
                if item is sentinel:
                    break
                await self.write_pcm(item)

    async def write_pcm(self, item):
        rate, width, channels, pcm = item
        if self._format is None:
            self._format = (rate, width, channels)
            await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        elif self._format != (rate, width, channels):
            raise RuntimeError("TTS format changed within a stream")
        if pcm:
            await self.write_event(AudioChunk(rate=rate, width=width, channels=channels, audio=pcm).event())

async def main():
    logging.basicConfig(level=logging.INFO)
    if OPTS["tts_engine"] == "edge":
        from .edge_backend import Edge
        synth = Edge()
    elif OPTS["tts_engine"] == "supertonic":
        synth = Supertonic()
    else:
        synth = Piper()
    if OPTS["tts_engine"] == "edge":
        voices = [
            TtsVoice(name=name, description=f"Microsoft Edge Neural / {name} (ONLINE)",
                     attribution=Attribution(name="Microsoft Edge Read Aloud", url="https://www.microsoft.com/edge"),
                     installed=True, version=None, languages=["ru", "ru_RU"], speakers=None)
            for name in sorted(EDGE_VOICES)
        ]
    else:
        voices = (
        [TtsVoice(name="supertonic-3", description="Supertonic 3 Russian (10 speakers)",
                  attribution=Attribution(name="Supertone", url="https://huggingface.co/Supertone/supertonic-3"),
                  installed=True, version=None, languages=["ru", "ru_RU"], speakers=[TtsVoiceSpeaker(name=str(i)) for i in range(10)])]
        if OPTS["tts_engine"] == "supertonic"
        else [TtsVoice(name=OPTS["voice"], description=OPTS["voice"],
                       attribution=Attribution(name="Piper",url="https://github.com/OHF-Voice/piper1-gpl"),
                       installed=True, version=None, languages=["ru", "ru_RU"], speakers=None)]
        )
    info = Info(tts=[TtsProgram(name=OPTS["tts_engine"], description="RusVoice TTS (Edge mode uses Microsoft online service)",
                                attribution=Attribution(name="RusVoice-HA", url="https://github.com/ruslan200065/RusVoice-HA"),
                                installed=True, version="0.3.0", voices=voices,
                                supports_synthesize_streaming=True)])
    await AsyncServer.from_uri("tcp://0.0.0.0:10200").run(partial(Handler, info, synth))


if __name__ == "__main__":
    asyncio.run(main())
