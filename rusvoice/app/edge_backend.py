"""Opt-in cloud voice backend: Microsoft Edge Read Aloud, not an offline engine.

The user's text is sent to Microsoft when selected in the add-on options.
PCM is decoded incrementally with ffmpeg instead of buffering a complete WAV.
"""
from __future__ import annotations

import asyncio
import logging

from .settings import EDGE_VOICES, load_options

LOG = logging.getLogger("rusvoice.edge")
RATE = 24000
READ_SIZE = 4800  # ~100 milliseconds of mono PCM16


class Edge:
    def __init__(self):
        self.default_voice = load_options()["edge_voice"]

    async def synthesize_async(self, text: str, voice_name: str | None = None):
        # Do not use the network until this backend has been explicitly selected.
        import edge_tts

        voice = voice_name if voice_name in EDGE_VOICES else self.default_voice
        if voice not in EDGE_VOICES:
            raise ValueError("Unsupported Edge voice")

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-f", "mp3", "-i", "pipe:0", "-f", "s16le",
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(RATE), "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        assert proc.stderr is not None

        async def feed_mp3():
            try:
                async for data in edge_tts.Communicate(text, voice=voice).stream():
                    if data.get("type") == "audio" and data.get("data"):
                        proc.stdin.write(data["data"])
                        await proc.stdin.drain()
            finally:
                proc.stdin.close()

        feeder = asyncio.create_task(feed_mp3(), name="edge-mp3-producer")
        carry = b""
        try:
            while True:
                raw = await asyncio.wait_for(proc.stdout.read(READ_SIZE), timeout=60)
                if not raw:
                    break
                carry += raw
                complete = len(carry) & ~1
                if complete:
                    yield RATE, 2, 1, carry[:complete]
                    carry = carry[complete:]
            await feeder
            code = await proc.wait()
            if code:
                stderr = (await proc.stderr.read()).decode("utf-8", "replace")
                raise RuntimeError(f"ffmpeg decoder exited {code}: {stderr[:250]}")
            if carry:
                raise RuntimeError("Odd-length PCM output from decoder")
        finally:
            if not feeder.done():
                feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
