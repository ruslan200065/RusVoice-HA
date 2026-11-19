"""RusVoice: supervise Wyoming backends and expose a small local speech gateway."""
import asyncio
import io
import logging
import os
import secrets
import sys
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from wyoming.asr import Transcribe, Transcript, TranscriptChunk
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeVoice

from .settings import load_options
from .models import ensure_piper, ensure_vosk, ensure_supertonic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("rusvoice")
OPTS = load_options()
DATA = Path(os.environ.get("RUSVOICE_DATA_DIR", "/data"))
BACKENDS = {"tts": 10200, "stt": 10300}


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=900)
    speaker: int | None = Field(default=None, ge=0, le=9)


class Runtime:
    def __init__(self):
        self.processes = {}
        self.tasks = {}
        self.last_exit = {}
        self.downloads = {"stt": {"phase": "pending"}, "tts": {"phase": "pending"}}

    def progress(self, name, **update):
        # Updates originate in the model-downloader worker thread.
        self.downloads[name] = {**self.downloads.get(name, {}), **update}


    def command(self, name):
        if name == "stt":
            cmd = [sys.executable, "-m", "wyoming_vosk", "--uri", "tcp://0.0.0.0:10300",
                   "--data-dir", str(DATA / "vosk"), "--download-dir", str(DATA / "vosk"),
                   "--language", "ru", "--preload-language", "ru",
                   "--model-for-language", "ru", OPTS["stt_model"]]
            if not OPTS["stt_streaming"]:
                cmd.append("--no-streaming")
            return cmd
        return [sys.executable, "-m", "app.tts_backend"]

    async def prepare_model(self, name):
        progress = lambda **state: self.progress(name, **state)
        if name == "stt":
            await asyncio.to_thread(ensure_vosk, DATA / "vosk", progress)
        elif OPTS["tts_engine"] == "supertonic":
            await asyncio.to_thread(ensure_supertonic, DATA / "supertonic", progress)
        else:
            await asyncio.to_thread(ensure_piper, DATA / "piper", OPTS["voice"], progress)

    async def supervise(self, name):
        while True:
            proc = None
            try:
                self.progress(name, phase="checking")
                await self.prepare_model(name)
                self.progress(name, phase="loading", error=None)
                LOG.info("Starting %s Wyoming server", name)
                proc = await asyncio.create_subprocess_exec(*self.command(name))
                self.progress(name, phase="installed")
                self.processes[name] = proc
                returncode = await proc.wait()
                self.last_exit[name] = returncode
                LOG.error("%s server exited with code %s; restarting in 10s", name, returncode)
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                if proc is not None and proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=8)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                raise
            except Exception as exc:
                self.progress(name, phase="error", error=str(exc)[:350])
                LOG.exception("Unable to start %s", name)
                await asyncio.sleep(30)
            finally:
                if self.processes.get(name) is proc:
                    self.processes.pop(name, None)

    async def start(self):
        (DATA / "piper").mkdir(parents=True, exist_ok=True)
        (DATA / "vosk").mkdir(parents=True, exist_ok=True)
        (DATA / "supertonic").mkdir(parents=True, exist_ok=True)
        for name in BACKENDS:
            self.tasks[name] = asyncio.create_task(self.supervise(name), name=f"wyoming-{name}")

    async def stop(self):
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)


runtime = Runtime()


@asynccontextmanager
async def lifespan(_app):
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="RusVoice HA", version="0.2.1", lifespan=lifespan, docs_url=None, redoc_url=None)


def authorize(request: Request):
    token = OPTS["api_token"]
    if token and not secrets.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="Bearer token required")


async def ready(port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 0.5)
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


@app.get("/")
async def index():
    return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")


@app.get("/api/status")
async def status(request: Request):
    authorize(request)
    return {"name": "RusVoice HA", "version": "0.2.1", "language": "ru",
            "tts_engine": OPTS["tts_engine"], "voice": OPTS["voice"] if OPTS["tts_engine"] == "piper" else f"supertonic-3 / speaker {OPTS['supertonic_speaker']}", "stt_model": OPTS["stt_model"],
            "downloads": runtime.downloads,
            "streaming": {"stt": OPTS["stt_streaming"], "tts": OPTS["tts_streaming"]},
            "wyoming": {name: {"port": port, "ready": await ready(port),
                                "last_exit": runtime.last_exit.get(name)}
                        for name, port in BACKENDS.items()}}


async def wyoming_tts(text: str, speaker: int | None = None):
    client = AsyncTcpClient("127.0.0.1", 10200, connect_timeout=3, read_timeout=120)
    await client.connect()
    try:
        await client.write_event(Synthesize(text=text,
                      voice=SynthesizeVoice(speaker=str(speaker)) if speaker is not None and OPTS["tts_engine"] == "supertonic" else None).event())
        while True:
            event = await client.read_event()
            if event is None:
                raise RuntimeError("TTS closed connection before audio-start")
            if AudioStart.is_type(event.type):
                return client, AudioStart.from_event(event)
            if event.type == "error":
                raise RuntimeError(str(event.data.get("text", "TTS error")))
    except BaseException:
        await client.disconnect()
        raise


async def tts_audio(client):
    try:
        while True:
            event = await client.read_event()
            if event is None or AudioStop.is_type(event.type):
                return
            if event.type == "error":
                raise RuntimeError(str(event.data.get("text", "TTS error")))
            if AudioChunk.is_type(event.type):
                yield AudioChunk.from_event(event).audio
    finally:
        await client.disconnect()


@app.post("/api/tts")
async def tts(body: TextInput, request: Request):
    authorize(request)
    try:
        client, fmt = await wyoming_tts(body.text, body.speaker)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(fmt.channels)
            wav.setsampwidth(fmt.width)
            wav.setframerate(fmt.rate)
            async for raw in tts_audio(client):
                wav.writeframes(raw)
        return Response(output.getvalue(), media_type="audio/wav")
    except (OSError, RuntimeError, TimeoutError) as exc:
        LOG.warning("TTS failure: %s", exc)
        raise HTTPException(503, "TTS unavailable") from exc


@app.post("/api/tts/stream")
async def tts_stream(body: TextInput, request: Request):
    """Progressive PCM16 bytes. Read format from response headers."""
    authorize(request)
    try:
        client, fmt = await wyoming_tts(body.text, body.speaker)
    except (OSError, RuntimeError, TimeoutError) as exc:
        raise HTTPException(503, "TTS unavailable") from exc
    return StreamingResponse(tts_audio(client), media_type="application/octet-stream",
                             headers={"X-Audio-Rate": str(fmt.rate),
                                      "X-Audio-Width": str(fmt.width),
                                      "X-Audio-Channels": str(fmt.channels),
                                      "X-Audio-Encoding": "pcm_s16le",
                                      "Cache-Control": "no-store"})


async def transcribe_pcm(pcm: bytes):
    async with AsyncTcpClient("127.0.0.1", 10300, connect_timeout=3, read_timeout=120) as client:
        await client.write_event(Transcribe(language="ru").event())
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        for start in range(0, len(pcm), 8192):
            await client.write_event(AudioChunk(rate=16000, width=2, channels=1,
                                                audio=pcm[start:start + 8192]).event())
        await client.write_event(AudioStop().event())
        while True:
            event = await client.read_event()
            if event is None:
                raise RuntimeError("Vosk closed connection without transcript")
            if Transcript.is_type(event.type):
                return Transcript.from_event(event).text
            if event.type == "error":
                raise RuntimeError(str(event.data.get("text", "Vosk error")))


@app.post("/api/stt")
async def stt(request: Request):
    """Accept a mono 16 kHz PCM16 WAV body (max 8 MiB)."""
    authorize(request)
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length") from exc
    if content_length > 8 * 1024 * 1024:
        raise HTTPException(413, "Maximum WAV size is 8 MiB")
    data = await request.body()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "Maximum WAV size is 8 MiB")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
                raise ValueError("Require mono, 16000 Hz, PCM16 WAV")
            pcm = wav.readframes(wav.getnframes())
    except (ValueError, EOFError, wave.Error) as exc:
        raise HTTPException(422, str(exc)) from exc
    if not pcm:
        raise HTTPException(422, "WAV is empty")
    try:
        text = await transcribe_pcm(pcm)
        return {"text": text, "language": "ru"}
    except (OSError, RuntimeError, TimeoutError) as exc:
        LOG.warning("STT failure: %s", exc)
        raise HTTPException(503, "STT unavailable") from exc


@app.websocket("/api/stt/ws")
async def ws_stt(ws: WebSocket):
    """Client sends raw 16kHz/mono/PCM16 binary frames; JSON {event:'end'} finishes."""
    token = OPTS["api_token"]
    if token and not secrets.compare_digest(ws.query_params.get("token", ""), token):
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        async with AsyncTcpClient("127.0.0.1", 10300, connect_timeout=3, read_timeout=120) as client:
            await client.write_event(Transcribe(language="ru").event())
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
            await ws.send_json({"type": "ready", "sample_rate": 16000})

            async def send_audio():
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    pcm = message.get("bytes")
                    if pcm is not None:
                        if len(pcm) > 65536 or len(pcm) % 2:
                            raise ValueError("Audio chunks must be even length, at most 64 KiB")
                        if pcm:
                            await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event())
                    elif message.get("text"):
                        import json
                        if json.loads(message["text"]).get("event") == "end":
                            await client.write_event(AudioStop().event())
                            return

            async def send_results():
                partial = ""
                while True:
                    event = await client.read_event()
                    if event is None:
                        raise RuntimeError("STT connection closed")
                    if TranscriptChunk.is_type(event.type):
                        partial += TranscriptChunk.from_event(event).text
                        await ws.send_json({"type": "partial", "text": partial})
                    elif Transcript.is_type(event.type):
                        await ws.send_json({"type": "final", "text": Transcript.from_event(event).text})
                        return
                    elif event.type == "error":
                        raise RuntimeError(str(event.data.get("text", "STT error")))

            receive_task = asyncio.create_task(send_audio())
            result_task = asyncio.create_task(send_results())
            try:
                done, _ = await asyncio.wait({receive_task, result_task}, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
                if receive_task in done and not result_task.done():
                    # Wait for final recognition after the client sends {event: end}.
                    await asyncio.wait_for(result_task, timeout=120)
            finally:
                for task in (receive_task, result_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(receive_task, result_task, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        LOG.warning("WebSocket STT failed: %s", exc)
        try:
            await ws.send_json({"type": "error", "message": "STT unavailable or invalid stream"})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099, log_level="info", proxy_headers=False)
