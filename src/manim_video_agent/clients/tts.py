from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

import httpx

from manim_video_agent.config import get_settings


class TTSModel(str, Enum):
    MAX = "inworld-tts-1.5-max"
    MINI = "inworld-tts-1.5-mini"


class AudioEncoding(str, Enum):
    MP3 = "MP3"
    LINEAR16 = "LINEAR16"
    OGG_OPUS = "OGG_OPUS"
    FLAC = "FLAC"


@dataclass(frozen=True)
class TTSConfig:
    api_key: str = ""
    base_url: str = ""
    voice_id: str = ""
    model_id: str = ""

    def __post_init__(self) -> None:
        settings = get_settings()
        if not self.api_key:
            object.__setattr__(self, "api_key", settings.inworld_api_key)
        if not self.base_url:
            object.__setattr__(self, "base_url", settings.tts_base_url)
        if not self.voice_id:
            object.__setattr__(self, "voice_id", settings.tts_voice_id)
        if not self.model_id:
            object.__setattr__(self, "model_id", settings.tts_model_id)


@dataclass
class TTSResult:
    audio: bytes
    characters_processed: int = 0
    model_id: str = ""


class InworldTTSError(Exception):
    pass


@dataclass
class InworldTTSClient:
    config: TTSConfig = field(default_factory=TTSConfig)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Basic {self.config.api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }

    def _payload(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str | None = None,
        encoding: AudioEncoding = AudioEncoding.MP3,
        sample_rate: int = 48000,
        speaking_rate: float = 1.0,
        temperature: float = 1.1,
    ) -> dict:
        return {
            "text": text,
            "voiceId": voice_id or self.config.voice_id,
            "modelId": model_id or self.config.model_id,
            "audioConfig": {
                "audioEncoding": encoding.value,
                "sampleRateHertz": sample_rate,
                "speakingRate": speaking_rate,
            },
            "temperature": temperature,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        model_id: str | None = None,
        encoding: AudioEncoding = AudioEncoding.MP3,
        sample_rate: int = 48000,
        speaking_rate: float = 1.0,
        temperature: float = 1.1,
    ) -> TTSResult:
        client = await self._get_client()
        payload = self._payload(
            text, voice_id, model_id, encoding, sample_rate, speaking_rate, temperature
        )

        response = await client.post(
            self.config.base_url,
            json=payload,
            headers=self._headers(),
        )

        if response.status_code != 200:
            raise InworldTTSError(
                f"TTS API failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        usage = result.get("usage", {})

        return TTSResult(
            audio=base64.b64decode(result["audioContent"]),
            characters_processed=usage.get("processedCharactersCount", 0),
            model_id=usage.get("modelId", ""),
        )

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        model_id: str | None = None,
        encoding: AudioEncoding = AudioEncoding.LINEAR16,
        sample_rate: int = 48000,
        speaking_rate: float = 1.0,
        temperature: float = 1.1,
    ) -> AsyncIterator[bytes]:
        client = await self._get_client()
        stream_url = self.config.base_url.rstrip("/") + ":stream"
        payload = self._payload(
            text, voice_id, model_id, encoding, sample_rate, speaking_rate, temperature
        )

        async with client.stream(
            "POST", stream_url, json=payload, headers=self._headers()
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise InworldTTSError(
                    f"TTS stream failed ({response.status_code}): {body.decode()}"
                )

            import json

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    result = chunk.get("result", {})
                    if "audioContent" in result:
                        yield base64.b64decode(result["audioContent"])
                except json.JSONDecodeError:
                    continue

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        *,
        voice_id: str | None = None,
        model_id: str | None = None,
    ) -> Path:
        result = await self.synthesize(text, voice_id=voice_id, model_id=model_id)
        output_path.write_bytes(result.audio)
        return output_path
