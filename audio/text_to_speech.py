"""Text-to-speech using ElevenLabs API."""

import asyncio
from typing import Optional, Tuple
import io
import pygame
from elevenlabs import ElevenLabs

from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, TTS_MODEL


class TextToSpeech:
    """Handles text-to-speech conversion using ElevenLabs."""

    def __init__(self):
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        pygame.mixer.init()
        self.is_playing = False

    async def speak(self, text: str, voice_id: Optional[str] = None) -> Tuple[float, float]:
        """
        Convert text to speech and play it.
        Returns (start_time, end_time) for engagement tracking.
        """
        voice_id = voice_id or ELEVENLABS_VOICE_ID

        try:
            # Generate audio
            start_time = asyncio.get_event_loop().time()

            # Run API call in executor
            loop = asyncio.get_event_loop()

            def _generate():
                response = self.client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=TTS_MODEL,
                    output_format="mp3_22050_32"
                )
                # ElevenLabs returns a generator, we need to collect the data
                audio_data = b''.join(response)
                return audio_data

            audio_data = await loop.run_in_executor(None, _generate)

            # Play audio
            audio_stream = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()

            self.is_playing = True
            tts_start = asyncio.get_event_loop().time()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

            tts_end = asyncio.get_event_loop().time()
            self.is_playing = False

            return tts_start, tts_end

        except Exception as e:
            print(f"TTS error: {e}")
            return start_time, start_time

    def stop(self):
        """Stop current playback."""
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False

    def cleanup(self):
        """Clean up audio resources."""
        pygame.mixer.quit()