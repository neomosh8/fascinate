"""Text-to-speech using OpenAI API."""

import asyncio
from typing import Optional, Tuple
import io
import pygame
from openai import OpenAI

from config import OPENAI_API_KEY, TTS_VOICE
from rl.strategy import Strategy


class TextToSpeech:
    """Handles text-to-speech conversion using OpenAI."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        pygame.mixer.init()
        self.is_playing = False

    def _strategy_to_instructions(self, strategy: Strategy) -> str:
        """Convert strategy to TTS instructions."""

        # Build instruction
        parts = []

        parts.append(strategy.tone)
        parts.append(strategy.emotion)


        if parts:
            instruction = f"Speak {' and '.join(parts)}. "
        else:
            instruction = "Speak in a natural, conversational tone."

        return instruction

    async def speak(self, text: str, strategy: Strategy, voice: Optional[str] = None) -> Tuple[float, float]:
        """
        Convert text to speech and play it.
        Returns (start_time, end_time) for engagement tracking.
        """
        voice = voice or TTS_VOICE

        try:
            # Generate TTS instructions from strategy
            instructions = self._strategy_to_instructions(strategy)

            print(f"TTS Instructions: {instructions}")
            print(f"TTS Text: {text[:100]}...")

            # Generate audio
            start_time = asyncio.get_event_loop().time()

            # Run API call in executor
            loop = asyncio.get_event_loop()

            def _generate():
                response = self.client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    input=text,
                    instructions=instructions,
                    response_format="mp3"
                )
                return response.content
            print("&&&&", instructions)
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