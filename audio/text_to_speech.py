"""Enhanced text-to-speech with pygame audio integration."""

import asyncio
from typing import Optional, Tuple
import io
import pygame
from openai import OpenAI
import tempfile
import os

from config import OPENAI_API_KEY, TTS_VOICE
from rl.strategy import Strategy


class TextToSpeech:
    """Handles text-to-speech conversion using OpenAI with pygame integration."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        pygame.mixer.init()
        self.is_playing = False
        self.current_audio_file = None

    def _strategy_to_instructions(self, strategy: Strategy) -> str:
        """Convert strategy to TTS instructions."""
        parts = []
        parts.append(strategy.tone)
        parts.append(strategy.emotion)

        if parts:
            instruction = f"Speak {' and '.join(parts)}. Don't read words in brackets, but match your energy from those words and mimic them, in terms of level, pace, tonality, personality etc"
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

            # print(f"TTS Instructions: {instructions}")
            # print(f"TTS Text: {text[:100]}...")

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

            audio_data = await loop.run_in_executor(None, _generate)

            # Save to temporary file for pygame
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(audio_data)
            temp_file.close()

            self.current_audio_file = temp_file.name

            # Play audio with pygame
            pygame.mixer.music.load(self.current_audio_file)
            pygame.mixer.music.play()

            self.is_playing = True
            tts_start = asyncio.get_event_loop().time()

            # Wait for playbook to complete
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

            tts_end = asyncio.get_event_loop().time()
            self.is_playing = False

            # Clean up temp file
            try:
                os.unlink(self.current_audio_file)
            except:
                pass
            self.current_audio_file = None

            return tts_start, tts_end

        except Exception as e:
            print(f"TTS error: {e}")
            return start_time, start_time

    def stop(self):
        """Stop current playback."""
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False

        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                os.unlink(self.current_audio_file)
            except:
                pass
            self.current_audio_file = None

    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        pygame.mixer.quit()