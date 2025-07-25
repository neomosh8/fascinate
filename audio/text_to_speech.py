"""Enhanced text-to-speech with pygame audio integration."""

import asyncio
from typing import Optional, Tuple, Dict
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
        parts = [strategy.tone, strategy.emotion]
        if parts:
            instruction = (
                f"Speak {' and '.join(parts)}. Don't read words in brackets, but match your energy from those words and mimic them, in terms of level, pace, tonality, personality etc"
            )
        else:
            instruction = "Speak in a natural, conversational tone."
        return instruction

    def _build_voice_instructions(self, tts_params: Dict[str, float]) -> str:
        instructions = []
        if tts_params["speed"] < 0.85:
            instructions.append("Speak slowly and deliberately")
        elif tts_params["speed"] > 1.1:
            instructions.append("Speak with good pace and energy")

        if tts_params["pitch"] < -0.1:
            instructions.append("use a deeper, grounded tone")
        elif tts_params["pitch"] > 0.1:
            instructions.append("use a lighter, more uplifting tone")

        if tts_params["energy"] < 0.85:
            instructions.append("speak calmly and gently")
        elif tts_params["energy"] > 1.1:
            instructions.append("speak with warmth and vitality")

        if tts_params["warmth"] > 1.1:
            instructions.append("infuse extra warmth and compassion into your voice")

        return ". ".join(instructions) + ". " if instructions else ""

    def _strategy_to_adaptive_instructions(
        self, strategy: Strategy, user_emotion: float, user_engagement: float
    ) -> str:
        """Convert strategy and user state to adaptive TTS instructions."""
        tts_params = strategy.get_emotion_adapted_tts_params(
            user_emotion, user_engagement
        )
        base_instructions = f"Speak {strategy.tone} and {strategy.emotion}. "

        if user_emotion < 0.4:
            emotion_instruction = "The person seems in a withdrawn or difficult emotional state. Respond with extra warmth, patience, and gentleness. "
        elif user_emotion > 0.6:
            emotion_instruction = "The person seems emotionally open and positive. You can match , talk energetic and happy "
        else:
            emotion_instruction = "The person seems emotionally neutral. use confident and professionl voice"

        if user_engagement < 0.4:
            engagement_instruction = "They seem disengaged - add more vocal variety  talk in whispers and then normal sequence, glitches and psst "
        elif user_engagement > 0.7:
            engagement_instruction = "They're highly engaged - maintain steady presence without overstimulating. "
        else:
            engagement_instruction = "-"

        voice_instructions = self._build_voice_instructions(tts_params)
        full = (
            base_instructions
            + emotion_instruction
            + engagement_instruction
            + voice_instructions
            + "Don't read words in brackets, but embody the energy and style described." )
        return full

    async def speak(
        self,
        text: str,
        strategy: Strategy,
        user_emotion: float = 0.5,
        user_engagement: float = 0.5,
        voice: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Convert text to speech and play it.
        Returns (start_time, end_time) for engagement tracking.
        """
        voice = voice or TTS_VOICE

        try:
            instructions = self._strategy_to_adaptive_instructions(
                strategy, user_emotion, user_engagement
            )

            # print(f"TTS Instructions: {instructions}")
            # print(f"TTS Text: {text[:100]}...")

            # Generate audio
            start_time = asyncio.get_event_loop().time()

            # Run API call in executor
            loop = asyncio.get_event_loop()

            voice_sig = getattr(strategy, "get_voice_signature", lambda: "default")()
            print(f"🎤 TTS Voice: {voice_sig}")
            print(f"🎭 Adapting to: emotion={user_emotion:.2f}, engagement={user_engagement:.2f}")

            def _generate():
                response = self.client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    input=text,
                    instructions=instructions,
                    response_format="mp3"
                )
                return response.content
            print(instructions,text)
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