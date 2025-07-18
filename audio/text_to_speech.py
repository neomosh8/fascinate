"""Text-to-speech using OpenAI API."""

import asyncio
import re
from typing import Optional, Tuple
import io
import pygame
from openai import OpenAI

from config import OPENAI_API_KEY


class TextToSpeech:
    """Handles text-to-speech conversion using OpenAI."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        pygame.mixer.init()
        self.is_playing = False

        # Default voice - you can make this configurable
        self.default_voice = "coral"

    def _extract_emotions_and_clean_text(self, text: str) -> Tuple[str, str]:
        """
        Extract emotion/voice cues from text and create instructions.
        Returns (cleaned_text, instructions)
        """
        # Pattern to match emotions in brackets like [excited], [whispers], etc.
        emotion_pattern = r'\[([^\]]+)\]'
        emotions = re.findall(emotion_pattern, text)

        # Remove emotion brackets from the text
        cleaned_text = re.sub(emotion_pattern, '', text)

        # Clean up extra spaces and punctuation
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Convert emotions to natural instructions
        instructions = self._emotions_to_instructions(emotions)

        return cleaned_text, instructions

    def _emotions_to_instructions(self, emotions: list) -> str:
        """Convert emotion tags to natural language instructions."""
        if not emotions:
            return "Speak in a natural, conversational tone."

        # Map common emotion tags to instructions
        emotion_mappings = {
            'excited': 'with excitement and energy',
            'whispers': 'in a soft, whispering voice',
            'laughs': 'with a cheerful, laughing tone',
            'curious': 'with curiosity and interest',
            'gentle': 'in a gentle, soothing manner',
            'calm': 'in a calm, relaxed tone',
            'enthusiastic': 'with enthusiasm and passion',
            'sarcastic': 'with a hint of sarcasm',
            'frustrated': 'with slight frustration',
            'concerned': 'with genuine concern',
            'joy': 'with joy and happiness',
            'interest': 'with genuine interest',
            'surprise': 'with surprise and wonder',
            'trust': 'with warmth and trust',
            'anticipation': 'with anticipation and eagerness',
            'pride': 'with confidence and pride',
            'gratitude': 'with gratitude and appreciation'
        }

        # Handle accent instructions
        accent_pattern = r'strong (\w+) accent'

        instruction_parts = []
        for emotion in emotions:
            emotion_lower = emotion.lower().strip()

            # Check for accent patterns
            accent_match = re.search(accent_pattern, emotion_lower)
            if accent_match:
                accent = accent_match.group(1)
                instruction_parts.append(f'with a subtle {accent} accent')
            elif emotion_lower in emotion_mappings:
                instruction_parts.append(emotion_mappings[emotion_lower])
            else:
                # For unmapped emotions, create a basic instruction
                instruction_parts.append(f'in a {emotion_lower} manner')

        if instruction_parts:
            base_instruction = "Speak " + ", and ".join(instruction_parts[:3])  # Limit to 3 for clarity
            return base_instruction + ". Keep the delivery natural and conversational."
        else:
            return "Speak in a natural, conversational tone."

    async def speak(self, text: str, voice: Optional[str] = None) -> Tuple[float, float]:
        """
        Convert text to speech and play it.
        Returns (start_time, end_time) for engagement tracking.
        """
        voice = voice or self.default_voice

        try:
            # Extract emotions and clean text
            cleaned_text, instructions = self._extract_emotions_and_clean_text(text)

            if not cleaned_text.strip():
                # If no text remains after cleaning, use original
                cleaned_text = text
                instructions = "Speak in a natural, conversational tone."

            print(f"TTS Instructions: {instructions}")
            print(f"TTS Text: {cleaned_text[:100]}...")

            # Generate audio
            start_time = asyncio.get_event_loop().time()

            # Run API call in executor
            loop = asyncio.get_event_loop()

            def _generate():
                response = self.client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    input=cleaned_text,
                    instructions=instructions,
                    response_format="mp3"
                )
                return response.content

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