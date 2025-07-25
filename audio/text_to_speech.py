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
        """Convert strategy to comprehensive TTS instructions."""
        base_tone = strategy.tone.lower()
        base_emotion = strategy.emotion.lower()

        # Build detailed voice characteristics based on strategy
        if "professional" in base_tone:
            voice_affect = "Composed, authoritative, and competent; project quiet confidence and expertise."
            pacing = "Steady and measured; deliberate enough to convey thoughtfulness, efficient enough to demonstrate professionalism."
            pronunciation = "Clear and precise, emphasizing key points with subtle emphasis."
        elif "friendly" in base_tone:
            voice_affect = "Warm, approachable, and engaging; project genuine interest and openness."
            pacing = "Natural and conversational; relaxed enough to feel personal, energetic enough to maintain engagement."
            pronunciation = "Clear and natural, with slight emphasis on positive words and welcoming phrases."
        elif "empathetic" in base_tone:
            voice_affect = "Gentle, understanding, and supportive; project deep care and emotional intelligence."
            pacing = "Calm and unhurried; slow enough to communicate patience, steady enough to provide comfort."
            pronunciation = "Soft and clear, with gentle emphasis on reassuring words and understanding phrases."
        else:
            voice_affect = "Natural and balanced; project authenticity and genuine presence."
            pacing = "Conversational and steady; natural rhythm that feels human and engaging."
            pronunciation = "Clear and natural, emphasizing important points with subtle vocal variation."

        # Adapt emotion overlay
        if "excited" in base_emotion:
            emotion_overlay = "Express genuine enthusiasm and energy; let excitement shine through without overwhelming."
        elif "calm" in base_emotion:
            emotion_overlay = "Maintain serene composure and peaceful energy; speak with centered tranquility."
        elif "concerned" in base_emotion:
            emotion_overlay = "Convey thoughtful concern and attentiveness; show you're taking things seriously."
        elif "joyful" in base_emotion:
            emotion_overlay = "Radiate warmth and positivity; let natural happiness color your delivery."
        else:
            emotion_overlay = "Express authentic emotion that matches the content naturally."

        return f"Voice Affect: {voice_affect} Emotion: {emotion_overlay} Pacing: {pacing} Pronunciation: {pronunciation} Don't read words in brackets - instead embody the energy and style they describe in your vocal delivery."

    def _build_adaptive_voice_instructions(self, tts_params: Dict[str, float], user_emotion: float, user_engagement: float) -> str:
        """Build sophisticated voice instructions based on parameters and user state."""
        instructions = []

        # Speed adaptations with nuanced descriptions
        if tts_params["speed"] < 0.85:
            instructions.append("Pacing: Deliberately slow and thoughtful; each word carries weight and intention, creating space for reflection")
        elif tts_params["speed"] > 1.1:
            instructions.append("Pacing: Energetic and dynamic; speak with vitality and forward momentum while maintaining clarity")

        # Pitch adaptations with emotional context
        if tts_params["pitch"] < -0.1:
            instructions.append("Tone Quality: Rich, grounded, and resonant; project depth and gravitas with a warm lower register")
        elif tts_params["pitch"] > 0.1:
            instructions.append("Tone Quality: Bright and uplifting; use a lighter register that conveys optimism and openness")

        # Energy adaptations with specific affect
        if tts_params["energy"] < 0.85:
            instructions.append("Voice Affect: Gentle and soothing; speak with soft intensity that creates calm and safety")
        elif tts_params["energy"] > 1.1:
            instructions.append("Voice Affect: Vibrant and engaging; radiate enthusiasm while maintaining natural warmth")

        # Warmth with emotional intelligence
        if tts_params["warmth"] > 1.1:
            instructions.append("Emotional Range: Infuse every word with genuine compassion and understanding; let care flow through your voice naturally")

        # User state adaptations
        if user_emotion < 0.4:
            instructions.append("Emotional Sensitivity: The person seems withdrawn or struggling - speak with extra gentleness, patience, and reassuring presence. Use softer intonation and allow natural pauses for comfort")
        elif user_emotion > 0.6:
            instructions.append("Emotional Resonance: The person seems positive and open - you can match their energy with warmth and enthusiasm, using brighter intonation")

        if user_engagement < 0.4:
            instructions.append("Engagement Style: They seem distracted - add subtle vocal variety and intrigue. Use strategic pauses, whisper-to-normal transitions, and gentle vocal texture changes to recapture attention")
        elif user_engagement > 0.7:
            instructions.append("Engagement Style: They're highly focused - maintain steady, consistent delivery without overwhelming. Use subtle emphasis and natural rhythm")

        return ". ".join(instructions) + "." if instructions else ""

    def _strategy_to_adaptive_instructions(
        self, strategy: Strategy, user_emotion: float, user_engagement: float
    ) -> str:
        """Convert strategy and user state to comprehensive adaptive TTS instructions."""
        tts_params = strategy.get_emotion_adapted_tts_params(
            user_emotion, user_engagement
        )

        # Get base strategy instructions
        base_instructions = self._strategy_to_instructions(strategy)

        # Add adaptive voice instructions
        adaptive_instructions = self._build_adaptive_voice_instructions(
            tts_params, user_emotion, user_engagement
        )

        # Combine with context-aware delivery style
        context_instructions = self._build_context_instructions(user_emotion, user_engagement)

        # Final instruction combining all elements
        full_instructions = f"{base_instructions} {adaptive_instructions} {context_instructions}"

        return full_instructions

    def _build_context_instructions(self, user_emotion: float, user_engagement: float) -> str:
        """Build context-aware delivery instructions."""
        context_parts = []

        # Emotional context
        if user_emotion < 0.3:
            context_parts.append("Delivery Context: This person may be having a difficult time - prioritize emotional safety, speak as if offering a warm embrace through your voice")
        elif user_emotion > 0.7:
            context_parts.append("Delivery Context: This person seems upbeat - you can be more expressive and mirror their positive energy naturally")

        # Engagement context
        if user_engagement < 0.3:
            context_parts.append("Attention Context: Low engagement detected - use vocal storytelling techniques like dramatic pauses, volume variation, and compelling rhythm to draw them in")
        elif user_engagement > 0.8:
            context_parts.append("Attention Context: High engagement - maintain their interest with confident, clear delivery and well-placed emphasis")

        return " ".join(context_parts)

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
                    instructions=instructions
                )
                return response.content

            print("📝 TTS Instructions:", instructions[:200] + "..." if len(instructions) > 200 else instructions)
            print("🗣️ TTS Text:", text[:100] + "..." if len(text) > 100 else text)

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

            # Wait for playback to complete
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