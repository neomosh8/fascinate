"""Enhanced text-to-speech with pygame audio integration and multiple TTS engines."""

import asyncio
from typing import Optional, Tuple, Dict
import io
import pygame
from openai import OpenAI
import tempfile
import os
import base64
import aiofiles
import threading
import queue
import time
import logging

from config import OPENAI_API_KEY, TTS_VOICE, HUME_API_KEY, TTS_ENGINE
from rl.strategy import Strategy

# Hume imports
try:
    from hume import AsyncHumeClient
    from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName, FormatMp3, FormatWav, FormatPcm
    HUME_AVAILABLE = True
except ImportError:
    print("⚠️ Hume SDK not available. Install with: pip install hume")
    HUME_AVAILABLE = False

# PyAudio imports for true streaming
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ PyAudio not available. True streaming disabled.")
    PYAUDIO_AVAILABLE = False


class StreamingAudioPlayer:
    """Real-time audio streaming player using PyAudio."""

    def __init__(self, sample_rate=48000, channels=1, chunk_size=1024):
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio required for streaming playback")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.playing = False
        self.player_thread = None

    def start_playback(self):
        """Start the audio playback stream."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        self.playing = True
        self.player_thread = threading.Thread(target=self._playback_worker)
        self.player_thread.daemon = True
        self.player_thread.start()

    def _playback_worker(self):
        """Worker thread that continuously plays queued audio chunks."""
        while self.playing:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                if audio_chunk is None:  # Sentinel to stop
                    break
                self.stream.write(audio_chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Playback error: {e}")
                break

    def queue_audio(self, audio_data: bytes):
        """Queue audio data for immediate playback."""
        if self.playing:
            self.audio_queue.put(audio_data)

    def stop_playback(self):
        """Stop playback and cleanup."""
        self.playing = False
        self.audio_queue.put(None)  # Sentinel

        if self.player_thread:
            self.player_thread.join(timeout=1.0)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()


class TextToSpeech:
    """Handles text-to-speech conversion using OpenAI or Hume with pygame integration."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        pygame.mixer.init()
        self.is_playing = False
        self.current_audio_file = None
        self.interrupted = False
        self.logger = logging.getLogger(__name__)

        # TTS Engine selection
        self.engine = TTS_ENGINE.lower()
        print(f"🎵 TTS Engine: {self.engine}")

        # Initialize Hume client if available
        self.hume_client = None
        if HUME_AVAILABLE and HUME_API_KEY:
            try:
                self.hume_client = AsyncHumeClient(api_key=HUME_API_KEY)
                print("✅ Hume async client initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Hume client: {e}")

        # Validate configuration
        if self.engine == "hume" and (not HUME_AVAILABLE or not self.hume_client):
            print("⚠️ Hume not available, falling back to OpenAI")
            self.engine = "openai"
        elif self.engine == "openai" and not OPENAI_API_KEY:
            print("⚠️ OpenAI API key not found")

    def _strategy_to_hume_description(self, strategy: Strategy, user_emotion: float, user_engagement: float,
                                      has_voice: bool = False) -> str:
        """Convert strategy to Hume-compatible description following official best practices."""

        if has_voice:
            # ACTING INSTRUCTIONS: Concise directions (≤100 chars) when voice is specified
            instructions = []

            # Map strategy components to precise, concise terms
            tone_instructions = {
                "professional": "confident, professional tone",
                "friendly": "warm, inviting",
                "empathetic": "gentle, understanding",
                "playful": "lighthearted, playful",
                "confident": "assertive, self-assured",
                "kind": "caring, nurturing",
                "sarcastic": "sarcastic, dry",
                "informational": "clear, instructional",
                "calm": "calm, measured",
            }

            emotion_instructions = {
                "happy": "cheerful, upbeat",
                "serious": "serious, focused",
                "thoughtful": "contemplative, reflective",
                "angry": "frustrated, intense",
                "flirting": "playful, charming",
                "sad": "melancholy, subdued",
                "scared": "nervous, cautious",
                "worried": "concerned, anxious",
                "whisper": "whispering, hushed",
                "laughter": "amused, laughing",
            }

            # Build concise acting instruction
            base_tone = strategy.tone.lower()
            if base_tone in tone_instructions:
                instructions.append(tone_instructions[base_tone])

            base_emotion = strategy.emotion.lower()
            if base_emotion in emotion_instructions:
                instructions.append(emotion_instructions[base_emotion])

            # Add user state adaptations (very concise)
            if user_emotion < 0.4:
                instructions.append("extra gentle")
            elif user_emotion > 0.6:
                if "upbeat" not in " ".join(instructions):
                    instructions.append("encouraging")

            if user_engagement < 0.4:
                instructions.append("engaging, varied")

            # Combine and ensure ≤100 characters
            acting_instruction = ", ".join(instructions)
            if len(acting_instruction) > 95:
                # Prioritize most important elements
                primary = instructions[0] if instructions else "natural"
                acting_instruction = primary[:95]

            return acting_instruction

        else:
            # VOICE GENERATION: Detailed description to create new voice (≤1000 chars)
            voice_parts = []

            # Detailed voice generation descriptions
            voice_generation_map = {
                "professional": "A confident business professional with authoritative presence and clear articulation",
                "friendly": "A warm, approachable person with genuine enthusiasm and welcoming energy",
                "empathetic": "A compassionate counselor with gentle, understanding vocal quality",
                "playful": "An energetic, fun-loving individual with spontaneous, lighthearted delivery",
                "confident": "A self-assured speaker with strong vocal presence and commanding delivery",
                "kind": "A nurturing, caring individual with warm, supportive vocal characteristics",
                "sarcastic": "A witty speaker with subtle irony and intelligent, dry humor",
                "informational": "A knowledgeable educator with clear, objective presentation style",
                "calm": "A serene, peaceful speaker with composed, tranquil delivery",
            }

            emotion_generation_map = {
                "happy": "expressing natural joy and positive energy",
                "serious": "with focused, thoughtful gravity",
                "thoughtful": "with contemplative, reflective depth",
                "angry": "capable of controlled intensity when needed",
                "flirting": "with subtle charm and engaging warmth",
                "sad": "able to convey gentle melancholy authentically",
                "scared": "expressing cautious uncertainty naturally",
                "worried": "showing genuine concern and attentiveness",
                "whisper": "capable of intimate, soft delivery",
                "laughter": "with infectious, joyful expressiveness",
            }

            # Build voice generation prompt
            base_tone = strategy.tone.lower()
            if base_tone in voice_generation_map:
                voice_parts.append(voice_generation_map[base_tone])

            base_emotion = strategy.emotion.lower()
            if base_emotion in emotion_generation_map:
                voice_parts.append(emotion_generation_map[base_emotion])

            # Add context based on user state
            if user_emotion < 0.4:
                voice_parts.append("especially skilled at providing comfort and reassurance")
            elif user_emotion > 0.6:
                voice_parts.append("able to match and enhance positive energy naturally")

            # Combine for voice generation (can be longer)
            voice_description = ", ".join(voice_parts)

            # Ensure within 1000 character limit
            if len(voice_description) > 950:
                voice_description = voice_description[:947] + "..."

            return voice_description

    def _get_hume_speed_from_strategy(self, strategy: Strategy, user_emotion: float, user_engagement: float) -> float:
        """Calculate Hume speed parameter (0.25-3.0) from strategy."""

        # Get base TTS parameters
        tts_params = strategy.get_emotion_adapted_tts_params(user_emotion, user_engagement)
        base_speed = tts_params["speed"]

        # Map from OpenAI-style speed (0.7-1.3) to Hume range (0.25-3.0)
        # Hume uses non-linear scale, so we'll be conservative
        if base_speed <= 0.8:
            hume_speed = 0.65  # Slower for calm, empathetic tones
        elif base_speed <= 0.9:
            hume_speed = 0.8  # Slightly slower
        elif base_speed <= 1.1:
            hume_speed = 1.0  # Normal pace
        elif base_speed <= 1.2:
            hume_speed = 1.25  # Slightly faster
        else:
            hume_speed = 1.5  # Faster for energetic emotions

        # Additional adjustments based on strategy
        if strategy.emotion.lower() in ["whisper", "calm", "sad"]:
            hume_speed *= 0.8  # Slower for these emotions
        elif strategy.emotion.lower() in ["happy", "excited", "angry"]:
            hume_speed *= 1.2  # Faster for energetic emotions

        # Ensure within Hume's bounds
        return max(0.25, min(3.0, hume_speed))

    def _get_trailing_silence_from_strategy(self, strategy: Strategy) -> float:
        """Get appropriate trailing silence based on strategy."""

        # Add pauses for certain emotional contexts
        if strategy.emotion.lower() in ["thoughtful", "serious", "sad"]:
            return 1.0  # Longer pause for contemplative emotions
        elif strategy.emotion.lower() in ["whisper", "calm"]:
            return 0.8  # Medium pause for gentle emotions
        elif strategy.tone.lower() in ["empathetic", "kind"]:
            return 0.6  # Slight pause for caring tones
        else:
            return 0.3  # Short default pause

    def _strategy_to_openai_instructions(self, strategy: Strategy, user_emotion: float, user_engagement: float) -> str:
        """Convert strategy to OpenAI TTS instructions (existing method)."""
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

    async def _write_audio_to_temp_file(self, base64_audio: str, suffix: str = '.wav') -> str:
        """Write base64 audio data to temporary file."""
        audio_data = base64.b64decode(base64_audio)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        async with aiofiles.open(temp_file.name, "wb") as f:
            await f.write(audio_data)
        temp_file.close()
        return temp_file.name

    async def _speak_with_hume(self, text: str, strategy: Strategy, user_emotion: float, user_engagement: float,
                               voice: Optional[str] = None) -> Tuple[float, float]:
        """Generate speech using Hume TTS API following official acting instructions best practices."""

        if not self.hume_client:
            print("❌ Hume client not available")
            return await self._speak_with_openai(text, strategy, user_emotion, user_engagement, voice)

        # Determine if we're using a voice
        has_voice = voice is not None

        # Generate description following Hume best practices
        description = self._strategy_to_hume_description(strategy, user_emotion, user_engagement, has_voice)

        # Calculate Hume-specific parameters
        hume_speed = self._get_hume_speed_from_strategy(strategy, user_emotion, user_engagement)
        trailing_silence = self._get_trailing_silence_from_strategy(strategy)

        print(f"🎤 Hume {'Acting' if has_voice else 'Voice Gen'}: {description}")
        print(f"⚡ Speed: {hume_speed:.2f}, Silence: {trailing_silence:.1f}s")
        print(f"🗣️ Text: {text[:100]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # Create utterance with all parameters following Hume best practices
            utterance_params = {
                "text": text,
                "description": description,
                "speed": hume_speed,
                "trailing_silence": trailing_silence,
            }

            # Add voice if specified (for acting instructions)
            if voice:
                utterance_params["voice"] = PostedUtteranceVoiceWithName(
                    name=voice,
                    provider="HUME_AI"
                )

            utterance = PostedUtterance(**utterance_params)

            # Use the proper async synthesize_json method
            response = await self.hume_client.tts.synthesize_json(
                utterances=[utterance],
                format=FormatWav(),  # Use WAV format for better compatibility
                num_generations=1
            )

            if not response.generations or not response.generations[0].audio:
                raise Exception("No audio data received from Hume")

            # Get the base64 audio data
            base64_audio = response.generations[0].audio

            # Write to temporary file
            audio_file_path = await self._write_audio_to_temp_file(base64_audio, '.wav')
            self.current_audio_file = audio_file_path

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
            print(f"Hume TTS error: {e}")
            print("🔄 Falling back to OpenAI TTS")
            return await self._speak_with_openai(text, strategy, user_emotion, user_engagement, voice)

    async def _speak_with_hume_streaming(self, text: str, strategy: Strategy, user_emotion: float,
                                         user_engagement: float, voice: Optional[str] = None) -> Tuple[float, float]:
        """Generate speech using Hume streaming TTS with instant mode for ultra-low latency."""

        if not self.hume_client:
            print("❌ Hume client not available")
            return await self._speak_with_openai(text, strategy, user_emotion, user_engagement, voice)

        # Determine if we're using a voice
        has_voice = voice is not None

        # For instant mode, we MUST have a voice - fallback to default if none provided
        if not voice:
            voice = "Ava Song"  # Default voice for instant mode
            has_voice = True
            print("🚀 Using default voice 'Ava Song' for instant mode")

        # Generate description following Hume best practices
        description = self._strategy_to_hume_description(strategy, user_emotion, user_engagement, has_voice)

        # Calculate Hume-specific parameters
        hume_speed = self._get_hume_speed_from_strategy(strategy, user_emotion, user_engagement)
        trailing_silence = self._get_trailing_silence_from_strategy(strategy)

        print(f"🎤 Hume Instant Streaming Acting: {description}")
        print(f"⚡ Speed: {hume_speed:.2f}, Silence: {trailing_silence:.1f}s")
        print(f"🗣️ Text: {text[:100]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # Create utterance with all parameters following Hume best practices
            utterance_params = {
                "text": text,
                "description": description,
                "speed": hume_speed,
                "trailing_silence": trailing_silence,
                "voice": PostedUtteranceVoiceWithName(
                    name=voice,
                    provider="HUME_AI"
                )
            }

            utterance = PostedUtterance(**utterance_params)

            # Collect all audio chunks from streaming with instant mode
            audio_chunks = []
            chunk_count = 0
            first_chunk_time = None

            async for snippet in self.hume_client.tts.synthesize_json_streaming(
                    utterances=[utterance],
                    format=FormatWav(),  # Use WAV for better compatibility
                    num_generations=1,  # Required for instant mode
                    instant_mode=True  # 🚀 ENABLE INSTANT MODE
            ):
                if snippet.audio:
                    if chunk_count == 0:
                        first_chunk_time = asyncio.get_event_loop().time()
                        latency = first_chunk_time - start_time
                        print(f"⚡ First chunk received in {latency:.3f}s (instant mode)")

                    audio_chunks.append(base64.b64decode(snippet.audio))
                    chunk_count += 1
                    if chunk_count == 1:
                        print("🌊 Started receiving audio chunks...")

            if not audio_chunks:
                raise Exception("No audio chunks received from Hume instant streaming")

            print(f"📦 Collected {len(audio_chunks)} audio chunks")

            # Combine all chunks
            combined_audio = b''.join(audio_chunks)

            # Write to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(combined_audio)
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
            print(f"Hume instant streaming TTS error: {e}")
            print("🔄 Falling back to regular Hume TTS")
            return await self._speak_with_hume(text, strategy, user_emotion, user_engagement, voice)

    async def _speak_with_hume_true_streaming(self, text: str, strategy: Strategy, user_emotion: float,
                                              user_engagement: float, voice: Optional[str] = None) -> Tuple[
        float, float]:
        """TRUE streaming with instant mode - plays audio chunks as they arrive using PyAudio."""

        self.interrupted = False

        if not self.hume_client:
            return await self._speak_with_openai(text, strategy, user_emotion, user_engagement, voice)

        if not PYAUDIO_AVAILABLE:
            print("⚠️ PyAudio not available, falling back to file-based streaming")
            return await self._speak_with_hume_streaming(text, strategy, user_emotion, user_engagement, voice)

        # For instant mode, we MUST have a voice - fallback to default if none provided
        if not voice:
            voice = "Ava Song"  # Default voice for instant mode
            print("🚀 Using default voice 'Ava Song' for instant mode")

        has_voice = True  # Always true since we ensure voice is set

        # Generate description following Hume best practices (acting instructions only)
        description = self._strategy_to_hume_description(strategy, user_emotion, user_engagement, has_voice)

        # Calculate Hume-specific parameters
        hume_speed = self._get_hume_speed_from_strategy(strategy, user_emotion, user_engagement)
        trailing_silence = self._get_trailing_silence_from_strategy(strategy)

        print(f"🎤 Hume TRUE Instant Streaming Acting: {description}")
        print(f"⚡ Speed: {hume_speed:.2f}, Silence: {trailing_silence:.1f}s")
        print(f"🗣️ Text: {text[:100]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # Create utterance with all parameters following Hume best practices
            utterance_params = {
                "text": text,
                "description": description,
                "speed": hume_speed,
                "trailing_silence": trailing_silence,
                "voice": PostedUtteranceVoiceWithName(
                    name=voice,
                    provider="HUME_AI"
                )
            }

            utterance = PostedUtterance(**utterance_params)

            # Initialize streaming player
            player = StreamingAudioPlayer()
            first_chunk_played = False
            first_chunk_time = None
            tts_start = None
            chunk_count = 0

            try:
                # Start streaming and play chunks as they arrive with INSTANT MODE
                async for snippet in self.hume_client.tts.synthesize_json_streaming(
                        utterances=[utterance],
                        format=FormatPcm(type="pcm"),  # Use PCM for direct audio playback
                        num_generations=1,  # Required for instant mode
                        instant_mode=True  # 🚀 ENABLE INSTANT MODE
                ):
                    if self.interrupted:
                        self.logger.info("TTS interrupted by user")
                        break
                    if snippet.audio:
                        audio_chunk = base64.b64decode(snippet.audio)
                        chunk_count += 1

                        # Start playback on first chunk
                        if not first_chunk_played:
                            first_chunk_time = asyncio.get_event_loop().time()
                            latency = first_chunk_time - start_time
                            print(f"⚡ First chunk received in {latency:.3f}s (instant mode)")

                            player.start_playback()
                            tts_start = first_chunk_time
                            first_chunk_played = True
                            print(f"🔊 Started playing first audio chunk! ({len(audio_chunk)} bytes)")

                        # Queue this chunk for immediate playback
                        player.queue_audio(audio_chunk)
                        if chunk_count <= 3:  # Log first few chunks
                            print(f"📦 Queued chunk {chunk_count} ({len(audio_chunk)} bytes)")

                if self.interrupted:
                    # Stop early due to interruption
                    await asyncio.sleep(0.1)
                    pass
                elif not first_chunk_played:
                    raise Exception("No audio chunks received")

                if self.interrupted:
                    player.stop_playback()
                    tts_end = asyncio.get_event_loop().time()
                    return tts_start or start_time, tts_end

                print(f"✅ Finished streaming {chunk_count} chunks")

                # Wait for all queued audio to finish playing
                await asyncio.sleep(0.3)  # Small buffer for last chunks
                while not player.audio_queue.empty():
                    await asyncio.sleep(0.1)

                # Give a bit more time for the last chunk to finish
                await asyncio.sleep(trailing_silence)

                tts_end = asyncio.get_event_loop().time()

                # Stop playback
                player.stop_playback()

                total_time = tts_end - start_time
                first_audio_latency = (first_chunk_time - start_time) if first_chunk_time else 0
                print(f"🎵 Instant mode completed in {total_time:.2f}s (first audio: {first_audio_latency:.3f}s)")

                return tts_start or start_time, tts_end

            except Exception as e:
                player.stop_playback()
                raise e

        except Exception as e:
            print(f"True instant streaming error: {e}")
            print("🔄 Falling back to file-based streaming")
            return await self._speak_with_hume_streaming(text, strategy, user_emotion, user_engagement, voice)
    async def _speak_with_openai(self, text: str, strategy: Strategy, user_emotion: float, user_engagement: float, voice: Optional[str] = None) -> Tuple[float, float]:
        """Generate speech using OpenAI TTS API (existing method)."""
        voice = voice or TTS_VOICE

        try:
            instructions = self._strategy_to_openai_instructions(strategy, user_emotion, user_engagement)

            start_time = asyncio.get_event_loop().time()

            loop = asyncio.get_event_loop()

            def _generate():
                response = self.client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    input=text,
                    instructions=instructions
                )
                return response.content

            print(f"📝 OpenAI TTS Instructions: {instructions[:200]}...")
            print(f"🗣️ TTS Text: {text[:100]}...")

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
            print(f"OpenAI TTS error: {e}")
            return start_time, start_time

    async def speak(
            self,
            text: str,
            strategy: Strategy,
            user_emotion: float = 0.5,
            user_engagement: float = 0.5,
            voice: Optional[str] = None,
            streaming_mode: str = "true_streaming",  # Changed default to true_streaming
    ) -> Tuple[float, float]:
        """
        Convert text to speech and play it using the selected engine.

        Args:
            streaming_mode:
                - "true_streaming": Plays chunks as they arrive with instant mode (requires voice)
                - "streaming": Collects chunks then plays with instant mode (requires voice)
                - "standard": Uses non-streaming endpoint (supports dynamic voice generation)

        For Hume voices (required for instant mode):
            - "Ava Song" - Female, warm and engaging (default for instant mode)
            - "Literature Professor" - Academic, authoritative
            - "Male English Actor" - Sophisticated male voice
            - "Female English Actor" - Professional female voice

        Note: Streaming modes use instant_mode=True for ultra-low latency.
              If no voice specified for streaming, defaults to "Ava Song".
        """

        if self.engine == "hume" and self.hume_client:
            if streaming_mode == "true_streaming":
                return await self._speak_with_hume_true_streaming(text, strategy, user_emotion, user_engagement, voice)
            elif streaming_mode == "streaming":
                return await self._speak_with_hume_streaming(text, strategy, user_emotion, user_engagement, voice)
            else:  # standard - allows dynamic voice generation
                return await self._speak_with_hume(text, strategy, user_emotion, user_engagement, voice)
        else:
            return await self._speak_with_openai(text, strategy, user_emotion, user_engagement, voice)

    def stop(self):
        """Stop current playback and set interruption flag."""
        self.interrupted = True
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False

        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                os.unlink(self.current_audio_file)
            except Exception:
                pass
            self.current_audio_file = None

    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        pygame.mixer.quit()