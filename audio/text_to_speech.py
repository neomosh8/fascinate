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
        self.interrupted = False  # Add this flag

    def start_playback(self):
        """Start the audio playback stream."""
        self.interrupted = False  # Reset on start
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
        while self.playing and not self.interrupted:  # Check interrupted flag
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                if audio_chunk is None or self.interrupted:  # Check interrupted
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
        if self.playing and not self.interrupted:  # Check interrupted
            self.audio_queue.put(audio_data)

    def interrupt(self):
        """Interrupt playback immediately."""
        self.interrupted = True
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def stop_playback(self):
        """Stop playback and cleanup."""
        self.playing = False
        self.interrupted = True
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
                "professional": "composed, authoritative presence",
                "friendly": "warm, welcoming energy",
                "empathetic": "compassionate, deeply understanding",
                "playful": "gentle humor, lighthearted warmth",
                "confident": "steady, reassuring presence",
                "kind": "tender, nurturing care",
                "sarcastic": "subtle irony, knowing smile",
                "informational": "clear, educational guidance",
                "calm": "serene, grounding presence",

                # Additional therapeutic tones
                "validating": "accepting, affirming presence",
                "gentle": "soft, careful approach",
                "reflective": "thoughtful, contemplative",
                "supportive": "encouraging, uplifting",
                "curious": "interested, gently probing",
            }
            #also approach for therapeutic
            emotion_instructions = {
                # Original emotions
                "happy": "joyful, radiant energy",
                "serious": "focused, weighty presence",
                "thoughtful": "contemplative, reflective pause",
                "angry": "controlled intensity, firm",
                "flirting": "playful charm, teasing warmth",
                "sad": "gentle melancholy, tender",
                "scared": "cautious, protective whisper",
                "worried": "concerned, attentive care",
                "whisper": "intimate, hushed confidence",
                "laughter": "amused, joyful lightness",

                # Therapeutic approaches
                "cognitive": "structured, clear reasoning",
                "mindful": "present, grounded awareness",
                "exploratory": "curious, gentle discovery",
                "validating": "accepting, affirming truth",
                "somatic": "body-aware, gentle attunement",
                "narrative": "story-weaving, reflective",
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
                instructions.append("extra tender, protective")
            elif user_emotion > 0.6:
                instructions.append("upbeat and energetic")

            if user_engagement < 0.4:
                instructions.append("drawing them in, captivating")
            elif user_engagement > 0.7:  # High engagement
                instructions.append("building momentum")

            # Combine and ensure ≤100 characters
            acting_instruction = ", ".join(instructions)

            return acting_instruction

    def _get_hume_speed_from_strategy(self, strategy: Strategy, user_emotion: float, user_engagement: float) -> float:
        """Calculate Hume speed parameter (0.25-3.0) from strategy with therapeutic awareness."""

        # Get base TTS parameters (handle both regular and therapeutic strategies)
        try:
            tts_params = strategy.get_emotion_adapted_tts_params(user_emotion, user_engagement)
            base_speed = tts_params["speed"]
        except (AttributeError, KeyError):
            # Fallback for strategies without TTS params
            base_speed = 1.0

        # Map from OpenAI-style speed (0.7-1.3) to Hume range (0.25-3.0)
        # More granular mapping for better control
        if base_speed <= 0.75:
            hume_speed = 0.6  # Very slow for deep therapy
        elif base_speed <= 0.85:
            hume_speed = 0.75  # Slow for empathetic/gentle
        elif base_speed <= 0.95:
            hume_speed = 0.9  # Slightly slow
        elif base_speed <= 1.05:
            hume_speed = 1.0  # Normal pace
        elif base_speed <= 1.15:
            hume_speed = 1.2  # Slightly faster
        elif base_speed <= 1.25:
            hume_speed = 1.4  # Faster
        else:
            hume_speed = 1.6  # Energetic

        # Strategy-specific adjustments
        tone = strategy.tone.lower()
        emotion_or_approach = strategy.emotion.lower()

        # Tone-based adjustments (works for both regular and therapeutic)
        if tone in ["gentle", "empathetic", "calm", "validating"]:
            hume_speed *= 0.85  # Slower for therapeutic tones
        elif tone in ["confident", "professional"]:
            hume_speed *= 1.1  # Slightly faster for assertive tones
        elif tone in ["playful", "friendly"]:
            hume_speed *= 1.05  # Slightly faster for engaging tones

        # Emotion/Approach-based adjustments
        slow_emotions = ["whisper", "sad", "scared", "gentle melancholy",
                         "somatic", "mindful", "reflective"]
        fast_emotions = ["happy", "excited", "angry", "laughter",
                         "cognitive", "exploratory"]

        if emotion_or_approach in slow_emotions:
            hume_speed *= 0.8  # Slower for contemplative emotions/approaches
        elif emotion_or_approach in fast_emotions:
            hume_speed *= 1.15  # Faster for energetic emotions/approaches

        # User state adaptations
        if user_emotion < 0.3:
            hume_speed *= 0.85  # Much slower for very withdrawn users
        elif user_emotion < 0.4:
            hume_speed *= 0.9  # Slower for withdrawn users
        elif user_emotion > 0.7:
            hume_speed *= 1.1  # Faster for very positive users

        if user_engagement < 0.3:
            hume_speed *= 1.1  # Slightly faster to re-engage
        elif user_engagement > 0.8:
            hume_speed *= 0.95  # Slightly slower to maintain deep engagement

        # Therapeutic mode special handling
        if hasattr(strategy, 'exploration_mode'):
            if strategy.exploration_mode:
                hume_speed *= 0.95  # Slightly slower for exploration
            else:
                hume_speed *= 0.9  # Slower for deep exploitation

        # Ensure within Hume's bounds (fixed to match documentation)
        return max(0.25, min(2.5, hume_speed))  # Conservative upper bound

    def _get_trailing_silence_from_strategy(self, strategy: Strategy, user_emotion: float = 0.5,
                                            user_engagement: float = 0.5) -> float:
        """Get appropriate trailing silence based on strategy and user state."""

        base_silence = 0.4  # Default pause

        tone = strategy.tone.lower()
        emotion_or_approach = strategy.emotion.lower()

        # TONE-BASED SILENCE (works for both regular and therapeutic)
        tone_silence_map = {
            # Therapeutic tones - longer pauses for processing
            "empathetic": 0.7,  # Allow emotional processing
            "validating": 0.6,  # Let affirmation sink in
            "gentle": 0.8,  # Extra gentle spacing
            "supportive": 0.5,  # Moderate supportive pause
            "reflective": 0.9,  # Long pause for reflection
            "curious": 0.4,  # Shorter for maintaining flow

            # Regular tones
            "kind": 0.6,  # Caring pause
            "calm": 0.7,  # Peaceful spacing
            "professional": 0.3,  # Efficient timing
            "confident": 0.3,  # Assertive timing
            "playful": 0.2,  # Quick, energetic
            "friendly": 0.4,  # Natural conversation
            "sarcastic": 0.5,  # Let sarcasm land
            "informational": 0.3,  # Clear, direct
        }

        # EMOTION/APPROACH-BASED SILENCE
        emotion_silence_map = {
            # Contemplative emotions - longer pauses
            "thoughtful": 1.0,
            "serious": 0.8,
            "sad": 0.9,
            "reflective": 1.0,
            "worried": 0.7,

            # Gentle emotions - medium pauses
            "whisper": 0.8,
            "scared": 0.6,

            # Energetic emotions - shorter pauses
            "happy": 0.3,
            "excited": 0.2,
            "laughter": 0.1,
            "playful": 0.2,

            # Therapeutic approaches
            "mindful": 0.9,  # Allow mindful processing
            "somatic": 1.0,  # Body awareness needs time
            "narrative": 0.7,  # Story processing
            "cognitive": 0.5,  # Structured thinking
            "exploratory": 0.6,  # Gentle discovery
            "validating": 0.6,  # Affirmation processing
        }

        # Start with tone-based silence
        if tone in tone_silence_map:
            base_silence = tone_silence_map[tone]

        # Override with emotion/approach if more specific
        if emotion_or_approach in emotion_silence_map:
            emotion_silence = emotion_silence_map[emotion_or_approach]
            # Use the longer of tone or emotion silence (more contemplative)
            base_silence = max(base_silence, emotion_silence)

        # USER STATE ADAPTATIONS

        # User emotional state adjustments
        if user_emotion < 0.3:
            base_silence *= 1.4  # Much longer pause for very distressed users
        elif user_emotion < 0.4:
            base_silence *= 1.2  # Longer pause for withdrawn users
        elif user_emotion > 0.7:
            base_silence *= 0.8  # Shorter pause for positive users

        # User engagement adjustments
        if user_engagement < 0.3:
            base_silence *= 0.7  # Shorter pause to re-engage lost users
        elif user_engagement > 0.8:
            base_silence *= 1.1  # Slightly longer to maintain deep engagement

        # THERAPEUTIC MODE SPECIAL HANDLING
        if hasattr(strategy, 'exploration_mode'):
            if strategy.exploration_mode:
                base_silence *= 1.1  # Longer pauses for exploration (let things settle)
            else:
                base_silence *= 1.3  # Much longer for exploitation (deep processing)

        # CONTEXT-AWARE BOUNDS
        # Ensure silence is appropriate for the context
        if user_engagement < 0.2:
            # Very disengaged - don't let silence get too long
            max_silence = 0.8
        elif hasattr(strategy, 'exploration_mode') and not strategy.exploration_mode:
            # Deep exploitation mode - allow longer silences
            max_silence = 1.5
        else:
            # Normal conversation
            max_silence = 1.2

        # Final bounds with context awareness
        final_silence = max(0.1, min(max_silence, base_silence))

        return round(final_silence, 1)  # Round to 1 decimal for cleaner values

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

    async def _speak_with_hume_true_streaming(self, text: str, strategy: Strategy, user_emotion: float,
                                              user_engagement: float, voice: Optional[str] = None) -> Tuple[
        float, float]:
        """TRUE streaming with instant mode - plays audio chunks as they arrive using PyAudio."""

        self.interrupted = False

        if not self.hume_client:
            return await self._speak_with_openai(text, strategy, user_emotion, user_engagement, voice)

        if not PYAUDIO_AVAILABLE:
            print("⚠️ PyAudio not available, falling back to file-based streaming")
            return await self._speak_with_hume(text, strategy, user_emotion, user_engagement, voice)

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
                    # Check for interruption MORE frequently
                    if self.interrupted:
                        self.logger.info("TTS interrupted by user during streaming")
                        player.interrupt()  # Interrupt the player immediately
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

                        # Queue this chunk for immediate playback (only if not interrupted)
                        if not self.interrupted:
                            player.queue_audio(audio_chunk)
                            if chunk_count <= 3:  # Log first few chunks
                                print(f"📦 Queued chunk {chunk_count} ({len(audio_chunk)} bytes)")

                # Handle interruption case
                if self.interrupted:
                    print("🛑 Streaming interrupted - stopping immediately")
                    player.stop_playback()
                    tts_end = asyncio.get_event_loop().time()
                    return tts_start or start_time, tts_end

                if not first_chunk_played:
                    raise Exception("No audio chunks received")

                print(f"✅ Finished streaming {chunk_count} chunks")

                # Wait for all queued audio to finish playing (unless interrupted)
                if not self.interrupted:
                    await asyncio.sleep(0.3)  # Small buffer for last chunks
                    while not player.audio_queue.empty() and not self.interrupted:
                        await asyncio.sleep(0.1)

                    # Give a bit more time for the last chunk to finish
                    if not self.interrupted:
                        await asyncio.sleep(trailing_silence)

                tts_end = asyncio.get_event_loop().time()

                # Stop playback
                player.stop_playback()

                total_time = tts_end - start_time
                first_audio_latency = (first_chunk_time - start_time) if first_chunk_time else 0
                if self.interrupted:
                    print(f"🛑 Instant mode interrupted after {total_time:.2f}s")
                else:
                    print(f"🎵 Instant mode completed in {total_time:.2f}s (first audio: {first_audio_latency:.3f}s)")

                return tts_start or start_time, tts_end

            except Exception as e:
                player.stop_playback()
                raise e

        except Exception as e:
            if not self.interrupted:  # Don't log interruption as error
                print(f"True instant streaming error: {e}")
            print("🔄 Falling back to file-based streaming")
            return await self._speak_with_hume(text, strategy, user_emotion, user_engagement, voice)

    async def _speak_with_openai(self, text: str, strategy: Strategy, user_emotion: float, user_engagement: float,
                                 voice: Optional[str] = None) -> Tuple[float, float]:
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

            # Wait for playback to complete with interruption checking
            while pygame.mixer.music.get_busy() and not self.interrupted:
                await asyncio.sleep(0.1)

            tts_end = asyncio.get_event_loop().time()
            self.is_playing = False

            # Clean up temp file
            try:
                os.unlink(self.current_audio_file)
            except:
                pass
            self.current_audio_file = None

            if self.interrupted:
                print("🛑 OpenAI TTS interrupted")

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
            else:  # standard - allows dynamic voice generation
                return await self._speak_with_hume(text, strategy, user_emotion, user_engagement, voice)
        else:
            return await self._speak_with_openai(text, strategy, user_emotion, user_engagement, voice)

    def stop(self):
        """Stop current playback and set interruption flag."""
        self.interrupted = True

        # Stop pygame audio
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False

        # Clean up temp file
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                os.unlink(self.current_audio_file)
            except Exception:
                pass
            self.current_audio_file = None

        self.logger.info("TTS stopped and interrupted")
    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        pygame.mixer.quit()