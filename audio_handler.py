# audio_handler.py
import asyncio
import pyaudio
import threading
import queue
import base64
import struct
import numpy as np


class AudioHandler:
    """Handle microphone input and speaker output for WebSocket"""

    def __init__(self):
        # OpenAI Realtime API uses 24kHz PCM16
        self.sample_rate = 24000
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        self.audio = pyaudio.PyAudio()

        # Audio streams
        self.input_stream = None
        self.output_stream = None

        # Queues for audio data
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_buffer = queue.Queue(maxsize=50)  # Larger buffer

        # Control flags
        self.recording = False
        self.playing = False

        # For thread-safe communication
        self.loop = None

    def set_event_loop(self, loop):
        """Set the asyncio event loop for thread-safe operations"""
        self.loop = loop

    def start_recording(self):
        """Start recording from microphone"""
        if self.recording:
            return

        self.recording = True

        try:
            # Input stream - match OpenAI format
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._input_callback
            )

            self.input_stream.start_stream()
            print("🎤 Microphone started (24kHz PCM16)")
        except Exception as e:
            print(f"Failed to start microphone: {e}")
            self.recording = False

    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        print("🎤 Microphone stopped")

    def start_playback(self):
        """Start audio playback"""
        if self.playing:
            return

        self.playing = True

        try:
            # Output stream - match OpenAI format
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._output_callback
            )

            self.output_stream.start_stream()
            print("🔊 Speaker started (24kHz PCM16)")
        except Exception as e:
            print(f"Failed to start speaker: {e}")
            self.playing = False

    def stop_playback(self):
        """Stop playback"""
        self.playing = False
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        print("🔊 Speaker stopped")

    def _input_callback(self, in_data, frame_count, time_info, status):
        """Microphone input callback"""
        if self.recording and self.loop:
            # Thread-safe way to put data into asyncio queue
            asyncio.run_coroutine_threadsafe(
                self.audio_input_queue.put(in_data),
                self.loop
            )
        return (None, pyaudio.paContinue)

    def _output_callback(self, in_data, frame_count, time_info, status):
        """Speaker output callback - improved buffering"""
        try:
            # Try to get enough audio data for smooth playback
            audio_chunks = []
            total_bytes = 0
            target_bytes = frame_count * 2  # 2 bytes per PCM16 sample

            # Collect audio chunks until we have enough
            while total_bytes < target_bytes and not self.audio_output_buffer.empty():
                try:
                    chunk = self.audio_output_buffer.get_nowait()
                    audio_chunks.append(chunk)
                    total_bytes += len(chunk)
                except queue.Empty:
                    break

            if audio_chunks:
                # Combine all chunks
                audio_data = b''.join(audio_chunks)

                # Ensure we have exactly the right amount of data
                if len(audio_data) < target_bytes:
                    # Pad with silence if too short
                    audio_data += b'\x00' * (target_bytes - len(audio_data))
                elif len(audio_data) > target_bytes:
                    # Truncate if too long, save remainder
                    remainder = audio_data[target_bytes:]
                    audio_data = audio_data[:target_bytes]
                    # Put remainder back in queue
                    if remainder:
                        try:
                            self.audio_output_buffer.put_nowait(remainder)
                        except queue.Full:
                            pass  # Drop if queue is full

                return (audio_data, pyaudio.paContinue)
            else:
                # No audio data available, return silence
                return (b'\x00' * target_bytes, pyaudio.paContinue)

        except Exception as e:
            print(f"Audio output error: {e}")
            return (b'\x00' * frame_count * 2, pyaudio.paContinue)

    async def get_audio_input(self):
        """Get microphone input (async)"""
        return await self.audio_input_queue.get()

    def play_audio(self, audio_data: bytes):
        """Add audio data to playback queue - improved"""
        if self.playing and audio_data:
            try:
                # Don't let the queue get too full to avoid latency
                if self.audio_output_buffer.qsize() < 40:
                    self.audio_output_buffer.put_nowait(audio_data)
                else:
                    # Queue is full, drop oldest audio to prevent buildup
                    try:
                        self.audio_output_buffer.get_nowait()  # Remove oldest
                        self.audio_output_buffer.put_nowait(audio_data)  # Add new
                    except:
                        pass
            except queue.Full:
                # If queue is full, just drop this chunk
                pass

    def clear_audio_buffer(self):
        """Clear the audio output buffer"""
        while not self.audio_output_buffer.empty():
            try:
                self.audio_output_buffer.get_nowait()
            except queue.Empty:
                break

    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        self.stop_playback()
        self.audio.terminate()