# audio_handler_debug.py - Enhanced with debugging
import asyncio
import sounddevice as sd
import numpy as np
import threading
import time
from collections import deque
from scipy import signal
import base64


class AudioHandler:
    """Audio handler with extensive debugging"""

    def __init__(self):
        self.openai_sample_rate = 24000
        self.channels = 1

        device_info = sd.query_devices(sd.default.device[1])
        self.system_sample_rate = int(device_info['default_samplerate'])

        print(f"🔊 Audio: {self.openai_sample_rate}Hz → {self.system_sample_rate}Hz")

        # Debug counters
        self.audio_chunks_captured = 0
        self.audio_chunks_sent = 0
        self.last_audio_time = 0

        # Circular buffer
        buffer_duration = 30
        self.buffer_size = self.system_sample_rate * buffer_duration
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.buffer_lock = threading.RLock()

        # Input
        self.audio_input_queue = asyncio.Queue()

        # Control
        self.recording = False
        self.playing = False
        self.loop = None

        # Streams
        self.input_stream = None
        self.output_stream = None

        # Callback for sending audio
        self.audio_send_callback = None

    def set_event_loop(self, loop):
        self.loop = loop

    def set_audio_send_callback(self, callback):
        """Set callback for sending audio to OpenAI"""
        self.audio_send_callback = callback

    def start_recording(self):
        if self.recording:
            return

        self.recording = True

        try:
            # Check microphone permissions first
            print("🎤 Checking microphone access...")

            self.input_stream = sd.InputStream(
                samplerate=self.openai_sample_rate,
                channels=self.channels,
                dtype='int16',
                callback=self._input_callback,
                blocksize=512
            )

            self.input_stream.start()
            print(f"🎤 Microphone started successfully")

            # Start audio processing task
            if self.loop:
                asyncio.run_coroutine_threadsafe(self._process_audio_input(), self.loop)

        except Exception as e:
            print(f"❌ Failed to start microphone: {e}")
            print("💡 Try: System Preferences → Security & Privacy → Microphone")
            self.recording = False

    def start_playback(self):
        if self.playing:
            return

        self.playing = True

        try:
            self.output_stream = sd.OutputStream(
                samplerate=self.system_sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=self._output_callback,
                blocksize=512
            )

            self.output_stream.start()
            print(f"🔊 Speaker started")
        except Exception as e:
            print(f"❌ Failed to start speaker: {e}")
            self.playing = False

    def _input_callback(self, indata, frames, time_info, status):
        """Enhanced input callback with debugging"""
        if status:
            print(f"⚠️ Audio input status: {status}")

        if self.recording and self.loop:
            # Check if we're getting actual audio
            max_level = np.max(np.abs(indata))

            if max_level > 100:  # Only process if there's actual audio
                self.audio_chunks_captured += 1
                self.last_audio_time = time.time()

                # Show audio level every 50 chunks
                if self.audio_chunks_captured % 50 == 0:
                    print(f"🎤 Audio captured: chunk {self.audio_chunks_captured}, level: {max_level:.0f}")

                audio_bytes = indata.astype(np.int16).tobytes()
                asyncio.run_coroutine_threadsafe(
                    self.audio_input_queue.put(audio_bytes),
                    self.loop
                )

    async def _process_audio_input(self):
        """Process audio input and send to OpenAI"""
        print("🎤 Audio processing started")

        while self.recording:
            try:
                # Get audio data
                audio_data = await asyncio.wait_for(self.audio_input_queue.get(), timeout=1.0)

                # Send to OpenAI if callback is set
                if self.audio_send_callback:
                    await self.audio_send_callback(audio_data)
                    self.audio_chunks_sent += 1

                    # Debug info
                    if self.audio_chunks_sent % 20 == 0:
                        print(f"📤 Sent {self.audio_chunks_sent} audio chunks to OpenAI")

            except asyncio.TimeoutError:
                # No audio received - check if mic is working
                if time.time() - self.last_audio_time > 10:  # 10 seconds of silence
                    print("🔇 No audio detected for 10 seconds - check microphone")
                continue
            except Exception as e:
                print(f"❌ Audio processing error: {e}")
                await asyncio.sleep(0.1)

    def _output_callback(self, outdata, frames, time_info, status):
        """Output callback for audio playback"""
        with self.buffer_lock:
            output_data = np.zeros(frames, dtype=np.float32)

            if self.write_pos >= self.read_pos:
                buffered = self.write_pos - self.read_pos
            else:
                buffered = self.buffer_size - self.read_pos + self.write_pos

            samples_to_read = min(frames, buffered)

            if samples_to_read > 0:
                for i in range(samples_to_read):
                    output_data[i] = self.audio_buffer[self.read_pos]
                    self.read_pos = (self.read_pos + 1) % self.buffer_size

            outdata[:, 0] = output_data

    def play_audio(self, audio_data: bytes):
        """Add audio to playback buffer"""
        if not self.playing or not audio_data:
            return

        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            if self.system_sample_rate != self.openai_sample_rate:
                num_samples_out = int(len(audio_float) * self.system_sample_rate / self.openai_sample_rate)
                if num_samples_out > 0:
                    audio_resampled = signal.resample(audio_float, num_samples_out)
                    audio_float = audio_resampled.astype(np.float32)

            with self.buffer_lock:
                for sample in audio_float:
                    self.audio_buffer[self.write_pos] = sample
                    self.write_pos = (self.write_pos + 1) % self.buffer_size

                    if self.write_pos == self.read_pos:
                        self.read_pos = (self.read_pos + 1) % self.buffer_size

        except Exception as e:
            print(f"❌ Error adding audio: {e}")

    async def get_audio_input(self):
        """Get audio input - for compatibility"""
        return await self.audio_input_queue.get()

    def clear_audio_buffer(self):
        with self.buffer_lock:
            self.audio_buffer.fill(0)
            self.write_pos = 0
            self.read_pos = 0

    def get_debug_stats(self):
        """Get debugging statistics"""
        return {
            'chunks_captured': self.audio_chunks_captured,
            'chunks_sent': self.audio_chunks_sent,
            'last_audio_ago': time.time() - self.last_audio_time if self.last_audio_time > 0 else None,
            'recording': self.recording,
            'playing': self.playing
        }

    def stop_recording(self):
        self.recording = False
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()

    def stop_playback(self):
        self.playing = False
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()

    def cleanup(self):
        self.stop_recording()
        self.stop_playback()
        sd.stop()