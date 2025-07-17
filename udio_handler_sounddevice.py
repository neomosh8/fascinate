# audio_handler_continuous.py
import asyncio
import sounddevice as sd
import numpy as np
import threading
from scipy import signal


class AudioHandler:
    """Continuous buffer audio handler"""

    def __init__(self):
        self.openai_sample_rate = 24000
        self.channels = 1

        device_info = sd.query_devices(sd.default.device[1])
        self.system_sample_rate = int(device_info['default_samplerate'])

        print(f"🔊 Audio: {self.openai_sample_rate}Hz → {self.system_sample_rate}Hz")

        # Large continuous buffer (30 seconds)
        buffer_duration = 30  # seconds
        self.buffer_size = self.system_sample_rate * buffer_duration
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # Circular buffer pointers
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

    def set_event_loop(self, loop):
        self.loop = loop

    def start_recording(self):
        if self.recording:
            return

        self.recording = True

        try:
            self.input_stream = sd.InputStream(
                samplerate=self.openai_sample_rate,
                channels=self.channels,
                dtype='int16',
                callback=self._input_callback,
                blocksize=512
            )

            self.input_stream.start()
            print(f"🎤 Microphone started")
        except Exception as e:
            print(f"Failed to start microphone: {e}")
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
            print(f"Failed to start speaker: {e}")
            self.playing = False

    def _input_callback(self, indata, frames, time, status):
        if self.recording and self.loop:
            audio_bytes = indata.astype(np.int16).tobytes()
            asyncio.run_coroutine_threadsafe(
                self.audio_input_queue.put(audio_bytes),
                self.loop
            )

    def _output_callback(self, outdata, frames, time, status):
        """Continuous buffer output"""
        with self.buffer_lock:
            output_data = np.zeros(frames, dtype=np.float32)

            # Calculate available samples
            if self.write_pos >= self.read_pos:
                available = self.write_pos - self.read_pos
            else:
                available = self.buffer_size - self.read_pos + self.write_pos

            # Only play if we have enough samples (prevents underruns)
            samples_to_read = min(frames, available)

            if samples_to_read > 0:
                for i in range(samples_to_read):
                    output_data[i] = self.audio_buffer[self.read_pos]
                    self.read_pos = (self.read_pos + 1) % self.buffer_size

            outdata[:, 0] = output_data

    def play_audio(self, audio_data: bytes):
        """Add audio to continuous buffer"""
        if not self.playing or not audio_data:
            return

        try:
            # Convert and resample
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Resample if needed
            if self.system_sample_rate != self.openai_sample_rate:
                num_samples_out = int(len(audio_float) * self.system_sample_rate / self.openai_sample_rate)
                if num_samples_out > 0:
                    audio_resampled = signal.resample(audio_float, num_samples_out)
                    audio_float = audio_resampled.astype(np.float32)

            # Add to circular buffer
            with self.buffer_lock:
                for sample in audio_float:
                    self.audio_buffer[self.write_pos] = sample
                    self.write_pos = (self.write_pos + 1) % self.buffer_size

                    # Prevent overwriting unplayed audio
                    if self.write_pos == self.read_pos:
                        # Buffer full - advance read position to make space
                        self.read_pos = (self.read_pos + 1) % self.buffer_size

        except Exception as e:
            print(f"Error adding audio: {e}")

    async def get_audio_input(self):
        return await self.audio_input_queue.get()

    def clear_audio_buffer(self):
        with self.buffer_lock:
            self.audio_buffer.fill(0)
            self.write_pos = 0
            self.read_pos = 0

    def get_buffer_status(self):
        with self.buffer_lock:
            if self.write_pos >= self.read_pos:
                buffered = self.write_pos - self.read_pos
            else:
                buffered = self.buffer_size - self.read_pos + self.write_pos

            duration = buffered / self.system_sample_rate
            return {'buffered_duration': duration, 'buffered_samples': buffered}

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