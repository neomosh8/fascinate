"""Speech-to-text using OpenAI Whisper API."""

import asyncio
import io
import wave
import numpy as np
from typing import Optional
import pyaudio
from openai import OpenAI

from config import WHISPER_MODEL, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE, MAX_RECORDING_DURATION


class SpeechToText:
    """Handles speech-to-text conversion using OpenAI Whisper."""

    def __init__(self):
        self.client = OpenAI()
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []

    def start_recording(self):
        """Start recording audio."""
        self.recording = True
        self.frames = []

        # Start recording in a separate thread
        import threading
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()

    def stop_recording(self) -> bytes:
        """Stop recording and return WAV data."""
        self.recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join(timeout=1.0)

        # Convert frames to WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))

        wav_buffer.seek(0)
        return wav_buffer.read()

    def _record_audio(self):
        """Record audio in a separate thread."""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE
        )

        max_frames = int(AUDIO_SAMPLE_RATE / AUDIO_CHUNK_SIZE * MAX_RECORDING_DURATION)
        frame_count = 0

        while self.recording and frame_count < max_frames:
            try:
                data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                self.frames.append(data)
                frame_count += 1
            except Exception as e:
                print(f"Recording error: {e}")
                break

        stream.stop_stream()
        stream.close()

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        try:
            # Create a file-like object
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "recording.wav"

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def _transcribe():
                return self.client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                    response_format="text"
                )

            result = await loop.run_in_executor(None, _transcribe)
            return result.strip()

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()