# audio_test.py - Test audio system independently
import pyaudio
import time


def test_audio_system():
    """Test if audio system works properly"""
    print("Testing audio system...")

    audio = pyaudio.PyAudio()

    # Test output device
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=1024
        )

        print("✅ Audio output device working")

        # Generate a test tone
        import numpy as np
        duration = 1  # seconds
        sample_rate = 24000
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)

        print("Playing test tone...")
        stream.write(tone.tobytes())

        stream.close()

    except Exception as e:
        print(f"❌ Audio output test failed: {e}")

    audio.terminate()


if __name__ == "__main__":
    test_audio_system()