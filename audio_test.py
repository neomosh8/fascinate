# audio_debug.py - Standalone audio testing
import sounddevice as sd
import numpy as np
import time
import base64


def test_microphone():
    """Test if microphone is working"""
    print("🎤 Testing microphone...")

    # List available audio devices
    print("\n📱 Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (input)")

    # Test recording
    print(f"\n🔊 Default input device: {sd.query_devices(sd.default.device[0])['name']}")

    duration = 3  # seconds
    sample_rate = 24000

    print(f"Recording for {duration} seconds...")
    try:
        audio_data = sd.rec(int(duration * sample_rate),
                            samplerate=sample_rate,
                            channels=1,
                            dtype='int16')
        sd.wait()  # Wait for recording to complete

        # Check if we got actual audio
        max_value = np.max(np.abs(audio_data))
        print(f"Max audio value: {max_value}")

        if max_value > 100:  # Arbitrary threshold
            print("✅ Microphone is working - audio detected!")

            # Test conversion to base64 (what OpenAI expects)
            audio_bytes = audio_data.astype(np.int16).tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            print(f"✅ Audio conversion successful - {len(audio_b64)} chars")

        else:
            print("❌ No audio detected - check microphone permissions/settings")

    except Exception as e:
        print(f"❌ Recording failed: {e}")


if __name__ == "__main__":
    test_microphone()