# custom_eeg_streamer.py
import asyncio
import struct
from bleak import BleakClient
from typing import Callable


class CustomEEGStreamer:
    """Custom EEG streamer that doesn't create plots, just feeds engagement processor"""

    def __init__(self, engagement_processor):
        self.engagement_processor = engagement_processor

        # BLE Configuration (from neocore_client)
        self.RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
        self.TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
        self.SAMPLES_PER_CHUNK = 27
        self.NUM_CHANNELS = 2

    def build_command(self, feature_id: int, pdu_id: int, payload: bytes = b"") -> bytes:
        """Build BLE command"""
        command_id = (feature_id << 9) | (0 << 7) | pdu_id  # PDU_TYPE_COMMAND = 0
        return command_id.to_bytes(2, 'big') + payload

    def build_stream_command(self, start: bool) -> bytes:
        """Build stream start/stop command"""
        payload = b"\x01" if start else b"\x00"
        return self.build_command(0x01, 0x00, payload)  # FEATURE_SENSOR_CFG, CMD_STREAM_CTRL

    def parse_eeg_packet(self, packet_data: bytes):
        """Parse EEG packet data"""
        if len(packet_data) < 4:
            return None, None

        cmd = packet_data[0]
        data_len = packet_data[1]

        if cmd != 0x02:
            return None, None

        sample_data = packet_data[4:4 + data_len]
        expected_len = self.SAMPLES_PER_CHUNK * self.NUM_CHANNELS * 4

        if len(sample_data) != expected_len:
            return None, None

        ch1_samples = []
        ch2_samples = []

        for i in range(0, len(sample_data), 8):
            ch1_value = struct.unpack('<i', sample_data[i:i + 4])[0]
            ch2_value = struct.unpack('<i', sample_data[i + 4:i + 8])[0]
            ch1_samples.append(float(ch1_value))
            ch2_samples.append(float(ch2_value))

        return ch1_samples, ch2_samples

    def notification_handler(self, sender: int, data: bytearray):
        """Handle BLE notifications"""
        try:
            if len(data) < 6:
                return
            ch1_samples, ch2_samples = self.parse_eeg_packet(data[2:])

            if ch1_samples is not None and ch2_samples is not None:
                # Feed to engagement processor (NO PLOTTING)
                self.engagement_processor.add_eeg_data(ch1_samples, ch2_samples)

        except Exception as e:
            print(f"EEG data parsing error: {e}")

    async def stream_data(self, device_address: str):
        """Stream EEG data"""
        print(f"Connecting to EEG device {device_address}...")

        async with BleakClient(device_address, timeout=20.0) as client:
            if not client.is_connected:
                raise RuntimeError("Failed to connect to EEG device")

            print("EEG Connected! Starting data stream...")

            try:
                await client.request_mtu(247)
            except:
                pass

            await client.start_notify(self.TX_UUID, self.notification_handler)

            start_cmd = self.build_stream_command(True)
            await client.write_gatt_char(self.RX_UUID, start_cmd, response=False)
            print("EEG streaming started!")

            try:
                # Keep streaming until interrupted
                while True:
                    await asyncio.sleep(1)
            except:
                print("\nStopping EEG stream...")
            finally:
                if client.is_connected:
                    stop_cmd = self.build_stream_command(False)
                    await client.write_gatt_char(self.RX_UUID, stop_cmd, response=False)
                    await client.stop_notify(self.TX_UUID)