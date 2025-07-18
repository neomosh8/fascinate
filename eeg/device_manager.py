"""EEG device connection and data management."""

import asyncio
from typing import Optional, Callable, Tuple
import numpy as np
from collections import deque
import sys
import os

# Add parent directory to path to import neocore_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neocore_client import EEGStreamer, find_device, build_stream_command

from config import EEG_SAMPLE_RATE


class EEGDeviceManager:
    """Manages connection to Neocore EEG device."""

    def __init__(self, data_callback: Optional[Callable] = None):
        self.device_address = None
        self.client = None
        self.is_streaming = False
        self.data_callback = data_callback

        # Buffer for raw EEG data
        self.buffer_size = EEG_SAMPLE_RATE * 10  # 10 seconds
        self.ch1_buffer = deque(maxlen=self.buffer_size)
        self.ch2_buffer = deque(maxlen=self.buffer_size)

    async def connect(self, target_mac: Optional[str] = None) -> bool:
        """Connect to EEG device."""
        try:
            self.device_address = await find_device(target_mac)
            return True
        except Exception as e:
            print(f"Failed to find device: {e}")
            return False

    def notification_handler(self, sender: int, data: bytearray):
        """Handle incoming EEG data."""
        try:
            from neocore_client import parse_eeg_packet

            if len(data) < 6:
                return

            ch1_samples, ch2_samples = parse_eeg_packet(data[2:])

            # Add to buffers
            self.ch1_buffer.extend(ch1_samples)
            self.ch2_buffer.extend(ch2_samples)

            # Call callback if provided
            if self.data_callback:
                self.data_callback(ch1_samples, ch2_samples)

        except Exception as e:
            print(f"Error parsing EEG data: {e}")

    async def start_streaming(self, client) -> bool:
        """Start EEG data streaming."""
        try:
            from neocore_client import RX_UUID, TX_UUID

            await client.start_notify(TX_UUID, self.notification_handler)

            start_cmd = build_stream_command(True)
            await client.write_gatt_char(RX_UUID, start_cmd, response=False)

            self.is_streaming = True
            self.client = client
            return True

        except Exception as e:
            print(f"Failed to start streaming: {e}")
            return False

    async def stop_streaming(self):
        """Stop EEG data streaming."""
        if self.client and self.is_streaming:
            from neocore_client import RX_UUID, TX_UUID

            try:
                stop_cmd = build_stream_command(False)
                await self.client.write_gatt_char(RX_UUID, stop_cmd, response=False)
                await self.client.stop_notify(TX_UUID)
                self.is_streaming = False
            except Exception as e:
                print(f"Error stopping stream: {e}")

    def get_recent_data(self, seconds: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Get recent EEG data from buffers."""
        num_samples = int(seconds * EEG_SAMPLE_RATE)

        ch1_data = np.array(list(self.ch1_buffer)[-num_samples:]) if len(self.ch1_buffer) >= num_samples else np.array(
            [])
        ch2_data = np.array(list(self.ch2_buffer)[-num_samples:]) if len(self.ch2_buffer) >= num_samples else np.array(
            [])

        return ch1_data, ch2_data