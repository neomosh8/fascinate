"""EEG device connection and data management with PROVEN filtering."""

import asyncio
from typing import Optional, Callable, Tuple
import numpy as np
from collections import deque
import sys
import os

# Add parent directory to path to import neocore_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neocore_client import find_device, build_stream_command, parse_eeg_packet

from config import EEG_SAMPLE_RATE
from eeg.online_filter import OnlineFilter


class EEGDeviceManager:
    """Manages connection to Neocore EEG device with PROVEN filtering pipeline."""

    def __init__(self, data_callback: Optional[Callable] = None):
        self.device_address = None
        self.client = None
        self.is_streaming = False
        self.data_callback = data_callback

        # Add the PROVEN filter that worked in neocore_client.py
        self.online_filter = OnlineFilter(EEG_SAMPLE_RATE)

        # Buffer for FILTERED EEG data (not raw!)
        self.buffer_size = EEG_SAMPLE_RATE * 10  # 10 seconds
        self.ch1_buffer = deque(maxlen=self.buffer_size)
        self.ch2_buffer = deque(maxlen=self.buffer_size)

        # Also keep raw data for debugging
        self.ch1_raw_buffer = deque(maxlen=self.buffer_size)
        self.ch2_raw_buffer = deque(maxlen=self.buffer_size)

    async def connect(self, target_mac: Optional[str] = None) -> bool:
        """Connect to EEG device."""
        try:
            self.device_address = await find_device(target_mac)
            print(f"✅ Found Neocore device: {self.device_address}")
            return True
        except Exception as e:
            print(f"Failed to find device: {e}")
            return False

    def notification_handler(self, sender: int, data: bytearray):
        """Handle incoming EEG data with PROPER filtering applied immediately."""
        try:
            if len(data) < 6:
                return

            # Parse raw EEG data
            ch1_samples, ch2_samples = parse_eeg_packet(data[2:])

            # Store raw data for debugging
            self.ch1_raw_buffer.extend(ch1_samples)
            self.ch2_raw_buffer.extend(ch2_samples)

            # Convert to numpy arrays
            ch1_array = np.array(ch1_samples)
            ch2_array = np.array(ch2_samples)

            # Apply the PROVEN filter pipeline immediately!
            ch1_filtered, ch2_filtered = self.online_filter.filter_chunk(ch1_array, ch2_array)

            # Store FILTERED data in main buffers
            self.ch1_buffer.extend(ch1_filtered.tolist())
            self.ch2_buffer.extend(ch2_filtered.tolist())

            # Call callback with FILTERED data (not raw!)
            if self.data_callback:
                self.data_callback(ch1_filtered.tolist(), ch2_filtered.tolist())

            # Debug output every 100 packets to show filtering effect
            if len(self.ch1_raw_buffer) % 2700 == 0:  # Every ~10 seconds
                raw_range = max(ch1_samples) - min(ch1_samples)
                filtered_range = np.max(ch1_filtered) - np.min(ch1_filtered)
                print(f"📊 FILTER PERFORMANCE: Raw {raw_range:,.0f}µV → Filtered {filtered_range:.0f}µV")

        except Exception as e:
            print(f"Error processing EEG data: {e}")

    async def start_streaming(self, client) -> bool:
        """Start EEG data streaming."""
        try:
            from neocore_client import RX_UUID, TX_UUID

            await client.start_notify(TX_UUID, self.notification_handler)

            start_cmd = build_stream_command(True)
            await client.write_gatt_char(RX_UUID, start_cmd, response=False)

            self.is_streaming = True
            self.client = client
            print("🎯 EEG streaming started with PROVEN filtering pipeline!")
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
                print("📴 EEG streaming stopped")
            except Exception as e:
                print(f"Error stopping stream: {e}")

    def get_recent_data(self, seconds: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Get recent FILTERED EEG data from buffers."""
        num_samples = int(seconds * EEG_SAMPLE_RATE)

        # Return FILTERED data, not raw!
        ch1_data = np.array(list(self.ch1_buffer)[-num_samples:]) if len(self.ch1_buffer) >= num_samples else np.array([])
        ch2_data = np.array(list(self.ch2_buffer)[-num_samples:]) if len(self.ch2_buffer) >= num_samples else np.array([])

        return ch1_data, ch2_data

    def get_recent_raw_data(self, seconds: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Get recent RAW EEG data for debugging purposes."""
        num_samples = int(seconds * EEG_SAMPLE_RATE)

        ch1_raw = np.array(list(self.ch1_raw_buffer)[-num_samples:]) if len(self.ch1_raw_buffer) >= num_samples else np.array([])
        ch2_raw = np.array(list(self.ch2_raw_buffer)[-num_samples:]) if len(self.ch2_raw_buffer) >= num_samples else np.array([])

        return ch1_raw, ch2_raw

    def get_signal_quality_report(self) -> dict:
        """Generate signal quality report comparing raw vs filtered."""
        if len(self.ch1_buffer) < 100 or len(self.ch1_raw_buffer) < 100:
            return {"error": "Insufficient data"}

        # Get recent data
        raw_ch1 = np.array(list(self.ch1_raw_buffer)[-250:])  # 1 second
        raw_ch2 = np.array(list(self.ch2_raw_buffer)[-250:])
        filt_ch1 = np.array(list(self.ch1_buffer)[-250:])
        filt_ch2 = np.array(list(self.ch2_buffer)[-250:])

        return {
            "raw_signal": {
                "ch1_range": float(np.max(raw_ch1) - np.min(raw_ch1)),
                "ch2_range": float(np.max(raw_ch2) - np.min(raw_ch2)),
                "ch1_std": float(np.std(raw_ch1)),
                "ch2_std": float(np.std(raw_ch2))
            },
            "filtered_signal": {
                "ch1_range": float(np.max(filt_ch1) - np.min(filt_ch1)),
                "ch2_range": float(np.max(filt_ch2) - np.min(filt_ch2)),
                "ch1_std": float(np.std(filt_ch1)),
                "ch2_std": float(np.std(filt_ch2))
            },
            "improvement_ratio": {
                "ch1": float((np.max(raw_ch1) - np.min(raw_ch1)) / (np.max(filt_ch1) - np.min(filt_ch1))) if np.max(filt_ch1) != np.min(filt_ch1) else 0,
                "ch2": float((np.max(raw_ch2) - np.min(raw_ch2)) / (np.max(filt_ch2) - np.min(filt_ch2))) if np.max(filt_ch2) != np.min(filt_ch2) else 0
            },
            "filter_info": self.online_filter.get_filter_info()
        }