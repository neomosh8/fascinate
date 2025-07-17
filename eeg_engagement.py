# eeg_engagement.py
import numpy as np
from collections import deque
from scipy import signal
import asyncio
from typing import Optional, Callable
import time


class EngagementProcessor:
    """Real-time EEG engagement/attention calculation"""

    def __init__(self, sample_rate: int = 250, window_size: int = 2):
        self.sample_rate = sample_rate
        self.window_size = window_size  # seconds
        self.window_samples = sample_rate * window_size

        # Data buffers
        self.ch1_buffer = deque(maxlen=self.window_samples * 2)
        self.ch2_buffer = deque(maxlen=self.window_samples * 2)

        # Engagement metrics
        self.current_engagement = 0.5  # baseline
        self.engagement_history = deque(maxlen=100)

        # Callback for engagement updates
        self.engagement_callback: Optional[Callable] = None

        self.last_calculation = time.time()

    def set_engagement_callback(self, callback: Callable[[float], None]):
        """Set callback function to receive engagement updates"""
        self.engagement_callback = callback

    def add_eeg_data(self, ch1_data: list, ch2_data: list):
        """Add new EEG data and calculate engagement"""
        self.ch1_buffer.extend(ch1_data)
        self.ch2_buffer.extend(ch2_data)

        # Calculate engagement every 0.5 seconds
        if time.time() - self.last_calculation > 0.5:
            engagement = self._calculate_engagement()
            if engagement is not None:
                self.current_engagement = engagement
                self.engagement_history.append(engagement)

                if self.engagement_callback:
                    self.engagement_callback(engagement)

            self.last_calculation = time.time()

    def _calculate_engagement(self) -> Optional[float]:
        """Calculate engagement/attention metric from EEG data"""
        if len(self.ch1_buffer) < self.window_samples:
            return None

        # Get recent data
        ch1_data = np.array(list(self.ch1_buffer)[-self.window_samples:])
        ch2_data = np.array(list(self.ch2_buffer)[-self.window_samples:])

        # Calculate power spectral density
        freqs1, psd1 = signal.welch(ch1_data, self.sample_rate, nperseg=self.sample_rate // 2)
        freqs2, psd2 = signal.welch(ch2_data, self.sample_rate, nperseg=self.sample_rate // 2)

        # Average across channels
        psd_avg = (psd1 + psd2) / 2

        # Define frequency bands
        alpha_band = (8, 12)  # Alpha waves (relaxed awareness)
        beta_band = (13, 30)  # Beta waves (focused attention)
        theta_band = (4, 7)  # Theta waves (drowsiness)

        # Calculate band powers
        alpha_power = self._band_power(freqs1, psd_avg, alpha_band)
        beta_power = self._band_power(freqs1, psd_avg, beta_band)
        theta_power = self._band_power(freqs1, psd_avg, theta_band)

        # Engagement metric: beta/(alpha + theta) ratio
        # Higher beta = more focused, lower alpha+theta = less drowsy
        if alpha_power + theta_power > 0:
            engagement_raw = beta_power / (alpha_power + theta_power)
            # Normalize to 0-1 range using sigmoid
            engagement = 1 / (1 + np.exp(-0.5 * (engagement_raw - 2)))
        else:
            engagement = 0.5

        return float(np.clip(engagement, 0, 1))

    def _band_power(self, freqs: np.ndarray, psd: np.ndarray, band: tuple) -> float:
        """Calculate power in frequency band"""
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(psd[band_mask], freqs[band_mask])

    def get_engagement_change(self) -> float:
        """Get recent change in engagement (-1 to 1)"""
        if len(self.engagement_history) < 5:
            return 0.0

        recent_avg = np.mean(list(self.engagement_history)[-3:])
        baseline_avg = np.mean(list(self.engagement_history)[-10:-3])

        change = recent_avg - baseline_avg
        return float(np.clip(change * 5, -1, 1))  # Scale and clip