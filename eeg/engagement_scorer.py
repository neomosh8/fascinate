"""Calculate engagement scores from EEG data."""

import numpy as np
from scipy import signal
from collections import deque
from typing import Tuple, Optional

from config import EEG_SAMPLE_RATE, BETA_BAND, ENGAGEMENT_WINDOW_SEC


class EngagementScorer:
    """Computes engagement/attention scores from EEG data."""

    def __init__(self):
        # Design bandpass filter for beta band
        nyq = EEG_SAMPLE_RATE / 2
        self.sos = signal.butter(
            4,
            [BETA_BAND[0] / nyq, BETA_BAND[1] / nyq],
            btype='band',
            output='sos'
        )

        # Initialize filter states
        self.zi_ch1 = signal.sosfilt_zi(self.sos)
        self.zi_ch2 = signal.sosfilt_zi(self.sos)
        self.filter_initialized = False

        # Engagement score buffer for smoothing
        self.engagement_buffer = deque(
            maxlen=int(ENGAGEMENT_WINDOW_SEC * 10)  # 10 scores per second
        )
        self.current_engagement = 0.5  # Start at neutral

    def process_chunk(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> float:
        """Process EEG chunk and return engagement score."""
        if len(ch1_data) == 0 or len(ch2_data) == 0:
            return self.current_engagement

        # Initialize filter states on first run
        if not self.filter_initialized:
            self.zi_ch1 *= ch1_data[0]
            self.zi_ch2 *= ch2_data[0]
            self.filter_initialized = True

        # Apply bandpass filter
        ch1_filtered, self.zi_ch1 = signal.sosfilt(self.sos, ch1_data, zi=self.zi_ch1)
        ch2_filtered, self.zi_ch2 = signal.sosfilt(self.sos, ch2_data, zi=self.zi_ch2)

        # Compute beta power
        ch1_power = np.mean(ch1_filtered ** 2)
        ch2_power = np.mean(ch2_filtered ** 2)

        # Average across channels
        avg_power = (ch1_power + ch2_power) / 2

        # Normalize to 0-1 range (using adaptive normalization)
        # In a real system, you'd calibrate this per user
        self._update_normalization(avg_power)
        engagement = self._normalize_power(avg_power)

        # Add to buffer for smoothing
        self.engagement_buffer.append(engagement)

        # Compute smoothed engagement
        if len(self.engagement_buffer) > 0:
            self.current_engagement = np.mean(self.engagement_buffer)

        return self.current_engagement

    def _update_normalization(self, power: float):
        """Update normalization parameters."""
        if not hasattr(self, 'power_min'):
            self.power_min = power
            self.power_max = power * 2  # Initial guess
        else:
            # Slowly adapt to observed range
            alpha = 0.01
            self.power_min = (1 - alpha) * self.power_min + alpha * min(power, self.power_min)
            self.power_max = (1 - alpha) * self.power_max + alpha * max(power, self.power_max)

    def _normalize_power(self, power: float) -> float:
        """Normalize power to 0-1 range."""
        if not hasattr(self, 'power_min') or self.power_max == self.power_min:
            return 0.5

        normalized = (power - self.power_min) / (self.power_max - self.power_min)
        return np.clip(normalized, 0.0, 1.0)

    def get_segment_engagement(self, start_time: float, end_time: float,
                               eeg_manager) -> float:
        """Get average engagement for a time segment."""
        # This is a simplified version - in production you'd track timestamps
        duration = end_time - start_time
        ch1_data, ch2_data = eeg_manager.get_recent_data(duration)

        if len(ch1_data) > 0:
            return self.process_chunk(ch1_data, ch2_data)
        else:
            return self.current_engagement