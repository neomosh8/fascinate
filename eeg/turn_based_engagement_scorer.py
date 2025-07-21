"""Turn-based EEG engagement scorer with raw beta/(alpha+theta)."""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, List
from dataclasses import dataclass
import time

from config import EEG_SAMPLE_RATE


@dataclass
class FrequencyBands:
    """Standard EEG frequency bands."""
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)


class TurnBasedEngagementScorer:
    """Turn-based engagement scorer with raw beta/(alpha+theta)."""

    def __init__(self,
                 baseline_duration_sec: float = 0.0,  # Not used anymore
                 adaptation_rate: float = 0.1):       # Not used anymore

        self.bands = FrequencyBands()

        # Create bandpass filters
        self.filters = {}
        self._create_filters()

        # Turn tracking
        self.current_turn_data = []
        self.turn_active = False
        self.turn_count = 0

        # Current engagement
        self.current_engagement = 0.5

    def _get_band_names(self) -> List[str]:
        """Get list of frequency band names."""
        return ['delta', 'theta', 'alpha', 'beta', 'gamma']

    def _create_filters(self):
        """Create bandpass filters for each frequency band."""
        nyq = EEG_SAMPLE_RATE / 2

        for band_name in self._get_band_names():
            band_range = getattr(self.bands, band_name)
            low_freq, high_freq = band_range

            # Ensure frequencies are within Nyquist limit
            high_freq = min(high_freq, nyq * 0.95)

            if low_freq < nyq and high_freq > low_freq:
                sos = signal.butter(
                    4, [low_freq / nyq, high_freq / nyq],
                    btype='band', output='sos'
                )
                self.filters[band_name] = sos

    def start_turn(self):
        """Start tracking EEG data for a new turn."""
        self.current_turn_data = []
        self.turn_active = True

    def add_eeg_chunk(self, ch1_samples: List[float], ch2_samples: List[float]):
        """Add EEG chunk to current turn data."""
        if self.turn_active:
            for ch1, ch2 in zip(ch1_samples, ch2_samples):
                self.current_turn_data.append((ch1, ch2))

    def end_turn(self, tts_duration: float) -> float:
        """
        End turn and calculate raw engagement.

        Returns:
            Raw engagement value beta/(alpha+theta)
        """
        if not self.turn_active or len(self.current_turn_data) < 100:
            self.turn_active = False
            return self.current_engagement

        try:
            # Convert to numpy arrays
            eeg_data = np.array(self.current_turn_data)
            ch1_data = eeg_data[:, 0]
            ch2_data = eeg_data[:, 1]

            # Calculate raw engagement
            engagement = self._calculate_raw_engagement(ch1_data, ch2_data)

            self.current_engagement = engagement
            self.turn_count += 1
            self.turn_active = False

            return engagement

        except Exception as e:
            print(f"Error calculating engagement: {e}")
            self.turn_active = False
            return self.current_engagement

    def _calculate_raw_engagement(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> float:
        """Calculate raw beta/(alpha+theta) engagement."""

        # Calculate band powers
        band_powers = self._calculate_band_powers(ch1_data, ch2_data)

        # Raw engagement = beta / (alpha + theta)
        beta = band_powers.get('beta', 0)
        alpha = band_powers.get('alpha', 0)
        theta = band_powers.get('theta', 0)

        denominator = alpha + theta

        if denominator > 0:
            engagement = beta / denominator
        else:
            engagement = 0.0

        print(f"Raw engagement: beta={beta:.2e}, alpha={alpha:.2e}, theta={theta:.2e}, ratio={engagement:.4f}")

        return max(0.0, engagement)  # Just prevent negative values

    def _calculate_band_powers(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> Dict[str, float]:
        """Calculate band powers from EEG data."""
        band_powers = {}

        for band_name, sos in self.filters.items():
            # Filter the data
            ch1_filtered = signal.sosfilt(sos, ch1_data)
            ch2_filtered = signal.sosfilt(sos, ch2_data)

            # Calculate power (mean squared amplitude)
            ch1_power = np.mean(ch1_filtered ** 2)
            ch2_power = np.mean(ch2_filtered ** 2)

            # Average across channels
            band_powers[band_name] = (ch1_power + ch2_power) / 2

        return band_powers

    def get_current_engagement(self) -> float:
        """Get current engagement value."""
        return self.current_engagement

    # Legacy compatibility properties
    @property
    def baseline_initialized(self) -> bool:
        """For compatibility - always true now."""
        return True

    @property
    def baseline_collected(self) -> bool:
        """For compatibility - always true now."""
        return True