"""Turn-based EEG engagement scorer with simple 4-minute rolling baseline."""

import numpy as np
from scipy import signal
from collections import deque
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
    """Turn-based engagement scorer with simple 4-minute rolling baseline."""

    def __init__(self,
                 baseline_duration_sec: float = 240.0,  # 4 minutes
                 adaptation_rate: float = 0.1):

        self.bands = FrequencyBands()

        # Create bandpass filters
        self.filters = {}
        self._create_filters()

        # Simple 4-minute rolling baseline
        # Store the last 4 minutes worth of band powers
        max_turns = 50  # ~4 minutes at ~5 seconds per turn
        self.baseline_powers = {
            band: deque(maxlen=max_turns) for band in self._get_band_names()
        }

        # Turn tracking
        self.current_turn_data = []
        self.turn_active = False
        self.turn_start_time = 0
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
        self.turn_start_time = time.time()

    def add_eeg_chunk(self, ch1_samples: List[float], ch2_samples: List[float]):
        """Add EEG chunk to current turn data."""
        if self.turn_active:
            # Store as pairs of (ch1, ch2) samples
            for ch1, ch2 in zip(ch1_samples, ch2_samples):
                self.current_turn_data.append((ch1, ch2))

    def end_turn(self, tts_duration: float) -> float:
        """
        End turn and calculate single engagement value.

        Returns:
            Single scalar engagement value [0.0, 1.0]
        """
        if not self.turn_active or len(self.current_turn_data) < 100:
            self.turn_active = False
            return self.current_engagement

        try:
            # Convert to numpy arrays
            eeg_data = np.array(self.current_turn_data)
            ch1_data = eeg_data[:, 0]
            ch2_data = eeg_data[:, 1]

            # Calculate engagement for this turn
            engagement = self._calculate_turn_engagement(ch1_data, ch2_data, tts_duration)

            self.current_engagement = engagement
            self.turn_count += 1
            self.turn_active = False

            return engagement

        except Exception as e:
            print(f"Error calculating turn engagement: {e}")
            self.turn_active = False
            return self.current_engagement

    def _calculate_turn_engagement(self, ch1_data: np.ndarray, ch2_data: np.ndarray,
                                   tts_duration: float) -> float:
        """Calculate engagement for entire turn window."""

        # 1. Calculate band powers from full window
        band_powers = self._calculate_window_band_powers(ch1_data, ch2_data)

        # 2. Add to rolling baseline (always)
        for band, power in band_powers.items():
            log_power = np.log10(max(power, 1e-12))
            self.baseline_powers[band].append(log_power)

        # 3. Calculate engagement using rolling baseline
        engagement = self._calculate_window_engagement(band_powers)

        return np.clip(engagement, 0.0, 1.0)

    def _calculate_window_band_powers(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> Dict[str, float]:
        """Calculate band powers from entire window."""
        band_powers = {}

        for band_name, sos in self.filters.items():
            # Filter entire window
            ch1_filtered = signal.sosfilt(sos, ch1_data)
            ch2_filtered = signal.sosfilt(sos, ch2_data)

            # Calculate power (mean squared amplitude)
            ch1_power = np.mean(ch1_filtered ** 2)
            ch2_power = np.mean(ch2_filtered ** 2)

            # Average across channels
            band_powers[band_name] = (ch1_power + ch2_power) / 2

        return band_powers

    def _calculate_window_engagement(self, band_powers: Dict[str, float]) -> float:
        """Calculate engagement using simple rolling baseline."""

        # Normalize against 4-minute rolling baseline
        normalized_powers = {}

        for band, power in band_powers.items():
            log_power = np.log10(max(power, 1e-12))

            baseline_data = list(self.baseline_powers[band])

            if len(baseline_data) >= 3:  # Need at least 3 samples
                # Simple rolling baseline
                mean = np.mean(baseline_data)
                std = max(np.std(baseline_data), 0.01)  # Prevent division by zero
                normalized_powers[band] = (log_power - mean) / std
            else:
                # Not enough data yet - return neutral
                normalized_powers[band] = 0.0

        # Calculate engagement using beta/(alpha + theta) ratio
        if all(band in normalized_powers for band in ['beta', 'alpha', 'theta']):
            beta = normalized_powers['beta']
            alpha = normalized_powers['alpha']
            theta = normalized_powers['theta']

            denominator = alpha + theta
            if abs(denominator) > 0.1:
                engagement_raw = beta / denominator
            else:
                engagement_raw = beta

            # Normalize to [0,1] using tanh (prevents extreme values)
            engagement = engagement_raw
        else:
            engagement = 0.5

        return engagement

    def get_current_engagement(self) -> float:
        """Get current engagement value."""
        return self.current_engagement

    # Legacy compatibility properties
    @property
    def baseline_initialized(self) -> bool:
        """For compatibility - true if we have enough data."""
        return any(len(self.baseline_powers[band]) >= 3 for band in self._get_band_names())

    @property
    def baseline_collected(self) -> bool:
        """For compatibility."""
        return self.baseline_initialized