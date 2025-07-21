"""Turn-based EEG engagement scorer with reliable frequency analysis."""

import numpy as np
from scipy import signal
from scipy.signal import hilbert
from collections import deque
from typing import Tuple, Dict, List, Optional
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
    """Turn-based engagement scorer that processes entire TTS windows."""

    def __init__(self,
                 baseline_duration_sec: float = 60.0,
                 adaptation_rate: float = 0.1):

        self.bands = FrequencyBands()
        self.baseline_duration_sec = baseline_duration_sec
        self.adaptation_rate = adaptation_rate

        # Create bandpass filters
        self.filters = {}
        self._create_filters()

        # Adaptive baseline tracking
        self.baseline_powers = {
            band: deque(maxlen=100) for band in self._get_band_names()
        }
        self.ema_baseline_stats = {}
        self.baseline_initialized = False

        # Turn tracking
        self.current_turn_data = []
        self.turn_active = False
        self.turn_start_time = 0

        # Session tracking
        self.session_start_time = time.time()
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

        # 1. Calculate reliable band powers from full window
        band_powers = self._calculate_window_band_powers(ch1_data, ch2_data)

        # 2. Update adaptive baseline
        self._update_adaptive_baseline(band_powers)

        # 3. Calculate window-level engagement
        window_engagement = self._calculate_window_engagement(band_powers)

        # 4. Calculate signal momentum (envelope slope)
        signal_momentum = self._calculate_signal_momentum(ch1_data, ch2_data, tts_duration)

        # 5. Calculate sub-window temporal dynamics
        sub_window_trend = self._calculate_sub_window_trend(ch1_data, ch2_data, tts_duration)

        # 6. Calculate consistency score
        consistency_score = self._calculate_consistency_score(ch1_data, ch2_data)

        # 7. Combine into single scalar
        final_engagement = (
                0.4 * window_engagement +  # Primary: overall engagement level
                0.3 * signal_momentum +  # Secondary: signal momentum
                0.2 * sub_window_trend +  # Tertiary: within-window trend
                0.1 * consistency_score  # Quaternary: consistency
        )

        return np.clip(final_engagement, 0.0, 1.0)

    def _calculate_window_band_powers(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> Dict[str, float]:
        """Calculate band powers from entire window (reliable estimates)."""
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

    def _update_adaptive_baseline(self, current_powers: Dict[str, float]):
        """Update adaptive baseline with current powers."""
        session_duration = time.time() - self.session_start_time

        # Add to rolling baseline
        for band, power in current_powers.items():
            # Log transform for better normalization
            log_power = np.log10(max(power, 1e-12))
            self.baseline_powers[band].append(log_power)

        # Initialize EMA baseline after sufficient data
        if not self.baseline_initialized and session_duration > self.baseline_duration_sec:
            for band in current_powers.keys():
                if len(self.baseline_powers[band]) > 10:
                    data = list(self.baseline_powers[band])
                    self.ema_baseline_stats[band] = {
                        'mean': np.mean(data),
                        'std': max(np.std(data), 0.01)
                    }
            self.baseline_initialized = True
            print("Turn-based adaptive baseline initialized")

        # Update EMA baseline
        elif self.baseline_initialized:
            for band, power in current_powers.items():
                if band in self.ema_baseline_stats:
                    log_power = np.log10(max(power, 1e-12))

                    old_mean = self.ema_baseline_stats[band]['mean']
                    new_mean = (1 - self.adaptation_rate) * old_mean + self.adaptation_rate * log_power

                    # Update variance too
                    old_std = self.ema_baseline_stats[band]['std']
                    delta = log_power - old_mean
                    delta2 = log_power - new_mean
                    new_var = (1 - self.adaptation_rate) * (old_std ** 2) + self.adaptation_rate * delta * delta2

                    self.ema_baseline_stats[band] = {
                        'mean': new_mean,
                        'std': max(np.sqrt(abs(new_var)), 0.01)
                    }

    def _calculate_window_engagement(self, band_powers: Dict[str, float]) -> float:
        """Calculate engagement index from band powers."""
        # Normalize powers against baseline
        normalized_powers = {}

        for band, power in band_powers.items():
            log_power = np.log10(max(power, 1e-12))

            if self.baseline_initialized and band in self.ema_baseline_stats:
                mean = self.ema_baseline_stats[band]['mean']
                std = self.ema_baseline_stats[band]['std']
                normalized_powers[band] = (log_power - mean) / std
            else:
                # Use recent baseline if available
                if len(self.baseline_powers[band]) > 5:
                    recent_data = list(self.baseline_powers[band])[-20:]
                    mean = np.mean(recent_data)
                    std = max(np.std(recent_data), 0.01)
                    normalized_powers[band] = (log_power - mean) / std
                else:
                    normalized_powers[band] = 0.0

        # Calculate engagement using beta/(alpha + theta) ratio
        if 'beta' in normalized_powers and 'alpha' in normalized_powers and 'theta' in normalized_powers:
            beta = normalized_powers['beta']
            alpha = normalized_powers['alpha']
            theta = normalized_powers['theta']

            denominator = alpha + theta
            if abs(denominator) > 0.1:
                engagement_raw = beta / denominator
            else:
                engagement_raw = beta

            # Normalize to [0,1] using tanh
            engagement = (np.tanh(engagement_raw) + 1) / 2
        else:
            engagement = 0.5

        return np.clip(engagement, 0.0, 1.0)

    def _calculate_signal_momentum(self, ch1_data: np.ndarray, ch2_data: np.ndarray,
                                   tts_duration: float) -> float:
        """Calculate signal momentum using envelope slope."""
        try:
            # Calculate continuous engagement signal
            signal_length = len(ch1_data)
            window_size = max(signal_length // 20, 50)  # 5% of signal or min 50 samples

            engagement_signal = []
            for i in range(0, signal_length - window_size, window_size // 2):
                window_ch1 = ch1_data[i:i + window_size]
                window_ch2 = ch2_data[i:i + window_size]

                # Quick engagement calculation for this mini-window
                if 'beta' in self.filters and 'alpha' in self.filters:
                    beta_filt = signal.sosfilt(self.filters['beta'], (window_ch1 + window_ch2) / 2)
                    alpha_filt = signal.sosfilt(self.filters['alpha'], (window_ch1 + window_ch2) / 2)

                    beta_power = np.mean(beta_filt ** 2)
                    alpha_power = np.mean(alpha_filt ** 2)

                    if alpha_power > 1e-12:
                        mini_engagement = beta_power / alpha_power
                    else:
                        mini_engagement = beta_power

                    engagement_signal.append(mini_engagement)

            if len(engagement_signal) < 3:
                return 0.5

            engagement_signal = np.array(engagement_signal)

            # Calculate envelope using moving average
            smoothing_window = max(len(engagement_signal) // 5, 1)
            envelope = np.convolve(engagement_signal,
                                   np.ones(smoothing_window) / smoothing_window,
                                   mode='same')

            # Calculate slope of envelope
            time_axis = np.linspace(0, tts_duration, len(envelope))
            slope, _ = np.polyfit(time_axis, envelope, 1)

            # Normalize slope to [0,1]
            momentum = np.tanh(slope * 5) * 0.5 + 0.5  # Sensitivity factor = 5

            return np.clip(momentum, 0.0, 1.0)

        except Exception as e:
            print(f"Error calculating signal momentum: {e}")
            return 0.5

    def _calculate_sub_window_trend(self, ch1_data: np.ndarray, ch2_data: np.ndarray,
                                    tts_duration: float) -> float:
        """Calculate trend across sub-windows."""
        try:
            # Divide into 4 sub-windows
            data_length = len(ch1_data)
            sub_window_size = data_length // 4

            if sub_window_size < 50:  # Need minimum samples
                return 0.5

            sub_engagements = []
            for i in range(4):
                start_idx = i * sub_window_size
                end_idx = start_idx + sub_window_size

                sub_ch1 = ch1_data[start_idx:end_idx]
                sub_ch2 = ch2_data[start_idx:end_idx]

                # Calculate mini band powers
                sub_powers = self._calculate_window_band_powers(sub_ch1, sub_ch2)
                sub_engagement = self._calculate_window_engagement(sub_powers)
                sub_engagements.append(sub_engagement)

            # Calculate trend (slope across sub-windows)
            time_points = np.arange(4)
            slope, _ = np.polyfit(time_points, sub_engagements, 1)

            # Normalize to [0,1]
            trend = np.tanh(slope * 10) * 0.5 + 0.5

            return np.clip(trend, 0.0, 1.0)

        except Exception as e:
            print(f"Error calculating sub-window trend: {e}")
            return 0.5

    def _calculate_consistency_score(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> float:
        """Calculate consistency of engagement across window."""
        try:
            # Calculate engagement for multiple mini-windows
            data_length = len(ch1_data)
            mini_window_size = max(data_length // 10, 100)  # 10 mini-windows

            mini_engagements = []
            for i in range(0, data_length - mini_window_size, mini_window_size):
                mini_ch1 = ch1_data[i:i + mini_window_size]
                mini_ch2 = ch2_data[i:i + mini_window_size]

                mini_powers = self._calculate_window_band_powers(mini_ch1, mini_ch2)
                mini_engagement = self._calculate_window_engagement(mini_powers)
                mini_engagements.append(mini_engagement)

            if len(mini_engagements) < 3:
                return 0.5

            # Calculate percentage above baseline (0.5)
            above_baseline = np.mean(np.array(mini_engagements) > 0.5)

            return np.clip(above_baseline, 0.0, 1.0)

        except Exception as e:
            print(f"Error calculating consistency: {e}")
            return 0.5

    def get_current_engagement(self) -> float:
        """Get current engagement value."""
        return self.current_engagement