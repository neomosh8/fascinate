"""Enhanced EEG engagement scorer with adaptive rolling baseline normalization."""

import numpy as np
from scipy import signal
from collections import deque
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import time

from config import EEG_SAMPLE_RATE, ENGAGEMENT_WINDOW_SEC


@dataclass
class FrequencyBands:
    """Standard EEG frequency bands."""
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)


class EngagementScorer:
    """Enhanced engagement scorer with adaptive rolling baseline normalization."""

    def __init__(self,
                 baseline_duration_sec: float = 60.0,
                 rolling_window_sec: float = 300.0,  # 5 minutes
                 adaptation_rate: float = 0.1):

        self.bands = FrequencyBands()
        self.baseline_duration_sec = baseline_duration_sec
        self.rolling_window_sec = rolling_window_sec
        self.adaptation_rate = adaptation_rate
        self.baseline_samples = int(baseline_duration_sec * EEG_SAMPLE_RATE)

        # Create bandpass filters for each frequency band
        self.filters = {}
        self.filter_states_ch1 = {}
        self.filter_states_ch2 = {}
        self._create_filters()

        # Multi-timescale baseline approach
        self.rolling_baseline_powers = {
            band: deque(maxlen=int(rolling_window_sec * 10))  # 10Hz sampling rate
            for band in self._get_band_names()
        }

        # Exponential moving average baseline stats
        self.ema_baseline_stats = {}
        self.baseline_initialized = False

        # Legacy baseline for initial period (compatibility)
        self.baseline_data_ch1 = deque(maxlen=self.baseline_samples)
        self.baseline_data_ch2 = deque(maxlen=self.baseline_samples)
        self.baseline_stats = {}
        self.baseline_collected = False  # Keep for compatibility

        # Current engagement tracking
        self.engagement_buffer = deque(maxlen=int(ENGAGEMENT_WINDOW_SEC * 10))
        self.current_engagement = 0.5

        # Store raw power history for analysis
        self.power_history = {
            'timestamp': deque(maxlen=1000),
            'ch1_powers': {band: deque(maxlen=1000) for band in self._get_band_names()},
            'ch2_powers': {band: deque(maxlen=1000) for band in self._get_band_names()},
            'relative_powers': {band: deque(maxlen=1000) for band in self._get_band_names()},
            'normalized_powers': {band: deque(maxlen=1000) for band in self._get_band_names()}
        }

        # Session tracking
        self.session_start_time = time.time()
        self.last_baseline_update = 0

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
                # Create bandpass filter
                sos = signal.butter(
                    4, [low_freq / nyq, high_freq / nyq],
                    btype='band', output='sos'
                )
                self.filters[band_name] = sos

                # Initialize filter states
                self.filter_states_ch1[band_name] = signal.sosfilt_zi(sos)
                self.filter_states_ch2[band_name] = signal.sosfilt_zi(sos)

    def _calculate_adaptive_baseline(self, current_powers: Dict[str, float]):
        """Update baseline using multiple adaptive strategies."""
        current_time = time.time()
        session_duration = current_time - self.session_start_time

        # Add current powers to rolling baseline
        for band, power in current_powers.items():
            self.rolling_baseline_powers[band].append(power)

        # Initialize EMA baseline after initial period
        if not self.baseline_initialized and session_duration > self.baseline_duration_sec:
            for band in current_powers.keys():
                if len(self.rolling_baseline_powers[band]) > 50:  # At least 5 seconds of data
                    initial_data = list(self.rolling_baseline_powers[band])[-600:]  # Last minute
                    self.ema_baseline_stats[band] = {
                        'mean': np.mean(initial_data),
                        'std': max(np.std(initial_data), 0.01)  # Prevent division by zero
                    }
            self.baseline_initialized = True
            print("Adaptive baseline initialized")

        # Update EMA baseline continuously
        elif self.baseline_initialized:
            for band, power in current_powers.items():
                if band in self.ema_baseline_stats:
                    # Exponential moving average update
                    old_mean = self.ema_baseline_stats[band]['mean']
                    new_mean = (1 - self.adaptation_rate) * old_mean + self.adaptation_rate * power

                    # Update variance using Welford's online algorithm
                    old_std = self.ema_baseline_stats[band]['std']
                    delta = power - old_mean
                    delta2 = power - new_mean
                    new_var = (1 - self.adaptation_rate) * (old_std**2) + self.adaptation_rate * delta * delta2

                    self.ema_baseline_stats[band] = {
                        'mean': new_mean,
                        'std': max(np.sqrt(abs(new_var)), 0.01)  # Prevent division by zero
                    }

    def _get_context_aware_normalization(self, current_powers: Dict[str, float]) -> Dict[str, float]:
        """Apply multi-timescale adaptive normalization."""
        normalized_powers = {}

        for band, power in current_powers.items():
            scores = []
            weights_sum = 0

            # 1. Short-term rolling baseline (recent context - high weight)
            if len(self.rolling_baseline_powers[band]) > 30:  # At least 3 seconds
                recent_data = list(self.rolling_baseline_powers[band])[-300:]  # Last 30 seconds
                if len(recent_data) > 10:
                    rolling_mean = np.mean(recent_data)
                    rolling_std = max(np.std(recent_data), 0.01)
                    rolling_score = (power - rolling_mean) / rolling_std
                    scores.append(rolling_score * 0.5)  # 50% weight for recent context
                    weights_sum += 0.5

            # 2. EMA adaptive baseline (session evolution - medium weight)
            if self.baseline_initialized and band in self.ema_baseline_stats:
                ema_mean = self.ema_baseline_stats[band]['mean']
                ema_std = self.ema_baseline_stats[band]['std']
                ema_score = (power - ema_mean) / ema_std
                scores.append(ema_score * 0.3)  # 30% weight for session adaptation
                weights_sum += 0.3

            # 3. Long-term session baseline (overall context - low weight)
            if len(self.rolling_baseline_powers[band]) > 100:
                session_data = list(self.rolling_baseline_powers[band])
                session_mean = np.mean(session_data)
                session_std = max(np.std(session_data), 0.01)
                session_score = (power - session_mean) / session_std
                scores.append(session_score * 0.2)  # 20% weight for overall context
                weights_sum += 0.2

            # Combine scores or fallback
            if scores and weights_sum > 0:
                combined_score = sum(scores) / weights_sum
                normalized_powers[band] = np.clip(combined_score, -4.0, 4.0)
            else:
                # Fallback to simple z-score if no baseline available
                normalized_powers[band] = 0.0

        return normalized_powers

    def _apply_best_practices_normalization(self,
                                          powers_ch1: Dict[str, float],
                                          powers_ch2: Dict[str, float]) -> Dict[str, float]:
        """Apply best practices for EEG band power normalization with adaptive baseline."""

        # Step 1: Average across channels
        avg_powers = {}
        for band in powers_ch1.keys():
            avg_powers[band] = (powers_ch1[band] + powers_ch2[band]) / 2

        # Step 2: Log transformation (reduces 1/f bias and skewness)
        log_powers = {}
        for band, power in avg_powers.items():
            log_powers[band] = np.log10(max(power, 1e-12))

        # Step 3: Relative power normalization
        total_log_power = sum(log_powers.values())
        relative_powers = {}
        for band, log_power in log_powers.items():
            relative_powers[band] = log_power / total_log_power if total_log_power > 0 else 0.0

        # Step 4: Update adaptive baseline
        self._calculate_adaptive_baseline(relative_powers)

        # Step 5: Apply context-aware normalization
        normalized_powers = self._get_context_aware_normalization(relative_powers)

        return {
            'raw_powers': avg_powers,
            'log_powers': log_powers,
            'relative_powers': relative_powers,
            'normalized_powers': normalized_powers
        }

    def _calculate_baseline_statistics(self):
        """Legacy method - now handled by adaptive baseline."""
        # Keep for compatibility but delegate to adaptive system
        if not self.baseline_collected and len(self.baseline_data_ch1) >= self.baseline_samples * 0.8:
            print("Legacy baseline collection completed - switching to adaptive baseline")
            self.baseline_collected = True

    def _calculate_band_powers(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate power in each frequency band for both channels."""
        powers_ch1 = {}
        powers_ch2 = {}

        for band_name, sos in self.filters.items():
            # Apply bandpass filter
            ch1_filtered, self.filter_states_ch1[band_name] = signal.sosfilt(
                sos, ch1_data, zi=self.filter_states_ch1[band_name]
            )
            ch2_filtered, self.filter_states_ch2[band_name] = signal.sosfilt(
                sos, ch2_data, zi=self.filter_states_ch2[band_name]
            )

            # Calculate power (mean squared amplitude)
            powers_ch1[band_name] = np.mean(ch1_filtered ** 2)
            powers_ch2[band_name] = np.mean(ch2_filtered ** 2)

        return powers_ch1, powers_ch2

    def process_chunk(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> float:
        """Process EEG chunk with adaptive baseline normalization."""
        if len(ch1_data) == 0 or len(ch2_data) == 0:
            return self.current_engagement

        # Initialize filter states on first run
        if not hasattr(self, 'filters_initialized'):
            for band_name in self.filters.keys():
                self.filter_states_ch1[band_name] *= ch1_data[0]
                self.filter_states_ch2[band_name] *= ch2_data[0]
            self.filters_initialized = True

        # Legacy baseline collection (for compatibility)
        if not self.baseline_collected:
            self.baseline_data_ch1.extend(ch1_data)
            self.baseline_data_ch2.extend(ch2_data)
            if len(self.baseline_data_ch1) >= self.baseline_samples:
                self._calculate_baseline_statistics()

        # Calculate band powers
        powers_ch1, powers_ch2 = self._calculate_band_powers(ch1_data, ch2_data)

        # Apply adaptive normalization
        result = self._apply_best_practices_normalization(powers_ch1, powers_ch2)

        # Store in history
        timestamp = time.time()
        self.power_history['timestamp'].append(timestamp)

        for band in self._get_band_names():
            if band in result['raw_powers']:
                self.power_history['ch1_powers'][band].append(powers_ch1[band])
                self.power_history['ch2_powers'][band].append(powers_ch2[band])
                self.power_history['relative_powers'][band].append(result['relative_powers'][band])
                self.power_history['normalized_powers'][band].append(result['normalized_powers'][band])

        # Calculate engagement score with adaptive sensitivity
        normalized_powers = result['normalized_powers']

        if 'beta' in normalized_powers and 'theta' in normalized_powers:
            theta_power = normalized_powers['theta']
            beta_power = normalized_powers['beta']

            # Adaptive sensitivity based on session progression
            session_duration = time.time() - self.session_start_time
            sensitivity_factor = min(1.0 + session_duration / 600, 2.0)  # Increase sensitivity over time

            # Context-aware engagement calculation
            engagement_ratio = (beta_power - theta_power) * sensitivity_factor

            # Apply tanh for smooth bounded output
            engagement = (np.tanh(engagement_ratio) + 1) / 2

            # Add small bonus for high beta in absolute terms
            if beta_power > 1.0:  # Well above baseline
                engagement += 0.05 * min(beta_power - 1.0, 1.0)
                engagement = min(engagement, 1.0)
        else:
            # Fallback to beta power alone
            engagement = (normalized_powers.get('beta', 0) + 4) / 8  # Map [-4,4] to [0,1]

        # Smooth engagement score
        self.engagement_buffer.append(engagement)
        if len(self.engagement_buffer) > 0:
            self.current_engagement = np.mean(self.engagement_buffer)

        return self.current_engagement

    def get_power_analysis(self) -> Dict:
        """Get detailed power analysis across all bands."""
        if not self.power_history['timestamp']:
            return {}

        analysis = {
            'baseline_collected': self.baseline_collected,
            'adaptive_baseline_initialized': self.baseline_initialized,
            'baseline_stats': self.baseline_stats,
            'ema_baseline_stats': self.ema_baseline_stats,
            'current_powers': {},
            'band_ratios': {},
            'session_duration': time.time() - self.session_start_time
        }

        # Get most recent normalized powers
        if self.power_history['normalized_powers']['beta']:
            for band in self._get_band_names():
                if self.power_history['normalized_powers'][band]:
                    analysis['current_powers'][band] = {
                        'raw': self.power_history['ch1_powers'][band][-1] if self.power_history['ch1_powers'][band] else 0,
                        'relative': self.power_history['relative_powers'][band][-1] if self.power_history['relative_powers'][band] else 0,
                        'normalized': self.power_history['normalized_powers'][band][-1] if self.power_history['normalized_powers'][band] else 0
                    }

            # Calculate commonly used ratios
            normalized = analysis['current_powers']
            if 'theta' in normalized and 'beta' in normalized:
                analysis['band_ratios']['beta_theta'] = (
                    normalized['beta']['normalized'] /
                    max(abs(normalized['theta']['normalized']), 0.1)
                )

            if 'alpha' in normalized and 'theta' in normalized:
                analysis['band_ratios']['alpha_theta'] = (
                    normalized['alpha']['normalized'] /
                    max(abs(normalized['theta']['normalized']), 0.1)
                )

        return analysis

    def get_segment_engagement(self, start_time: float, end_time: float, eeg_manager) -> float:
        """Get average engagement for a time segment."""
        duration = end_time - start_time
        ch1_data, ch2_data = eeg_manager.get_recent_data(duration)

        if len(ch1_data) > 0:
            return self.process_chunk(ch1_data, ch2_data)
        else:
            return self.current_engagement