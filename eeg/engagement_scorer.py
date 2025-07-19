"""Enhanced EEG engagement scorer with research-validated metrics."""

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
    """Enhanced engagement scorer with research-validated multi-metric approach."""

    def __init__(self,
                 baseline_duration_sec: float = 60.0,
                 rolling_window_sec: float = 300.0,
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
            band: deque(maxlen=int(rolling_window_sec * 10))
            for band in self._get_band_names()
        }

        # EMA baseline stats
        self.ema_baseline_stats = {}
        self.baseline_initialized = False

        # Legacy baseline for compatibility
        self.baseline_data_ch1 = deque(maxlen=self.baseline_samples)
        self.baseline_data_ch2 = deque(maxlen=self.baseline_samples)
        self.baseline_stats = {}
        self.baseline_collected = False

        # Multi-metric engagement tracking
        self.engagement_buffer = deque(maxlen=int(ENGAGEMENT_WINDOW_SEC * 10))
        self.attention_buffer = deque(maxlen=int(ENGAGEMENT_WINDOW_SEC * 10))
        self.memory_buffer = deque(maxlen=int(ENGAGEMENT_WINDOW_SEC * 10))
        self.motivation_buffer = deque(maxlen=int(ENGAGEMENT_WINDOW_SEC * 10))

        self.current_engagement = 0.5
        self.current_attention = 0.5
        self.current_memory = 0.5
        self.current_motivation = 0.5

        # Power history for analysis
        self.power_history = {
            'timestamp': deque(maxlen=1000),
            'ch1_powers': {band: deque(maxlen=1000) for band in self._get_band_names()},
            'ch2_powers': {band: deque(maxlen=1000) for band in self._get_band_names()},
            'relative_powers': {band: deque(maxlen=1000) for band in self._get_band_names()},
            'normalized_powers': {band: deque(maxlen=1000) for band in self._get_band_names()},
            'engagement_index': deque(maxlen=1000),
            'attention_index': deque(maxlen=1000),
            'memory_index': deque(maxlen=1000),
            'motivation_index': deque(maxlen=1000)
        }

        # Session tracking
        self.session_start_time = time.time()

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
                if len(self.rolling_baseline_powers[band]) > 50:
                    initial_data = list(self.rolling_baseline_powers[band])[-600:]
                    self.ema_baseline_stats[band] = {
                        'mean': np.mean(initial_data),
                        'std': max(np.std(initial_data), 0.01)
                    }
            self.baseline_initialized = True
            print("Adaptive baseline initialized")

        # Update EMA baseline continuously
        elif self.baseline_initialized:
            for band, power in current_powers.items():
                if band in self.ema_baseline_stats:
                    old_mean = self.ema_baseline_stats[band]['mean']
                    new_mean = (1 - self.adaptation_rate) * old_mean + self.adaptation_rate * power

                    old_std = self.ema_baseline_stats[band]['std']
                    delta = power - old_mean
                    delta2 = power - new_mean
                    new_var = (1 - self.adaptation_rate) * (old_std**2) + self.adaptation_rate * delta * delta2

                    self.ema_baseline_stats[band] = {
                        'mean': new_mean,
                        'std': max(np.sqrt(abs(new_var)), 0.01)
                    }

    def _get_context_aware_normalization(self, current_powers: Dict[str, float]) -> Dict[str, float]:
        """Apply multi-timescale adaptive normalization."""
        normalized_powers = {}

        for band, power in current_powers.items():
            scores = []
            weights_sum = 0

            # Short-term rolling baseline (recent context - high weight)
            if len(self.rolling_baseline_powers[band]) > 30:
                recent_data = list(self.rolling_baseline_powers[band])[-300:]
                if len(recent_data) > 10:
                    rolling_mean = np.mean(recent_data)
                    rolling_std = max(np.std(recent_data), 0.01)
                    rolling_score = (power - rolling_mean) / rolling_std
                    scores.append(rolling_score * 0.5)
                    weights_sum += 0.5

            # EMA adaptive baseline (session evolution - medium weight)
            if self.baseline_initialized and band in self.ema_baseline_stats:
                ema_mean = self.ema_baseline_stats[band]['mean']
                ema_std = self.ema_baseline_stats[band]['std']
                ema_score = (power - ema_mean) / ema_std
                scores.append(ema_score * 0.3)
                weights_sum += 0.3

            # Long-term session baseline (overall context - low weight)
            if len(self.rolling_baseline_powers[band]) > 100:
                session_data = list(self.rolling_baseline_powers[band])
                session_mean = np.mean(session_data)
                session_std = max(np.std(session_data), 0.01)
                session_score = (power - session_mean) / session_std
                scores.append(session_score * 0.2)
                weights_sum += 0.2

            # Combine scores or fallback
            if scores and weights_sum > 0:
                combined_score = sum(scores) / weights_sum
                normalized_powers[band] = np.clip(combined_score, -4.0, 4.0)
            else:
                normalized_powers[band] = 0.0

        return normalized_powers

    def _calculate_research_validated_metrics(self, normalized_powers: Dict[str, float],
                                            ch1_powers: Dict[str, float], ch2_powers: Dict[str, float]) -> Dict[str, float]:
        """Calculate research-validated engagement metrics."""
        metrics = {}

        # Required bands
        required_bands = ['theta', 'alpha', 'beta', 'gamma']
        if not all(band in normalized_powers for band in required_bands):
            return {
                'engagement_index': 0.5,
                'attention_index': 0.5,
                'memory_index': 0.5,
                'motivation_index': 0.5
            }

        theta = normalized_powers['theta']
        alpha = normalized_powers['alpha']
        beta = normalized_powers['beta']
        gamma = normalized_powers['gamma']

        # 1. Engagement Index: beta/(alpha + theta) - GOLD STANDARD
        denominator = alpha + theta
        if abs(denominator) > 0.1:
            engagement_raw = beta / denominator
        else:
            engagement_raw = beta  # Fallback when denominator near zero

        # Normalize to [0,1] range
        engagement_index = (np.tanh(engagement_raw) + 1) / 2

        # 2. Attention Index: Alpha decreases + Theta increases during task
        # High attention = low alpha + appropriate theta
        attention_raw = theta - alpha  # Task-related theta boost minus alpha suppression
        attention_index = (np.tanh(attention_raw) + 1) / 2

        # 3. Memory Index: Theta + Gamma power (working memory)
        memory_raw = (theta + gamma) / 2
        memory_index = (np.tanh(memory_raw) + 1) / 2

        # 4. Motivation Index: Left-right alpha/beta asymmetry
        # Simplified version using channel differences as proxy
        if 'alpha' in ch1_powers and 'alpha' in ch2_powers:
            left_alpha = ch1_powers['alpha']  # Assuming ch1 is left-ish
            right_alpha = ch2_powers['alpha']  # Assuming ch2 is right-ish
            left_beta = ch1_powers['beta']
            right_beta = ch2_powers['beta']

            # Approach motivation = left alpha down, left beta up relative to right
            asymmetry = (left_beta / max(left_alpha, 1e-6)) - (right_beta / max(right_alpha, 1e-6))
            motivation_index = (np.tanh(asymmetry) + 1) / 2
        else:
            motivation_index = 0.5  # Neutral when can't calculate

        metrics = {
            'engagement_index': np.clip(engagement_index, 0, 1),
            'attention_index': np.clip(attention_index, 0, 1),
            'memory_index': np.clip(memory_index, 0, 1),
            'motivation_index': np.clip(motivation_index, 0, 1)
        }

        return metrics

    def _apply_best_practices_normalization(self,
                                          powers_ch1: Dict[str, float],
                                          powers_ch2: Dict[str, float]) -> Dict[str, float]:
        """Apply best practices for EEG band power normalization with research metrics."""

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

        # Step 6: Calculate research-validated metrics
        metrics = self._calculate_research_validated_metrics(normalized_powers, powers_ch1, powers_ch2)

        return {
            'raw_powers': avg_powers,
            'log_powers': log_powers,
            'relative_powers': relative_powers,
            'normalized_powers': normalized_powers,
            **metrics
        }

    def _calculate_baseline_statistics(self):
        """Legacy method - now handled by adaptive baseline."""
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
        """Process EEG chunk with research-validated multi-metric approach."""
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

        # Apply enhanced normalization and calculate metrics
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

        # Store metric history
        self.power_history['engagement_index'].append(result.get('engagement_index', 0.5))
        self.power_history['attention_index'].append(result.get('attention_index', 0.5))
        self.power_history['memory_index'].append(result.get('memory_index', 0.5))
        self.power_history['motivation_index'].append(result.get('motivation_index', 0.5))

        # Update metric buffers
        self.engagement_buffer.append(result.get('engagement_index', 0.5))
        self.attention_buffer.append(result.get('attention_index', 0.5))
        self.memory_buffer.append(result.get('memory_index', 0.5))
        self.motivation_buffer.append(result.get('motivation_index', 0.5))

        # Update current values (smoothed)
        if len(self.engagement_buffer) > 0:
            self.current_engagement = np.mean(self.engagement_buffer)
            self.current_attention = np.mean(self.attention_buffer)
            self.current_memory = np.mean(self.memory_buffer)
            self.current_motivation = np.mean(self.motivation_buffer)

        return self.current_engagement

    def get_power_analysis(self) -> Dict:
        """Get detailed power analysis across all bands and metrics."""
        if not self.power_history['timestamp']:
            return {}

        analysis = {
            'baseline_collected': self.baseline_collected,
            'adaptive_baseline_initialized': self.baseline_initialized,
            'baseline_stats': self.baseline_stats,
            'ema_baseline_stats': self.ema_baseline_stats,
            'current_powers': {},
            'current_metrics': {
                'engagement': self.current_engagement,
                'attention': self.current_attention,
                'memory': self.current_memory,
                'motivation': self.current_motivation
            },
            'band_ratios': {},
            'session_duration': time.time() - self.session_start_time
        }

        # Get most recent powers
        if self.power_history['normalized_powers']['beta']:
            for band in self._get_band_names():
                if self.power_history['normalized_powers'][band]:
                    analysis['current_powers'][band] = {
                        'raw': self.power_history['ch1_powers'][band][-1] if self.power_history['ch1_powers'][band] else 0,
                        'relative': self.power_history['relative_powers'][band][-1] if self.power_history['relative_powers'][band] else 0,
                        'normalized': self.power_history['normalized_powers'][band][-1] if self.power_history['normalized_powers'][band] else 0
                    }

            # Calculate research-validated ratios
            normalized = analysis['current_powers']
            if 'theta' in normalized and 'alpha' in normalized and 'beta' in normalized:
                # Primary engagement index
                alpha_theta_sum = normalized['alpha']['normalized'] + normalized['theta']['normalized']
                if abs(alpha_theta_sum) > 0.1:
                    analysis['band_ratios']['engagement_index'] = (
                        normalized['beta']['normalized'] / alpha_theta_sum
                    )

                # Secondary ratios
                analysis['band_ratios']['attention_index'] = (
                    normalized['theta']['normalized'] - normalized['alpha']['normalized']
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

    def get_composite_engagement(self) -> float:
        """Get composite engagement score combining multiple metrics."""
        # Weighted combination of metrics based on research
        weights = {
            'engagement': 0.4,   # Primary engagement index
            'attention': 0.3,    # Attention focus
            'memory': 0.2,       # Working memory
            'motivation': 0.1    # Motivational approach
        }

        composite = (
            weights['engagement'] * self.current_engagement +
            weights['attention'] * self.current_attention +
            weights['memory'] * self.current_memory +
            weights['motivation'] * self.current_motivation
        )

        return np.clip(composite, 0, 1)