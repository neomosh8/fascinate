"""Enhanced EEG engagement scorer with proper band power normalization."""

import numpy as np
from scipy import signal
from collections import deque
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

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
    """Enhanced engagement scorer with proper band power normalization."""

    def __init__(self, baseline_duration_sec: float = 60.0):
        self.bands = FrequencyBands()
        self.baseline_duration_sec = baseline_duration_sec
        self.baseline_samples = int(baseline_duration_sec * EEG_SAMPLE_RATE)

        # Create bandpass filters for each frequency band
        self.filters = {}
        self.filter_states_ch1 = {}
        self.filter_states_ch2 = {}
        self._create_filters()

        # Baseline statistics for normalization
        self.baseline_data_ch1 = deque(maxlen=self.baseline_samples)
        self.baseline_data_ch2 = deque(maxlen=self.baseline_samples)
        self.baseline_stats = {}
        self.baseline_collected = False

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

    def _apply_best_practices_normalization(self,
                                          powers_ch1: Dict[str, float],
                                          powers_ch2: Dict[str, float]) -> Dict[str, float]:
        """Apply best practices for EEG band power normalization."""

        # Step 1: Average across channels
        avg_powers = {}
        for band in powers_ch1.keys():
            avg_powers[band] = (powers_ch1[band] + powers_ch2[band]) / 2

        # Step 2: Log transformation (reduces 1/f bias and skewness)
        log_powers = {}
        for band, power in avg_powers.items():
            # Add small epsilon to avoid log(0)
            log_powers[band] = np.log10(max(power, 1e-12))

        # Step 3: Relative power normalization
        total_log_power = sum(log_powers.values())
        relative_powers = {}
        for band, log_power in log_powers.items():
            relative_powers[band] = log_power / total_log_power if total_log_power > 0 else 0.0

        # Step 4: Z-score normalization using baseline (if available)
        normalized_powers = {}
        if self.baseline_collected and self.baseline_stats:
            for band in relative_powers.keys():
                if band in self.baseline_stats:
                    mean = self.baseline_stats[band]['mean']
                    std = self.baseline_stats[band]['std']
                    if std > 0:
                        z_score = (relative_powers[band] - mean) / std
                        # Clip extreme values
                        normalized_powers[band] = np.clip(z_score, -3.0, 3.0)
                    else:
                        normalized_powers[band] = 0.0
                else:
                    normalized_powers[band] = relative_powers[band]
        else:
            normalized_powers = relative_powers.copy()

        return {
            'raw_powers': avg_powers,
            'log_powers': log_powers,
            'relative_powers': relative_powers,
            'normalized_powers': normalized_powers
        }

    def _calculate_baseline_statistics(self):
        """Calculate baseline statistics for z-score normalization."""
        if len(self.baseline_data_ch1) < self.baseline_samples * 0.8:
            return  # Need at least 80% of baseline data

        # Process baseline data through the same pipeline
        baseline_powers = {band: [] for band in self._get_band_names()}

        # Process baseline data in chunks
        chunk_size = int(EEG_SAMPLE_RATE)  # 1 second chunks
        ch1_data = np.array(list(self.baseline_data_ch1))
        ch2_data = np.array(list(self.baseline_data_ch2))

        for i in range(0, len(ch1_data) - chunk_size, chunk_size // 2):
            chunk_ch1 = ch1_data[i:i + chunk_size]
            chunk_ch2 = ch2_data[i:i + chunk_size]

            # Calculate powers for this chunk
            powers_ch1, powers_ch2 = self._calculate_band_powers(chunk_ch1, chunk_ch2)

            # Apply log transformation and relative power
            result = self._apply_best_practices_normalization(powers_ch1, powers_ch2)

            # Store relative powers for baseline stats
            for band in baseline_powers.keys():
                if band in result['relative_powers']:
                    baseline_powers[band].append(result['relative_powers'][band])

        # Calculate baseline statistics
        self.baseline_stats = {}
        for band, powers in baseline_powers.items():
            if len(powers) > 0:
                self.baseline_stats[band] = {
                    'mean': np.mean(powers),
                    'std': np.std(powers),
                    'median': np.median(powers),
                    'q25': np.percentile(powers, 25),
                    'q75': np.percentile(powers, 75)
                }

        self.baseline_collected = True
        print(f"Baseline statistics calculated from {len(ch1_data)} samples")

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
        """Process EEG chunk with enhanced normalization."""
        if len(ch1_data) == 0 or len(ch2_data) == 0:
            return self.current_engagement

        # Initialize filter states on first run
        if not hasattr(self, 'filters_initialized'):
            for band_name in self.filters.keys():
                self.filter_states_ch1[band_name] *= ch1_data[0]
                self.filter_states_ch2[band_name] *= ch2_data[0]
            self.filters_initialized = True

        # Collect baseline data if needed
        if not self.baseline_collected:
            self.baseline_data_ch1.extend(ch1_data)
            self.baseline_data_ch2.extend(ch2_data)

            if len(self.baseline_data_ch1) >= self.baseline_samples:
                self._calculate_baseline_statistics()

        # Calculate band powers
        powers_ch1, powers_ch2 = self._calculate_band_powers(ch1_data, ch2_data)

        # Apply normalization best practices
        result = self._apply_best_practices_normalization(powers_ch1, powers_ch2)

        # Store in history
        import time
        timestamp = time.time()
        self.power_history['timestamp'].append(timestamp)

        for band in self._get_band_names():
            if band in result['raw_powers']:
                self.power_history['ch1_powers'][band].append(powers_ch1[band])
                self.power_history['ch2_powers'][band].append(powers_ch2[band])
                self.power_history['relative_powers'][band].append(result['relative_powers'][band])
                self.power_history['normalized_powers'][band].append(result['normalized_powers'][band])

        # Calculate engagement score
        # Use normalized beta/theta ratio as engagement metric
        normalized_powers = result['normalized_powers']

        if 'beta' in normalized_powers and 'theta' in normalized_powers:
            # Beta/theta ratio is commonly used for attention/engagement
            theta_power = normalized_powers['theta']
            beta_power = normalized_powers['beta']

            # Calculate ratio with safeguards
            if theta_power != 0:
                engagement_ratio = beta_power / (theta_power + 1e-6)  # Add small epsilon
            else:
                engagement_ratio = beta_power

            # Normalize to 0-1 range using tanh transformation
            engagement = (np.tanh(engagement_ratio) + 1) / 2
        else:
            # Fallback to beta power alone
            engagement = (normalized_powers.get('beta', 0) + 3) / 6  # Map [-3,3] to [0,1]

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
            'baseline_stats': self.baseline_stats,
            'current_powers': {},
            'band_ratios': {}
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
                    max(normalized['theta']['normalized'], 1e-6)
                )

            if 'alpha' in normalized and 'theta' in normalized:
                analysis['band_ratios']['alpha_theta'] = (
                    normalized['alpha']['normalized'] /
                    max(normalized['theta']['normalized'], 1e-6)
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