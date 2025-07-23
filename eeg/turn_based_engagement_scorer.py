import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import EEG_SAMPLE_RATE
from eeg.online_filter import OnlineFilter


@dataclass
class FrequencyBands:
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)


class TurnBasedEngagementScorer:
    """
    Engagement scorer using PROVEN filtering pipeline.
    Now processes clean, filtered data instead of corrupted raw signals.
    """

    def __init__(self,
                 baseline_duration_sec: float = 30,
                 baseline_update_rate: float = 0.001,
                 smoothing: float = 0.0):

        self.bands = FrequencyBands()

        # Use the PROVEN filter instead of custom band filters
        self.online_filter = OnlineFilter(EEG_SAMPLE_RATE)

        self.smoothing = smoothing

        # Baseline collection parameters
        self.baseline_duration_sec = baseline_duration_sec
        self.baseline_update_rate = float(np.clip(baseline_update_rate, 0, 1))
        self._samples_needed = int(baseline_duration_sec * EEG_SAMPLE_RATE)

        # Baseline data collection (now using FILTERED data)
        self.baseline_chunks = {'alpha': [], 'beta': [], 'theta': [], 'delta': [], 'gamma': []}
        self.baseline_stats = {}
        self.baseline_ready = False

        # Turn state
        self.turn_data: List[Tuple[float, float]] = []
        self.turn_active = False
        self.turn_idx = 0
        self.current_engagement = 0.5

        # Buffer for collecting baseline during initial period (FILTERED data)
        self._baseline_buffer_ch1: List[float] = []
        self._baseline_buffer_ch2: List[float] = []

    def add_eeg_chunk(self, ch1: List[float], ch2: List[float]):
        """Add EEG data chunk with PROPER filtering applied first."""

        # Convert to numpy arrays
        ch1_array = np.array(ch1)
        ch2_array = np.array(ch2)

        # Apply the PROVEN filter pipeline that worked in neocore_client.py
        ch1_filtered, ch2_filtered = self.online_filter.filter_chunk(ch1_array, ch2_array)

        # Debug: Show the filtering effect
        if len(ch1) > 0:
            raw_range_ch1 = max(ch1) - min(ch1)
            filtered_range_ch1 = np.max(ch1_filtered) - np.min(ch1_filtered)
            if raw_range_ch1 > 1000:  # Only log when we see the extreme signals
                print(f"🔧 FILTERING EFFECT: CH1 {raw_range_ch1:,.0f}µV → {filtered_range_ch1:.0f}µV")

        # Now collect baseline using FILTERED data (not raw!)
        if not self.baseline_ready:
            self._collect_baseline_chunk(ch1_filtered.tolist(), ch2_filtered.tolist())

        # Store turn data using FILTERED signals
        if self.turn_active:
            self.turn_data.extend(zip(ch1_filtered, ch2_filtered))

    def _collect_baseline_chunk(self, ch1_filtered: List[float], ch2_filtered: List[float]):
        """Collect FILTERED data for baseline calculation."""
        if self.baseline_ready:
            return

        # Add FILTERED data to baseline buffer
        self._baseline_buffer_ch1.extend(ch1_filtered)
        self._baseline_buffer_ch2.extend(ch2_filtered)

        # Check if we have enough FILTERED data for baseline
        if len(self._baseline_buffer_ch1) >= self._samples_needed:
            self._calculate_baseline()

    def _calculate_baseline(self):
        """Calculate baseline statistics from FILTERED data."""
        # Convert to numpy arrays
        ch1_data = np.array(self._baseline_buffer_ch1[:self._samples_needed])
        ch2_data = np.array(self._baseline_buffer_ch2[:self._samples_needed])

        # Split into chunks for statistical robustness
        chunk_size = EEG_SAMPLE_RATE * 2  # 2-second chunks
        num_chunks = len(ch1_data) // chunk_size

        # Calculate powers for each chunk (data is already filtered!)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size

            chunk_ch1 = ch1_data[start_idx:end_idx]
            chunk_ch2 = ch2_data[start_idx:end_idx]

            if len(chunk_ch1) >= chunk_size:  # Full chunk
                chunk_powers = self._band_powers(chunk_ch1, chunk_ch2)

                # Store powers for each band
                for band in ['alpha', 'beta', 'theta', 'delta', 'gamma']:
                    self.baseline_chunks[band].append(chunk_powers[band])

        # Calculate mean and std for each band
        self.baseline_stats = {}
        for band in ['alpha', 'beta', 'theta', 'delta', 'gamma']:
            if len(self.baseline_chunks[band]) > 0:
                powers = np.array(self.baseline_chunks[band])
                mean_power = np.mean(powers)
                std_power = np.std(powers)

                self.baseline_stats[band] = {
                    'mean': mean_power,
                    'std': std_power
                }

        self.baseline_ready = True

        # Print baseline info
        print(f"\n✅ Baseline established from FILTERED data ({len(self.baseline_chunks['alpha'])} chunks):")
        for band, stats in self.baseline_stats.items():
            print(f"  {band:5s}: μ={stats['mean']:.2e}, σ={stats['std']:.2e}")
        print()

    def start_turn(self):
        """Start collecting EEG data for a new turn."""
        self.turn_data = []
        self.turn_active = True

    def end_turn(self, tts_duration: Optional[float] = None) -> float:
        """Process turn using properly filtered data."""

        if not self.turn_active:
            return self.current_engagement

        if len(self.turn_data) < 500:  # Need minimum data
            self.turn_active = False
            return self.current_engagement

        # Convert turn data to arrays (data is already filtered!)
        eeg = np.asarray(self.turn_data)
        ch1_data, ch2_data = eeg[:, 0], eeg[:, 1]

        # Calculate engagement trajectory using CLEAN data
        if self.baseline_ready:
            engagement_trajectory = self._calculate_baseline_trajectory(ch1_data, ch2_data)
        else:
            engagement_trajectory = self._calculate_ratio_trajectory(ch1_data, ch2_data)

        # Aggregate trajectory into single score
        final_engagement = self._aggregate_trajectory(engagement_trajectory)

        # Apply smoothing if needed
        if self.smoothing > 0:
            final_engagement = (self.smoothing * final_engagement +
                                (1 - self.smoothing) * self.current_engagement)

        self.current_engagement = np.clip(final_engagement, 0.0, 1.0)
        self.turn_idx += 1
        self.turn_active = False

        print(f"\n[Turn {self.turn_idx}] Clean filtered data → engagement={self.current_engagement:.3f}")
        return self.current_engagement

    def _band_powers(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate band powers from ALREADY FILTERED data."""
        nyq = EEG_SAMPLE_RATE / 2
        out = {}

        for name, (lo, hi) in self.bands.__dict__.items():
            hi = min(hi, nyq * 0.95)

            # Design band-specific filter (lighter, since data is already clean)
            sos = signal.butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")

            # Apply to already-filtered data
            xf = signal.sosfilt(sos, x)
            yf = signal.sosfilt(sos, y)

            # Calculate power
            power = 0.5 * (np.mean(xf ** 2) + np.mean(yf ** 2))
            out[name] = power

        return out

    def _calculate_baseline_trajectory(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> np.ndarray:
        """Calculate engagement using baseline-normalized approach."""
        # Window parameters
        window_size = int(2.0 * EEG_SAMPLE_RATE)  # 2-second windows
        overlap = int(0.5 * EEG_SAMPLE_RATE)  # 0.5-second overlap
        step_size = window_size - overlap

        engagement_values = []

        # Slide window across the entire epoch
        for start_idx in range(0, len(ch1_data) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # Extract window
            window_ch1 = ch1_data[start_idx:end_idx]
            window_ch2 = ch2_data[start_idx:end_idx]

            # Calculate powers for this window
            window_powers = self._band_powers(window_ch1, window_ch2)

            # Calculate z-scores using baseline
            z_scores = {}
            for band in ['alpha', 'beta', 'theta']:
                if band in self.baseline_stats:
                    baseline_mean = self.baseline_stats[band]['mean']
                    baseline_std = self.baseline_stats[band]['std']
                    z_scores[band] = (window_powers[band] - baseline_mean) / baseline_std
                else:
                    z_scores[band] = 0.0

            # Calculate engagement from z-scores
            window_engagement = self._calculate_engagement_from_z_scores(z_scores)
            engagement_values.append(window_engagement)

        return np.array(engagement_values)

    def _calculate_engagement_from_z_scores(self, z_scores: Dict[str, float]) -> float:
        """Calculate engagement from baseline-normalized z-scores."""
        alpha_z = z_scores.get('alpha', 0.0)
        beta_z = z_scores.get('beta', 0.0)
        theta_z = z_scores.get('theta', 0.0)

        # Engagement: high beta, low alpha (classic pattern)
        engagement_score = beta_z - 0.5 * alpha_z + 0.2 * theta_z

        # Normalize to [0,1] range
        normalized = 1 / (1 + np.exp(-engagement_score))  # Sigmoid

        return np.clip(normalized, 0.1, 0.9)

    def _calculate_ratio_trajectory(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> np.ndarray:
        """Calculate engagement over time using direct ratios (fallback)."""
        window_size = int(2.0 * EEG_SAMPLE_RATE)
        overlap = int(0.5 * EEG_SAMPLE_RATE)
        step_size = window_size - overlap

        engagement_values = []

        for start_idx in range(0, len(ch1_data) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            window_ch1 = ch1_data[start_idx:end_idx]
            window_ch2 = ch2_data[start_idx:end_idx]

            window_powers = self._band_powers(window_ch1, window_ch2)
            window_engagement = self._calculate_engagement_from_ratios(window_powers)
            engagement_values.append(window_engagement)

        return np.array(engagement_values)

    def _calculate_engagement_from_ratios(self, powers: Dict[str, float]) -> float:
        """Calculate engagement using direct band power ratios."""
        alpha = powers.get('alpha', 1.0)
        beta = powers.get('beta', 1.0)
        theta = powers.get('theta', 1.0)

        # Classic engagement ratio: Beta / (Alpha + Theta)
        denominator = alpha + theta
        if denominator > 0:
            engagement_ratio = beta / denominator
        else:
            engagement_ratio = 1.0

        # Normalize to [0,1] range
        normalized = np.tanh(engagement_ratio - 0.5) * 0.3 + 0.5
        return np.clip(normalized, 0.1, 0.9)

    def _aggregate_trajectory(self, trajectory: np.ndarray) -> float:
        """Aggregate engagement trajectory into single score."""
        if len(trajectory) == 0:
            return 0.5
        if len(trajectory) == 1:
            return trajectory[0]

        # Weighted average with more weight to recent values
        weights = np.linspace(0.5, 1.5, len(trajectory))
        weighted_avg = np.average(trajectory, weights=weights)

        # Add small trend bonus
        if len(trajectory) >= 3:
            trend = trajectory[-1] - trajectory[0]
            trend_bonus = trend * 0.1
        else:
            trend_bonus = 0

        final_score = weighted_avg + trend_bonus
        return np.clip(final_score, 0.1, 0.9)

    @property
    def baseline_collected(self) -> bool:
        """Return baseline collection status."""
        return self.baseline_ready

    def get_current_engagement(self) -> float:
        """Return the most recent engagement value."""
        return self.current_engagement

    def get_filter_info(self) -> dict:
        """Get information about the filtering being used."""
        return self.online_filter.get_filter_info()