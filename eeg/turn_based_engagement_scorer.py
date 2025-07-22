import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import EEG_SAMPLE_RATE


@dataclass
class FrequencyBands:
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)


class TurnBasedEngagementScorer:
    """
    Engagement scorer with proper individualized baseline correction using z-scores.
    Collects baseline during initial period, then normalizes all future measurements
    relative to that user's personal baseline.
    """

    def __init__(self,
                 baseline_duration_sec: float = 30,
                 baseline_update_rate: float = 0.001,
                 smoothing: float = 0.0,
                 normalisation_mode: str = "z_score"  # Only z_score mode now
                 ):

        self.bands = FrequencyBands()
        self.filters = self._design_filters()
        self.smoothing = smoothing

        # Baseline collection parameters
        self.baseline_duration_sec = baseline_duration_sec
        self.baseline_update_rate = float(np.clip(baseline_update_rate, 0, 1))
        self._samples_needed = int(baseline_duration_sec * EEG_SAMPLE_RATE)

        # Baseline data collection
        self.baseline_chunks = {'alpha': [], 'beta': [], 'theta': [], 'delta': [], 'gamma': []}
        self.baseline_stats = {}  # Will store {'mean': X, 'std': Y} for each band
        self.baseline_ready = False

        # Turn state
        self.turn_data: List[Tuple[float, float]] = []
        self.turn_active = False
        self.turn_idx = 0
        self.current_engagement = 0.5

        # Buffer for collecting baseline during initial period
        self._baseline_buffer_ch1: List[float] = []
        self._baseline_buffer_ch2: List[float] = []

    def _design_filters(self):
        """Design bandpass filters for each frequency band."""
        nyq = EEG_SAMPLE_RATE / 2
        flts = {}
        for name, (lo, hi) in self.bands.__dict__.items():
            hi = min(hi, nyq * 0.95)
            sos = signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
            flts[name] = sos
        return flts

    def _band_powers(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate power in each frequency band for both channels."""
        out = {}
        for name, sos in self.filters.items():
            xf = signal.sosfilt(sos, x)
            yf = signal.sosfilt(sos, y)
            # Average power across both channels
            out[name] = 0.5 * (np.mean(xf ** 2) + np.mean(yf ** 2))
        return out

    def _collect_baseline_chunk(self, ch1: List[float], ch2: List[float]):
        """Collect data for baseline calculation during initial period."""
        if self.baseline_ready:
            return

        # Add to baseline buffer
        self._baseline_buffer_ch1.extend(ch1)
        self._baseline_buffer_ch2.extend(ch2)

        # Check if we have enough data for baseline
        if len(self._baseline_buffer_ch1) >= self._samples_needed:
            self._calculate_baseline()

    def _calculate_baseline(self):
        """Calculate baseline statistics from collected data."""
        # Convert to numpy arrays
        ch1_data = np.array(self._baseline_buffer_ch1[:self._samples_needed])
        ch2_data = np.array(self._baseline_buffer_ch2[:self._samples_needed])

        # Split into chunks for statistical robustness
        chunk_size = EEG_SAMPLE_RATE * 2  # 2-second chunks
        num_chunks = len(ch1_data) // chunk_size

        # Calculate powers for each chunk
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

                # Ensure minimum std (10% of mean) to avoid division by tiny numbers
                min_std = mean_power * 0.1
                std_power = max(std_power, min_std)

                self.baseline_stats[band] = {
                    'mean': mean_power,
                    'std': std_power
                }

        self.baseline_ready = True

        # Print baseline info
        print(f"\n✅ Baseline established from {len(self.baseline_chunks['alpha'])} chunks:")
        for band, stats in self.baseline_stats.items():
            print(f"  {band:5s}: μ={stats['mean']:.2e}, σ={stats['std']:.2e}")
        print()

    def start_turn(self):
        """Start collecting EEG data for a new turn."""
        self.turn_data = []
        self.turn_active = True

    def add_eeg_chunk(self, ch1: List[float], ch2: List[float]):
        """Add EEG data chunk - either for baseline or current turn."""
        if not self.baseline_ready:
            # Still collecting baseline
            self._collect_baseline_chunk(ch1, ch2)
        elif self.turn_active:
            # Collecting turn data
            self.turn_data.extend(zip(ch1, ch2))

    def end_turn(self, tts_duration: Optional[float] = None) -> float:
        """Process turn with temporal dynamics analysis."""
        if not self.turn_active or not self.baseline_ready:
            self.turn_active = False
            return self.current_engagement

        if len(self.turn_data) < 500:  # Need minimum data
            self.turn_active = False
            return self.current_engagement

        # Convert turn data to arrays
        eeg = np.asarray(self.turn_data)
        ch1_data, ch2_data = eeg[:, 0], eeg[:, 1]

        # 🎯 NEW: Windowed analysis instead of single mean
        engagement_trajectory = self._calculate_engagement_trajectory(ch1_data, ch2_data)

        # Aggregate trajectory into single score with temporal awareness
        final_engagement = self._aggregate_trajectory(engagement_trajectory)

        # Apply smoothing and update
        if self.smoothing > 0:
            final_engagement = (self.smoothing * final_engagement +
                                (1 - self.smoothing) * self.current_engagement)

        self.current_engagement = max(0, min(1, final_engagement))

        # Update baseline
        current_powers = self._band_powers(ch1_data, ch2_data)
        self._update_baseline(current_powers)

        self.turn_idx += 1
        self.turn_active = False

        # Debug output with trajectory info
        traj_info = f"peak={np.max(engagement_trajectory):.3f}, trend={engagement_trajectory[-1] - engagement_trajectory[0]:+.3f}"
        print(f"\n[Turn {self.turn_idx}] Trajectory: {traj_info} → final={self.current_engagement:.3f}")

        return self.current_engagement

    def _calculate_engagement_trajectory(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> np.ndarray:
        """Calculate engagement over time using sliding windows."""

        # Window parameters
        window_size = int(2.0 * EEG_SAMPLE_RATE)  # 2-second windows
        overlap = int(0.5 * EEG_SAMPLE_RATE)  # 0.5-second overlap
        step_size = window_size - overlap  # 1.5-second steps

        engagement_values = []

        # Slide window across the entire epoch
        for start_idx in range(0, len(ch1_data) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # Extract window
            window_ch1 = ch1_data[start_idx:end_idx]
            window_ch2 = ch2_data[start_idx:end_idx]

            # Calculate powers for this window
            window_powers = self._band_powers(window_ch1, window_ch2)

            # Calculate z-scores for this window
            z_scores = {}
            for band in ['alpha', 'beta', 'theta']:
                if band in self.baseline_stats:
                    baseline_mean = self.baseline_stats[band]['mean']
                    baseline_std = self.baseline_stats[band]['std']
                    z_scores[band] = (window_powers[band] - baseline_mean) / baseline_std
                else:
                    z_scores[band] = 0.0

            # Calculate engagement for this window
            window_engagement = self._calculate_engagement_from_z_scores(z_scores)
            engagement_values.append(window_engagement)

        return np.array(engagement_values)

    def _aggregate_trajectory(self, trajectory: np.ndarray) -> float:
        """Aggregate engagement trajectory into single score with temporal awareness."""

        if len(trajectory) == 0:
            return 0.5

        if len(trajectory) == 1:
            return trajectory[0]

        # Multiple aggregation strategies - choose based on what matters most

        # Strategy 1: Weighted average (more weight to recent)
        weights = np.linspace(0.5, 1.5, len(trajectory))  # More weight to end
        weighted_avg = np.average(trajectory, weights=weights)

        # Strategy 2: Peak engagement (reward high moments)
        peak_engagement = np.max(trajectory)

        # Strategy 3: Final vs initial (trend matters)
        if len(trajectory) >= 3:
            trend = trajectory[-1] - trajectory[0]  # Positive = improving
        else:
            trend = 0

        # Strategy 4: Stability (consistent engagement is good)
        stability = 1.0 - np.std(trajectory)  # High stability = low variance
        stability = max(0, stability)

        # Strategy 5: 75th percentile (most of the time was good)
        percentile_75 = np.percentile(trajectory, 75)

        # 🎯 Combine strategies (you can tune these weights)
        final_score = (
                0.3 * weighted_avg +  # Moderate weight to overall average
                0.1 * peak_engagement +  # Small bonus for high moments
                0.4 * max(0, trend * 5) +  # BIG bonus for positive trends (×5 amplification)
                0.2 * percentile_75  # Sustained good performance
            # Note: removed stability weight to focus more on building engagement
        )

        return np.clip(final_score, 0, 1)

    def _calculate_engagement_from_z_scores(self, z_scores: Dict[str, float]) -> float:
        """Calculate engagement score from z-scored band powers with proper sensitivity."""

        z_alpha = z_scores.get('alpha', 0.0)
        z_beta = z_scores.get('beta', 0.0)
        z_theta = z_scores.get('theta', 0.0)

        # Method: Linear combination approach (more interpretable and sensitive)
        # High engagement = high beta + low alpha + low theta (relative to baseline)

        # Weights based on EEG literature
        beta_weight = 1.0  # Beta up = good
        alpha_weight = -0.5  # Alpha down = good (attention)
        theta_weight = -0.3  # Theta down = good (but less weight)

        # Calculate raw engagement score
        engagement_raw = (beta_weight * z_beta +
                          alpha_weight * z_alpha +
                          theta_weight * z_theta)

        # Alternative: Traditional ratio but with proper handling
        # ratio_raw = z_beta - 0.5 * (z_alpha + z_theta)  # Beta high, alpha+theta low

        # More sensitive normalization - don't compress as much
        # Map from roughly [-6, +6] range to [0, 1]
        engagement_normalized = engagement_raw / 12.0 + 0.5

        # Clip to valid range
        engagement = np.clip(engagement_normalized, 0.0, 1.0)

        return engagement

    def _update_baseline(self, current_powers: Dict[str, float]):
        """Slowly adapt baseline to account for session-long changes."""
        if not self.baseline_ready:
            return

        for band in ['alpha', 'beta', 'theta', 'delta', 'gamma']:
            if band in self.baseline_stats and band in current_powers:
                # Update mean with exponential moving average
                old_mean = self.baseline_stats[band]['mean']
                new_power = current_powers[band]
                updated_mean = ((1 - self.baseline_update_rate) * old_mean +
                                self.baseline_update_rate * new_power)

                # Update std more conservatively
                old_std = self.baseline_stats[band]['std']
                power_deviation = abs(new_power - old_mean)
                updated_std = ((1 - self.baseline_update_rate * 0.1) * old_std +
                               self.baseline_update_rate * 0.1 * power_deviation)

                # Ensure minimum std
                min_std = updated_mean * 0.1
                updated_std = max(updated_std, min_std)

                self.baseline_stats[band]['mean'] = updated_mean
                self.baseline_stats[band]['std'] = updated_std

    @property
    def baseline_collected(self) -> bool:
        """Compatibility property for old interface."""
        return self.baseline_ready

    def get_current_engagement(self) -> float:
        """Return the most recent engagement value."""
        return self.current_engagement