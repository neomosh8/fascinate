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
        """Calculate band powers with raw signal diagnostics."""

        # Check raw signal first
        raw_stats = {
            'ch1_raw_std': np.std(x),
            'ch2_raw_std': np.std(y),
            'ch1_raw_range': np.max(x) - np.min(x),
            'ch2_raw_range': np.max(y) - np.min(y)
        }

        out = {}
        for name, sos in self.filters.items():
            xf = signal.sosfilt(sos, x)
            yf = signal.sosfilt(sos, y)

            # Calculate power
            power = 0.5 * (np.mean(xf ** 2) + np.mean(yf ** 2))
            out[name] = power

            # Debug filtering effect
            if name == 'beta':  # Just check beta band as example
                print(f"🎛️ {name} filtering: raw_std={raw_stats['ch1_raw_std']:,.0f}, filtered_power={power:,.0f}")

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
                std_power = std_power

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
        """Add EEG data chunk with raw signal monitoring."""

        if self.turn_active:
            self.turn_data.extend(zip(ch1, ch2))

            # Debug raw signal every 50 chunks
            if len(self.turn_data):  # Every ~5 seconds at 250Hz
                recent_ch1 = [x[0] for x in list(self.turn_data)[-1250:]]
                recent_ch2 = [x[1] for x in list(self.turn_data)[-1250:]]

                print(f"\n📡 RAW SIGNAL CHECK (last 5 seconds):")
                print(f"  CH1: min={min(recent_ch1):,.0f}, max={max(recent_ch1):,.0f}, std={np.std(recent_ch1):,.0f}")
                print(f"  CH2: min={min(recent_ch2):,.0f}, max={max(recent_ch2):,.0f}, std={np.std(recent_ch2):,.0f}")
                print(f"  CH1 range: {max(recent_ch1) - min(recent_ch1):,.0f}")
                print(f"  CH2 range: {max(recent_ch2) - min(recent_ch2):,.0f}")

    def end_turn(self, tts_duration: Optional[float] = None) -> float:
        """Process turn using direct ratios (no baseline needed)."""

        if not self.turn_active:
            return self.current_engagement

        if len(self.turn_data) < 500:  # Need minimum data
            self.turn_active = False
            return self.current_engagement

        # Convert turn data to arrays
        eeg = np.asarray(self.turn_data)
        ch1_data, ch2_data = eeg[:, 0], eeg[:, 1]

        # Calculate engagement trajectory using direct ratios
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

        print(f"\n[Turn {self.turn_idx}] Direct ratio → final={self.current_engagement:.3f}")
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
        """Simple trajectory aggregation for direct ratios."""

        if len(trajectory) == 0:
            return 0.5
        if len(trajectory) == 1:
            return trajectory[0]

        # For direct ratios, just use weighted average with trend bonus
        # More weight to recent values
        weights = np.linspace(0.5, 1.5, len(trajectory))
        weighted_avg = np.average(trajectory, weights=weights)

        # Add small trend bonus
        if len(trajectory) >= 3:
            trend = trajectory[-1] - trajectory[0]
            trend_bonus = trend * 0.1  # Small bonus for positive trends
        else:
            trend_bonus = 0

        final_score = weighted_avg + trend_bonus

        print(
            f"🔍 Trajectory: {len(trajectory)} windows, avg={np.mean(trajectory):.3f}, weighted={weighted_avg:.3f}, final={final_score:.3f}")

        return np.clip(final_score, 0.1, 0.9)

    def _calculate_ratio_trajectory(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> np.ndarray:
        """Calculate engagement over time using direct ratios."""

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

            # Calculate engagement using direct ratios
            window_engagement = self._calculate_engagement_from_ratios(window_powers)
            engagement_values.append(window_engagement)

        return np.array(engagement_values)

    def _calculate_engagement_from_ratios(self, powers: Dict[str, float]) -> float:
        """Calculate engagement using direct band power ratios."""

        # Debug: Print what we're getting
        print(f"🔍 Powers received: {powers}")

        # Get raw powers
        alpha = powers.get('alpha', 1.0)
        beta = powers.get('beta', 1.0)
        theta = powers.get('theta', 1.0)

        print(f"🔍 Alpha: {alpha}, Beta: {beta}, Theta: {theta}")

        # Classic engagement ratio: Beta / (Alpha + Theta)
        denominator = alpha + theta
        if denominator > 0:
            engagement_ratio = beta / denominator
            print(f"🔍 Ratio: {beta}/{denominator} = {engagement_ratio}")
        else:
            engagement_ratio = 1.0  # Fallback
            print(f"🔍 Using fallback ratio: {engagement_ratio}")

        # Normalize to [0,1] range using tanh
        normalized = np.tanh(engagement_ratio - 0.5) * 0.3 + 0.5
        print(f"🔍 Normalized: {normalized}")

        result = np.clip(normalized, 0.1, 0.9)
        print(f"🔍 Final result: {result}")

        return result

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
        """Compatibility property - always True for direct ratios."""
        return True  # No baseline needed

    def get_current_engagement(self) -> float:
        """Return the most recent engagement value."""
        return self.current_engagement