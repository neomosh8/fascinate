import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

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

        # Emotion tracking
        self.current_emotion = 0.5
        self.emotion_history = deque(maxlen=10)

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
        """Calculate baseline statistics from FILTERED data with
        log‑power transform and 10 % outlier trimming."""
        ch1 = np.asarray(self._baseline_buffer_ch1[: self._samples_needed])
        ch2 = np.asarray(self._baseline_buffer_ch2[: self._samples_needed])

        chunk_size = EEG_SAMPLE_RATE * 2  # 2 s
        num_chunks = len(ch1) // chunk_size

        # Collect log‑powers for each band per 2 s chunk
        for i in range(num_chunks):
            s = i * chunk_size
            e = s + chunk_size
            powers = self._band_powers(ch1[s:e], ch2[s:e])

            for band, p in powers.items():
                # log power to curb heavy tails
                self.baseline_chunks[band].append(np.log(p + 1e-12))

        # Compute trimmed mean / std (drop top and bottom 10 %)
        self.baseline_stats = {}
        for band, arr in self.baseline_chunks.items():
            data = np.sort(np.asarray(arr))
            trim = max(1, int(0.10 * len(data)))
            core = data[trim:-trim] if len(data) > 2 * trim else data
            self.baseline_stats[band] = {
                "mean": np.mean(core),
                "std": np.std(core) + 1e-12,  # avoid div‑by‑zero
            }

        self.baseline_ready = True
        print("\n✅ Baseline established:")
        for b, s in self.baseline_stats.items():
            print(f"  {b:5s}: μ={s['mean']:.2e}, σ={s['std']:.2e}")
        print()

    def start_turn(self):
        """Start collecting EEG data for a new turn."""
        self.turn_data = []
        self.turn_active = True

    # ------------------------------------------------------------------
    # 3.  Adaptive baseline update called at the end of every turn
    # ------------------------------------------------------------------
    def end_turn(self, tts_duration: Optional[float] = None) -> Tuple[float, float]:
        if not self.turn_active:
            return self.current_engagement, self.current_emotion
        if len(self.turn_data) < 500:
            self.turn_active = False
            return self.current_engagement, self.current_emotion
        # ADD THIS CHECK:
        if not self.baseline_ready:
            self.turn_active = False
            return 0.5, 0.5  # Return neutral engagement if baseline not ready

        eeg = np.asarray(self.turn_data)
        ch1 = eeg[:, 0]
        ch2 = eeg[:, 1]

        # Engagement trajectory
        if self.baseline_ready:
            traj = self._calculate_baseline_trajectory(ch1, ch2)
        else:
            traj = self._calculate_ratio_trajectory(ch1, ch2)

        # Use the latest window only (more reactive)
        final_engagement = traj[-1] if len(traj) else 0.5
        self.current_engagement = np.clip(final_engagement, 0.0, 1.0)

        # Calculate emotion from alpha asymmetry
        self.current_emotion = self._calculate_emotion_from_alpha_asymmetry(ch1, ch2)
        self.emotion_history.append(self.current_emotion)
        self.turn_idx += 1
        self.turn_active = False
        print(
            f"[Turn {self.turn_idx}] engagement={self.current_engagement:.3f}, emotion={self.current_emotion:.3f}"
        )

        # ---- adaptive baseline blend ---------------------------------
        if self.baseline_ready and self.baseline_update_rate > 0:
            r = self.baseline_update_rate
            new_pow = self._band_powers(ch1, ch2)
            for band, stats in self.baseline_stats.items():
                new_log = np.log(new_pow[band] + 1e-12)
                mean = stats["mean"]
                std = stats["std"]
                stats["mean"] = (1 - r) * mean + r * new_log
                stats["std"] = (1 - r) * std + r * abs(new_log - mean)
        # --------------------------------------------------------------

        return self.current_engagement, self.current_emotion

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

    def _extract_alpha_power(self, data: np.ndarray) -> float:
        """Extract alpha band power from EEG data."""
        if len(data) < 100:
            return 1.0
        nyq = EEG_SAMPLE_RATE / 2
        sos = signal.butter(4, [8.0 / nyq, 13.0 / nyq], btype="band", output="sos")
        alpha_filtered = signal.sosfilt(sos, data)
        alpha_power = np.mean(alpha_filtered ** 2)
        return alpha_power

    def _calculate_emotion_from_alpha_asymmetry(
        self, ch1_data: np.ndarray, ch2_data: np.ndarray
    ) -> float:
        """Calculate emotion from alpha asymmetry."""
        alpha_power_ch1 = self._extract_alpha_power(ch1_data)
        alpha_power_ch2 = self._extract_alpha_power(ch2_data)

        if alpha_power_ch1 + alpha_power_ch2 > 0:
            asymmetry = (alpha_power_ch2 - alpha_power_ch1) / (
                alpha_power_ch1 + alpha_power_ch2
            )
            emotion_score = 0.5 + (asymmetry * 0.5)
            emotion_score = np.clip(emotion_score, 0.0, 1.0)
        else:
            emotion_score = 0.5

        return emotion_score

    # ------------------------------------------------------------------
    # Consistent baseline‑normalised trajectory (log power everywhere)
    # ------------------------------------------------------------------
    def _calculate_baseline_trajectory(self,
                                       ch1_data: np.ndarray,
                                       ch2_data: np.ndarray) -> np.ndarray:
        win = int(2.0 * EEG_SAMPLE_RATE)  # 2 s
        step = int(1.5 * EEG_SAMPLE_RATE)  # 0.5 s overlap

        out = []
        for s in range(0, len(ch1_data) - win + 1, step):
            e = s + win
            powers = self._band_powers_log(ch1_data[s:e], ch2_data[s:e])

            z = {}
            for band in ("alpha", "beta", "theta"):
                mu = self.baseline_stats[band]["mean"]
                sig = self.baseline_stats[band]["std"]
                z[band] = (powers[band] - mu) / sig

            out.append(self._calculate_engagement_from_z_scores(z))
        return np.asarray(out)

    # ------------------------------------------------------------------
    # 4.  Linear engagement mapping (no sigmoid, no extra smoothing)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Gentler linear mapping (gain = 0.04) and hard clip
    # ------------------------------------------------------------------
    def _calculate_engagement_from_z_scores(self,
                                            z: Dict[str, float]) -> float:
        beta_z = z.get("beta", 0.0)
        alpha_z = z.get("alpha", 0.0)
        theta_z = z.get("theta", 0.0)

        score = beta_z - 0.5 * alpha_z + 0.2 * theta_z
        return np.clip(0.5 + 0.15 * score, 0.0, 1.0)

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
            if self.turn_idx == 0:
                print('alpha', window_powers['alpha'],
                      'beta', window_powers['beta'],
                      'theta', window_powers['theta'])
            z = {b: (window_powers[b] - self.baseline_stats[b]['mean']) /
                    self.baseline_stats[b]['std']
                 for b in ['alpha', 'beta', 'theta']}
            print('z', z)

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

    def get_current_emotion(self) -> float:
        """Get current emotion score."""
        return self.current_emotion

    def get_emotion_trend(self) -> str:
        """Get emotion trend description."""
        if self.current_emotion > 0.6:
            return "approach/positive"
        elif self.current_emotion < 0.4:
            return "withdrawal/negative"
        else:
            return "neutral"

    # helper: always return log power
    def _band_powers_log(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        raw = self._band_powers(x, y)  # existing function
        return {b: np.log(p + 1e-12) for b, p in raw.items()}
