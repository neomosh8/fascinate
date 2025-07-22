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
    beta:  Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)


class TurnBasedEngagementScorer:
    """
    Engagement scorer with two normalisation modes:

    mode="relative"   -> beta / (alpha+theta) using relative power,
                         **excluding delta** from the denominator.
    mode="baseline"   -> beta' / (alpha'+theta') where each band is
                         divided by its own running baseline.

    Switch modes by passing `normalisation_mode` in __init__.
    """

    def __init__(self,
                 baseline_duration_sec: float = 10,
                 baseline_update_rate: float = 0.001,
                 smoothing: float = 0.0,
                 normalisation_mode: str = "baseline"  # "relative" or "baseline"
                 ):

        self.mode = normalisation_mode.lower()
        if self.mode not in {"relative", "baseline"}:
            raise ValueError("normalisation_mode must be 'relative' or 'baseline'")

        self.bands = FrequencyBands()
        self.filters = self._design_filters()

        # rolling baseline (absolute powers)
        self._samples_needed = int(baseline_duration_sec * EEG_SAMPLE_RATE)
        self._base_raw: Dict[str, float] | None = None   # absolute powers baseline
        self._buf_ch1: List[float] = []
        self._buf_ch2: List[float] = []
        self.beta_rate = float(np.clip(baseline_update_rate, 0, 1))

        # state
        self.turn_data: List[Tuple[float, float]] = []
        self.turn_active = False
        self.turn_idx = 0
        self.smoothing = smoothing
        self.current_engagement = 0.5

    # --------------- filter design & helpers -----------------
    def _design_filters(self):
        nyq = EEG_SAMPLE_RATE / 2
        flts = {}
        for name, (lo, hi) in self.bands.__dict__.items():
            hi = min(hi, nyq * 0.95)
            sos = signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
            flts[name] = sos
        return flts

    def _band_powers(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        out = {}
        for name, sos in self.filters.items():
            xf = signal.sosfilt(sos, x)
            yf = signal.sosfilt(sos, y)
            out[name] = 0.5 * (np.mean(xf ** 2) + np.mean(yf ** 2))
        return out

    # --------------- baseline handling -----------------
    @property
    def baseline_ready(self) -> bool:
        return self._base_raw is not None

    def _feed_baseline(self, c1: List[float], c2: List[float]):
        if self.baseline_ready:
            return
        self._buf_ch1.extend(c1)
        self._buf_ch2.extend(c2)
        if len(self._buf_ch1) >= self._samples_needed:
            a = np.array(self._buf_ch1[:self._samples_needed])
            b = np.array(self._buf_ch2[:self._samples_needed])
            self._base_raw = self._band_powers(a, b)
            print("\n[Baseline captured]")
            for k in self.filters:
                print(f"{k:5s} abs={self._base_raw[k]:.4e}")
            print()

    def _update_baseline(self, new_raw: Dict[str, float]):
        r = self.beta_rate
        for k in self._base_raw:
            self._base_raw[k] = (1 - r) * self._base_raw[k] + r * new_raw[k]

    # --------------- public API -----------------
    def start_turn(self):
        self.turn_data = []
        self.turn_active = True

    def add_eeg_chunk(self, ch1: List[float], ch2: List[float]):
        self._feed_baseline(ch1, ch2)
        if self.turn_active and self.baseline_ready:
            self.turn_data.extend(zip(ch1, ch2))

    def end_turn(self, tts_duration: Optional[float] = None) -> float:
        if not self.turn_active or len(self.turn_data) < 100:
            self.turn_active = False
            return self.current_engagement

        eeg = np.asarray(self.turn_data)
        a, b = eeg[:, 0], eeg[:, 1]
        raw_pow = self._band_powers(a, b)

        # -------- normalisation branch -------
        if self.mode == "relative":
            total_no_delta = sum(raw_pow[k] for k in raw_pow if k != "delta")
            rel = {k: raw_pow[k] / total_no_delta if total_no_delta else 0
                   for k in raw_pow}
            beta, alpha, theta = rel["beta"], rel["alpha"], rel["theta"]
            base_rel = {k: self._base_raw[k] / total_no_delta
                        for k in ("beta", "alpha", "theta")}
            base_ratio = base_rel["beta"] / (base_rel["alpha"] + base_rel["theta"])
            raw_ratio = beta / (alpha + theta) if (alpha + theta) else 0

        else:  # "baseline" mode
            beta = raw_pow["beta"]  / self._base_raw["beta"]
            alpha = raw_pow["alpha"] / self._base_raw["alpha"]
            theta = raw_pow["theta"] / self._base_raw["theta"]
            base_ratio = 1.0  # by definition after band-wise normalisation
            raw_ratio = beta / (alpha + theta) if (alpha + theta) else 0

        engagement = raw_ratio / base_ratio if base_ratio else 0

        if self.smoothing:
            engagement = (self.smoothing * engagement +
                          (1 - self.smoothing) * self.current_engagement)

        self.current_engagement = max(0, engagement)
        self.turn_idx += 1
        self.turn_active = False
        self._update_baseline(raw_pow)

        # ------------ debug print ------------
        print(f"\n[Turn {self.turn_idx}] ({self.mode}) "
              f"raw_ratio={raw_ratio:.4f} base_ratio={base_ratio:.4f} "
              f"engagement={self.current_engagement:.3f}")
        return self.current_engagement

    # ------------------------------------------------ utility getter
    def get_current_engagement(self) -> float:
        """Return the most recent engagement value."""
        return self.current_engagement
