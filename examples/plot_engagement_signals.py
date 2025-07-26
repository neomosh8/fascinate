import numpy as np
import matplotlib.pyplot as plt

from config import EEG_SAMPLE_RATE
from neocore_client import OnlineFilter


def simulate_eeg(duration_sec: float = 5.0) -> tuple:
    """Generate two-channel synthetic EEG data."""
    fs = EEG_SAMPLE_RATE
    t = np.arange(0, duration_sec, 1 / fs)

    # Channel 1: mix of 10 Hz and 20 Hz + noise
    ch1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    ch1 += 0.1 * np.random.randn(len(t))

    # Channel 2: mix of 10 Hz and 15 Hz + noise
    ch2 = np.sin(2 * np.pi * 10 * t + np.pi / 8) + 0.5 * np.sin(2 * np.pi * 15 * t)
    ch2 += 0.1 * np.random.randn(len(t))

    return t, ch1, ch2


def apply_filter(ch1: np.ndarray, ch2: np.ndarray) -> tuple:
    """Filter signals using the same preprocessing as engagement scoring."""
    filt = OnlineFilter(EEG_SAMPLE_RATE)
    chunk = 250  # 1 second chunks
    f_ch1, f_ch2 = [], []
    for i in range(0, len(ch1), chunk):
        seg1 = ch1[i : i + chunk]
        seg2 = ch2[i : i + chunk]
        out1, out2 = filt.filter_chunk(seg1, seg2)
        f_ch1.append(out1)
        f_ch2.append(out2)
    return np.concatenate(f_ch1), np.concatenate(f_ch2)


def main():
    t, ch1_raw, ch2_raw = simulate_eeg()
    ch1_filt, ch2_filt = apply_filter(ch1_raw, ch2_raw)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, ch1_raw, label="Raw Ch1", alpha=0.5)
    axes[0].plot(t, ch1_filt, label="Filtered Ch1", linewidth=1)
    axes[0].set_title("Channel 1")
    axes[0].legend()

    axes[1].plot(t, ch2_raw, label="Raw Ch2", alpha=0.5)
    axes[1].plot(t, ch2_filt, label="Filtered Ch2", linewidth=1)
    axes[1].set_title("Channel 2")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
