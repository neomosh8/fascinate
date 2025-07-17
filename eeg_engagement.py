# eeg_engagement.py
import numpy as np
from collections import deque
from scipy import signal
import asyncio
from typing import Optional, Callable
import time
import matplotlib.pyplot as plt


class EngagementProcessor:
    """Real-time EEG engagement/attention calculation - Adaptive for different EEG types"""

    def __init__(self, sample_rate: int = 250, window_size: int = 2):
        self.sample_rate = sample_rate
        self.window_size = window_size  # seconds
        self.window_samples = sample_rate * window_size

        # Data buffers
        self.ch1_buffer = deque(maxlen=self.window_samples * 2)
        self.ch2_buffer = deque(maxlen=self.window_samples * 2)

        # Engagement metrics
        self.current_engagement = 0.5  # baseline
        self.engagement_history = deque(maxlen=1000)  # Store more for final plot
        self.timestamps = deque(maxlen=1000)  # Store timestamps

        # Baseline tracking for adaptive calculation
        self.baseline_alpha = deque(maxlen=50)  # Track baseline band powers
        self.baseline_beta = deque(maxlen=50)
        self.baseline_theta = deque(maxlen=50)
        self.baseline_established = False

        # Callback for engagement updates
        self.engagement_callback: Optional[Callable] = None

        self.last_calculation = time.time()
        self.start_time = time.time()

    def set_engagement_callback(self, callback: Callable[[float], None]):
        """Set callback function to receive engagement updates"""
        self.engagement_callback = callback

    def add_eeg_data(self, ch1_data: list, ch2_data: list):
        """Add new EEG data and calculate engagement"""
        self.ch1_buffer.extend(ch1_data)
        self.ch2_buffer.extend(ch2_data)

        # Calculate engagement every 0.5 seconds
        if time.time() - self.last_calculation > 0.5:
            engagement = self._calculate_engagement()
            if engagement is not None:
                self.current_engagement = engagement
                self.engagement_history.append(engagement)
                self.timestamps.append(time.time() - self.start_time)  # Relative time

                if self.engagement_callback:
                    self.engagement_callback(engagement)

            self.last_calculation = time.time()

    def _calculate_engagement(self) -> Optional[float]:
        """Calculate engagement/attention metric from EEG data - Adaptive for in-ear EEG"""
        if len(self.ch1_buffer) < self.window_samples:
            return None

        # Get recent data
        ch1_data = np.array(list(self.ch1_buffer)[-self.window_samples:])
        ch2_data = np.array(list(self.ch2_buffer)[-self.window_samples:])

        # Calculate power spectral density
        freqs1, psd1 = signal.welch(ch1_data, self.sample_rate, nperseg=self.sample_rate // 2)
        freqs2, psd2 = signal.welch(ch2_data, self.sample_rate, nperseg=self.sample_rate // 2)

        # Average across channels
        psd_avg = (psd1 + psd2) / 2

        # Define frequency bands
        alpha_band = (8, 12)  # Alpha waves (relaxed awareness)
        beta_band = (13, 30)  # Beta waves (focused attention)
        theta_band = (4, 7)  # Theta waves (drowsiness)

        # Calculate band powers
        alpha_power = self._band_power(freqs1, psd_avg, alpha_band)
        beta_power = self._band_power(freqs1, psd_avg, beta_band)
        theta_power = self._band_power(freqs1, psd_avg, theta_band)

        print(f"Raw Powers - Alpha: {alpha_power:.4f}, Beta: {beta_power:.4f}, Theta: {theta_power:.4f}")

        # Build baseline for adaptive calculation
        self.baseline_alpha.append(alpha_power)
        self.baseline_beta.append(beta_power)
        self.baseline_theta.append(theta_power)

        if len(self.baseline_alpha) < 20:  # Need some baseline data
            return 0.5  # Return neutral until we have baseline

        # Calculate adaptive engagement based on your EEG characteristics
        engagement = self._calculate_adaptive_engagement(alpha_power, beta_power, theta_power)

        return float(np.clip(engagement, 0, 1))

    def _calculate_adaptive_engagement(self, alpha_power: float, beta_power: float, theta_power: float) -> float:
        """Calculate engagement using multiple adaptive methods"""

        # Get baseline statistics
        alpha_baseline = np.mean(self.baseline_alpha)
        beta_baseline = np.mean(self.baseline_beta)
        theta_baseline = np.mean(self.baseline_theta)

        alpha_std = np.std(self.baseline_alpha) + 1e-6
        beta_std = np.std(self.baseline_beta) + 1e-6
        theta_std = np.std(self.baseline_theta) + 1e-6

        # Method 1: Relative band power changes (z-scores)
        alpha_z = (alpha_power - alpha_baseline) / alpha_std
        beta_z = (beta_power - beta_baseline) / beta_std
        theta_z = (theta_power - theta_baseline) / theta_std

        # Higher beta = more attention, lower theta = less drowsy, moderate alpha = relaxed focus
        attention_score1 = beta_z - theta_z + (0.5 - abs(alpha_z)) * 0.5

        # Method 2: Ratio-based (normalized by total power)
        total_power = alpha_power + beta_power + theta_power
        if total_power > 0:
            alpha_ratio = alpha_power / total_power
            beta_ratio = beta_power / total_power
            theta_ratio = theta_power / total_power

            # For in-ear EEG, focus on beta ratio increase and theta ratio decrease
            attention_score2 = (beta_ratio * 3) - (theta_ratio * 1.5) + (alpha_ratio * 0.5)
        else:
            attention_score2 = 0

        # Method 3: Beta/Theta ratio with adaptive baseline
        if theta_power > 0:
            bt_ratio = beta_power / theta_power
            bt_baseline = beta_baseline / (theta_baseline + 1e-6)
            bt_change = (bt_ratio - bt_baseline) / (bt_baseline + 1e-6)
            attention_score3 = bt_change
        else:
            attention_score3 = 0

        # Method 4: High-frequency vs low-frequency power
        high_freq_power = beta_power
        low_freq_power = theta_power + alpha_power * 0.5
        if low_freq_power > 0:
            hf_lf_ratio = high_freq_power / low_freq_power
            hf_lf_baseline = beta_baseline / (theta_baseline + alpha_baseline * 0.5 + 1e-6)
            attention_score4 = (hf_lf_ratio - hf_lf_baseline) / (hf_lf_baseline + 1e-6)
        else:
            attention_score4 = 0

        # Combine all methods with weights
        combined_score = (
                attention_score1 * 0.3 +  # Z-score based
                attention_score2 * 0.3 +  # Ratio based
                attention_score3 * 0.2 +  # Beta/Theta ratio
                attention_score4 * 0.2  # High/Low frequency ratio
        )

        # Convert to 0-1 range using adaptive sigmoid
        engagement = 1 / (1 + np.exp(-2.0 * combined_score))

        print(
            f"Engagement Components - Z:{attention_score1:.2f}, Ratio:{attention_score2:.2f}, BT:{attention_score3:.2f}, HFLF:{attention_score4:.2f} -> {engagement:.2f}")

        return engagement

    def _band_power(self, freqs: np.ndarray, psd: np.ndarray, band: tuple) -> float:
        """Calculate power in frequency band"""
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(psd[band_mask], freqs[band_mask])

    def get_engagement_change(self) -> float:
        """Get recent change in engagement (-1 to 1)"""
        if len(self.engagement_history) < 5:
            return 0.0

        recent_avg = np.mean(list(self.engagement_history)[-3:])
        baseline_avg = np.mean(list(self.engagement_history)[-10:-3])

        change = recent_avg - baseline_avg
        return float(np.clip(change * 5, -1, 1))

    def save_engagement_plot(self, filename: str = "engagement_plot.png"):
        """Save final engagement plot as PNG"""
        if len(self.engagement_history) < 10:
            print("Not enough data for plot")
            return

        plt.figure(figsize=(12, 8))

        # Main engagement plot
        plt.subplot(2, 1, 1)
        plt.plot(list(self.timestamps), list(self.engagement_history), 'b-', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Engagement Level')
        plt.title('EEG Engagement Over Time (Adaptive for In-Ear EEG)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # Add average line
        avg_engagement = np.mean(self.engagement_history)
        plt.axhline(y=avg_engagement, color='r', linestyle='--',
                    label=f'Average: {avg_engagement:.2f}')
        plt.legend()

        # Band power evolution (if we have baseline data)
        if len(self.baseline_alpha) > 10:
            plt.subplot(2, 1, 2)
            time_baseline = np.linspace(0, list(self.timestamps)[-1], len(self.baseline_alpha))

            # Normalize band powers for visualization
            alpha_norm = np.array(self.baseline_alpha) / np.mean(self.baseline_alpha)
            beta_norm = np.array(self.baseline_beta) / np.mean(self.baseline_beta)
            theta_norm = np.array(self.baseline_theta) / np.mean(self.baseline_theta)

            plt.plot(time_baseline, alpha_norm, 'g-', label='Alpha (normalized)', alpha=0.7)
            plt.plot(time_baseline, beta_norm, 'b-', label='Beta (normalized)', alpha=0.7)
            plt.plot(time_baseline, theta_norm, 'r-', label='Theta (normalized)', alpha=0.7)

            plt.xlabel('Time (seconds)')
            plt.ylabel('Normalized Band Power')
            plt.title('EEG Band Power Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Don't show, just save
        print(f"📊 Engagement plot saved as {filename}")