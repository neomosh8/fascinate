# eeg_engagement.py
import numpy as np
from collections import deque
from scipy import signal
import asyncio
from typing import Optional, Callable
import time
import matplotlib.pyplot as plt


class EngagementProcessor:
    """Real-time EEG engagement/attention calculation - NO REAL-TIME PLOTTING"""

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
        """Calculate engagement/attention metric from EEG data"""
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
        print(alpha_power,beta_power,theta_power)
        # Engagement metric: beta/(alpha + theta) ratio
        if alpha_power + theta_power > 0:
            engagement_raw = beta_power / (alpha_power + theta_power)
            engagement = 1 / (1 + np.exp(-0.5 * (engagement_raw - 2)))
        else:
            engagement = 0.5

        return float(np.clip(engagement, 0, 1))

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

        plt.figure(figsize=(12, 6))
        plt.plot(list(self.timestamps), list(self.engagement_history), 'b-', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Engagement Level')
        plt.title('EEG Engagement Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # Add average line
        avg_engagement = np.mean(self.engagement_history)
        plt.axhline(y=avg_engagement, color='r', linestyle='--',
                    label=f'Average: {avg_engagement:.2f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Don't show, just save
        print(f"📊 Engagement plot saved as {filename}")