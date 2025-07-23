"""
Online EEG filtering with state preservation - PROVEN TO WORK
Exact copy of the successful filter from neocore_client.py
"""

import numpy as np
from scipy import signal
from typing import Tuple

class OnlineFilter:
    """Online filtering that eliminates edge artifacts and produces clean EEG signals."""

    def __init__(self, sample_rate: int):
        self.fs = sample_rate

        # Design filters - EXACT same as working version
        nyq = sample_rate / 2

        # Bandpass: 0.5-40 Hz (4th order Butterworth) - removes DC and high-freq noise
        self.bp_sos = signal.butter(4, [0.5 / nyq, 40 / nyq], btype='band', output='sos')

        # Notch: 60 Hz (2nd order) - removes power line interference
        notch_b, notch_a = signal.iirnotch(60, 30, sample_rate)
        self.notch_sos = signal.tf2sos(notch_b, notch_a)

        # Initialize filter states for both channels
        self.bp_zi_ch1 = signal.sosfilt_zi(self.bp_sos)
        self.bp_zi_ch2 = signal.sosfilt_zi(self.bp_sos)
        self.notch_zi_ch1 = signal.sosfilt_zi(self.notch_sos)
        self.notch_zi_ch2 = signal.sosfilt_zi(self.notch_sos)

        self.initialized = False

    def filter_chunk(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filtering to new data chunk while preserving filter state.
        This is the EXACT method that produced clean results.
        """

        if not self.initialized:
            # Initialize filter states with first sample to prevent transients
            self.bp_zi_ch1 *= ch1_data[0]
            self.bp_zi_ch2 *= ch2_data[0]
            self.notch_zi_ch1 *= ch1_data[0]
            self.notch_zi_ch2 *= ch2_data[0]
            self.initialized = True

        # Apply bandpass filter (removes DC drift and high-frequency noise)
        ch1_bp, self.bp_zi_ch1 = signal.sosfilt(self.bp_sos, ch1_data, zi=self.bp_zi_ch1)
        ch2_bp, self.bp_zi_ch2 = signal.sosfilt(self.bp_sos, ch2_data, zi=self.bp_zi_ch2)

        # Apply notch filter (removes 60Hz power line interference)
        ch1_filt, self.notch_zi_ch1 = signal.sosfilt(self.notch_sos, ch1_bp, zi=self.notch_zi_ch1)
        ch2_filt, self.notch_zi_ch2 = signal.sosfilt(self.notch_sos, ch2_bp, zi=self.notch_zi_ch2)

        return ch1_filt, ch2_filt

    def get_filter_info(self) -> dict:
        """Get information about the filter design."""
        return {
            'bandpass_range': '0.5-40 Hz',
            'notch_frequency': '60 Hz',
            'sample_rate': self.fs,
            'initialized': self.initialized,
            'bandpass_order': 4,
            'notch_order': 2
        }