# engagement_tracker.py
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np


@dataclass
class StrategySession:
    """Track a strategy and its associated audio playback"""
    strategy_index: int
    strategy: object  # CommunicationStrategy
    start_time: float
    audio_start_time: Optional[float] = None
    audio_end_time: Optional[float] = None
    engagement_measurements: list = None

    def __post_init__(self):
        if self.engagement_measurements is None:
            self.engagement_measurements = []


class TimeAlignedEngagementTracker:
    """Tracks engagement and aligns it with the correct strategy timing"""

    def __init__(self, audio_delay_estimate: float = 0.5):
        self.audio_delay_estimate = audio_delay_estimate  # Estimated system audio delay

        # Track strategy sessions
        self.strategy_sessions = deque(maxlen=10)  # Keep last 10 sessions
        self.current_session: Optional[StrategySession] = None

        # Engagement measurements with timestamps
        self.engagement_buffer = deque(maxlen=1000)  # (timestamp, engagement_value)

        # Callbacks
        self.rl_update_callback: Optional[Callable] = None

        # Audio playback tracking
        self.audio_chunks_playing = 0
        self.last_audio_chunk_time = 0

    def set_rl_update_callback(self, callback: Callable):
        """Set callback for when we have properly aligned engagement data"""
        self.rl_update_callback = callback

    def start_new_strategy(self, strategy_index: int, strategy: object):
        """Start tracking a new strategy"""
        current_time = time.time()

        # Finalize previous session if exists
        if self.current_session:
            self._finalize_session(self.current_session)

        # Start new session
        self.current_session = StrategySession(
            strategy_index=strategy_index,
            strategy=strategy,
            start_time=current_time
        )

        print(f"🎯 Started tracking strategy {strategy_index} at {current_time:.2f}")

    def on_audio_chunk_received(self):
        """Called when AI audio chunk is received"""
        current_time = time.time()
        self.last_audio_chunk_time = current_time

        if self.current_session and self.current_session.audio_start_time is None:
            # First audio chunk - mark audio start
            self.current_session.audio_start_time = current_time + self.audio_delay_estimate
            print(f"🔊 Audio playback started for strategy {self.current_session.strategy_index} (estimated)")

        self.audio_chunks_playing += 1

    def on_audio_playback_complete(self):
        """Called when AI finishes speaking"""
        current_time = time.time()

        if self.current_session and self.current_session.audio_start_time:
            # Audio finished - mark end time
            self.current_session.audio_end_time = current_time
            print(f"🔊 Audio playback ended for strategy {self.current_session.strategy_index}")

            # Process accumulated engagement for this session
            self._process_engagement_for_session(self.current_session)

    def add_engagement_measurement(self, engagement: float):
        """Add engagement measurement with timestamp"""
        current_time = time.time()
        self.engagement_buffer.append((current_time, engagement))

        # Clean old measurements (older than 60 seconds)
        cutoff_time = current_time - 60
        while self.engagement_buffer and self.engagement_buffer[0][0] < cutoff_time:
            self.engagement_buffer.popleft()

    def _process_engagement_for_session(self, session: StrategySession):
        """Process engagement measurements for a completed session"""
        if not session.audio_start_time or not session.audio_end_time:
            return

        # Find engagement measurements that occurred during this session's audio
        session_engagements = []

        for timestamp, engagement in self.engagement_buffer:
            if session.audio_start_time <= timestamp <= session.audio_end_time:
                session_engagements.append(engagement)

        if session_engagements:
            session.engagement_measurements = session_engagements

            # Calculate average engagement for this session
            avg_engagement = np.mean(session_engagements)
            engagement_change = self._calculate_engagement_change(session_engagements)

            print(f"📊 Strategy {session.strategy_index} engagement: {avg_engagement:.2f} "
                  f"(change: {engagement_change:+.2f}) from {len(session_engagements)} measurements")

            # Now we can safely update the RL agent with correctly aligned data
            if self.rl_update_callback:
                self.rl_update_callback(
                    strategy_index=session.strategy_index,
                    engagement=avg_engagement,
                    engagement_change=engagement_change,
                    session=session
                )
        else:
            print(f"⚠️ No engagement measurements found for strategy {session.strategy_index}")

    def _calculate_engagement_change(self, engagements: list) -> float:
        """Calculate engagement change within a session"""
        if len(engagements) < 3:
            return 0.0

        # Compare first third vs last third
        n = len(engagements)
        first_third = engagements[:n // 3]
        last_third = engagements[-n // 3:]

        if first_third and last_third:
            change = np.mean(last_third) - np.mean(first_third)
            return float(np.clip(change * 5, -1, 1))  # Scale and clip

        return 0.0

    def _finalize_session(self, session: StrategySession):
        """Finalize a strategy session"""
        if session.audio_end_time is None:
            # Audio never ended properly, estimate end time
            if session.audio_start_time:
                session.audio_end_time = time.time()
            else:
                # Audio never started, skip this session
                return

        # Process this session
        self._process_engagement_for_session(session)

        # Add to history
        self.strategy_sessions.append(session)

    def get_session_history(self):
        """Get history of completed sessions"""
        return list(self.strategy_sessions)

    def estimate_audio_delay(self):
        """Estimate system audio delay based on timing"""
        # This could be enhanced to dynamically estimate delay
        return self.audio_delay_estimate