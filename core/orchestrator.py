"""Main orchestrator with proper engagement tracking during TTS."""

import asyncio
import time
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

from config import AUTO_ADVANCE_TIMEOUT_SEC
from utils.logger import ConversationLogger
from audio.speech_to_text import SpeechToText
from audio.text_to_speech import TextToSpeech
from conversation.gpt_wrapper import GPTConversation
from eeg.device_manager import EEGDeviceManager
from eeg.engagement_scorer import EngagementScorer
from rl.contextual_bandit import ContextualBanditAgent


@dataclass
class TurnData:
    """Data for a single conversation turn."""

    user_text: str
    user_spoke: bool
    strategy: "Strategy"
    assistant_text: str
    engagement_before: float
    engagement_during_tts: List[float]  # Track engagement throughout TTS
    engagement_after: float
    reward: float
    duration: float


class ConversationOrchestrator:
    """Orchestrates the entire conversation flow with proper engagement tracking."""

    def __init__(self, ui_callbacks: Optional[dict] = None):
        # Initialize all components
        self.logger = ConversationLogger()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.gpt = GPTConversation()
        self.eeg_manager = EEGDeviceManager()
        self.engagement_scorer = EngagementScorer()
        self.event_loop = None

        # Ensure turns are processed sequentially
        self.turn_lock = asyncio.Lock()

        # Initialize Contextual Bandit agent
        self.bandit_agent = ContextualBanditAgent(context_window_size=5)

        # Load saved bandit state
        from pathlib import Path

        models_path = Path("models")
        models_path.mkdir(exist_ok=True)
        bandit_file = models_path / "contextual_bandit.pkl"
        try:
            self.bandit_agent.load(bandit_file)
            print(f"Loaded bandit state from {bandit_file}")
        except Exception as e:
            print(f"Could not load bandit state: {e}")

        # State tracking
        self.last_engagement = 0.5
        self.turn_count = 0
        self.is_running = False

        # Enhanced reward calculation
        self.engagement_history = []
        self.reward_baseline = 0.0
        self.reward_std = 1.0

        # TTS engagement tracking
        self.tts_engagement_buffer = []
        self.tts_tracking_active = False

        # Auto-advance timer
        self.auto_advance_task: Optional[asyncio.Task] = None
        self.auto_advance_timeout = AUTO_ADVANCE_TIMEOUT_SEC

        # UI callbacks
        self.ui_callbacks = ui_callbacks or {}

        # Set EEG data callback
        self.eeg_manager.data_callback = self._on_eeg_data

    def interrupt_current_turn(self):
        """Interrupt any ongoing TTS and cancel timers."""
        try:
            self.tts.stop()
        except Exception:
            pass
        self.cancel_auto_advance_timer()

    def _on_eeg_data(self, ch1_samples, ch2_samples):
        """Process incoming EEG data and track TTS engagement."""
        import numpy as np

        engagement = self.engagement_scorer.process_chunk(
            np.array(ch1_samples), np.array(ch2_samples)
        )

        # Track engagement during TTS playback
        if self.tts_tracking_active:
            self.tts_engagement_buffer.append(engagement)

        # Update UI if callback provided
        if "update_engagement" in self.ui_callbacks:
            self.ui_callbacks["update_engagement"](engagement)

    # In core/orchestrator.py, add this method after _on_eeg_data method (around line 70)

    def _calculate_tts_engagement_score(
        self, engagement_during_tts: List[float], tts_duration: float
    ) -> Dict[str, float]:
        """Calculate comprehensive TTS engagement score using windowed analysis."""

        if not engagement_during_tts or tts_duration < 0.5:
            return {
                "mean_engagement": 0.5,
                "percentile_75": 0.5,
                "peak_engagement": 0.5,
                "engagement_stability": 0.5,
                "positive_trend": 0.0,
                "final_score": 0.5,
            }

        # Convert to numpy for easier processing
        engagement_array = np.array(engagement_during_tts)

        # 1. Create 1-second windows (overlapping 50%)
        window_size = max(
            10, len(engagement_array) // max(1, int(tts_duration))
        )  # ~10 samples per second
        step_size = window_size // 2  # 50% overlap

        window_scores = []
        for i in range(0, len(engagement_array) - window_size + 1, step_size):
            window = engagement_array[i : i + window_size]
            window_mean = np.mean(window)
            window_scores.append(window_mean)

        if not window_scores:
            window_scores = [np.mean(engagement_array)]

        window_scores = np.array(window_scores)

        # 2. Calculate key metrics
        mean_engagement = np.mean(engagement_array)

        # Use 75th percentile - rewards when MOST of the time was engaging
        percentile_75 = np.percentile(window_scores, 75)

        # Peak engagement - bonus for high moments
        peak_engagement = np.max(engagement_array)

        # Stability - how consistent the engagement was
        engagement_stability = 1.0 - np.std(window_scores)
        engagement_stability = np.clip(engagement_stability, 0, 1)

        # Trend analysis - did engagement improve during speech?
        if len(engagement_array) > 5:
            # Compare first third vs last third
            first_third = np.mean(engagement_array[: len(engagement_array) // 3])
            last_third = np.mean(engagement_array[-len(engagement_array) // 3 :])
            positive_trend = last_third - first_third
        else:
            positive_trend = 0.0

        # 3. Combine into final score
        final_score = (
            0.5 * percentile_75  # Main score: 75th percentile
            + 0.2 * mean_engagement  # Overall level
            + 0.1 * min(peak_engagement, 1.0)  # Peak bonus (capped)
            + 0.1 * np.clip(positive_trend, 0, 0.5)  # Trend bonus
            + 0.1 * engagement_stability  # Stability bonus
        )

        return {
            "mean_engagement": float(mean_engagement),
            "percentile_75": float(percentile_75),
            "peak_engagement": float(peak_engagement),
            "engagement_stability": float(engagement_stability),
            "positive_trend": float(positive_trend),
            "final_score": float(np.clip(final_score, 0, 1)),
        }

    async def initialize(self, eeg_mac: Optional[str] = None) -> bool:
        """Initialize all components."""
        try:
            # Connect to EEG device
            self.logger.info("Connecting to EEG device...")
            connected = await self.eeg_manager.connect(eeg_mac)
            if not connected:
                self.logger.error("Failed to connect to EEG device")
                return False

            self.logger.info("Successfully initialized")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def _calculate_adaptive_reward(
        self,
        engagement_before: float,
        engagement_during_tts: List[float],
        tts_duration: float,
        user_spoke: bool,
        session_duration: float,
    ) -> float:
        """Calculate reward using windowed TTS engagement analysis."""

        # Get comprehensive TTS engagement analysis
        tts_analysis = self._calculate_tts_engagement_score(
            engagement_during_tts, tts_duration
        )

        # Use the sophisticated TTS score instead of simple average
        engagement_after = tts_analysis["final_score"]
        engagement_delta = engagement_after - engagement_before

        # Store engagement history for adaptive normalization
        self.engagement_history.append(engagement_delta)
        if len(self.engagement_history) > 50:
            self.engagement_history.pop(0)

        # Adaptive baseline
        if len(self.engagement_history) > 10:
            self.reward_baseline = np.mean(self.engagement_history)
            self.reward_std = max(np.std(self.engagement_history), 0.01)

        # Base reward from engagement change
        normalized_delta = (engagement_delta - self.reward_baseline) / self.reward_std
        reward = normalized_delta

        # Bonuses based on TTS analysis

        # 1. High overall engagement bonus
        if tts_analysis["percentile_75"] > 0.6:
            reward += 0.3 * (tts_analysis["percentile_75"] - 0.6)

        # 2. Peak engagement bonus (rewards exciting moments)
        if tts_analysis["peak_engagement"] > 0.8:
            reward += 0.2 * (tts_analysis["peak_engagement"] - 0.8)

        # 3. Positive trend bonus (engagement improved during speech)
        if tts_analysis["positive_trend"] > 0.05:
            reward += 0.2 * min(tts_analysis["positive_trend"], 0.3)

        # 4. Stability bonus (consistent engagement)
        if tts_analysis["engagement_stability"] > 0.7:
            reward += 0.15 * (tts_analysis["engagement_stability"] - 0.7)

        # 5. Penalties

        # Penalty for low 75th percentile (most of the time was boring)
        if tts_analysis["percentile_75"] < 0.4:
            reward -= 0.3 * (0.4 - tts_analysis["percentile_75"])

        # Penalty for negative trend (lost attention during speech)
        if tts_analysis["positive_trend"] < -0.05:
            reward -= 0.2 * abs(tts_analysis["positive_trend"])

        # User interaction bonus
        if user_spoke:
            reward += 0.2

        # Session progression
        progression_multiplier = min(1.0 + session_duration / 300, 1.5)
        reward *= progression_multiplier

        # Clip final reward
        reward = np.clip(reward, -2.0, 2.0)

        return reward

    async def _speak_with_engagement_tracking(
        self, text: str, strategy
    ) -> Tuple[float, float, List[float]]:
        """Speak text while tracking engagement throughout."""

        # Clear TTS engagement buffer and start tracking
        self.tts_engagement_buffer = []
        self.tts_tracking_active = True

        try:
            # Get TTS start time and engagement
            tts_start = time.time()
            start_engagement = self.engagement_scorer.current_engagement

            # Speak with TTS (this will trigger engagement tracking via _on_eeg_data)
            await self.tts.speak(text, strategy)

            tts_end = time.time()

            # Stop tracking and get the engagement buffer
            self.tts_tracking_active = False
            engagement_during_tts = list(self.tts_engagement_buffer)

            self.logger.info(
                f"TTS engagement tracking: {len(engagement_during_tts)} samples over {tts_end - tts_start:.1f}s"
            )
            if engagement_during_tts:
                self.logger.info(
                    f"Engagement range: {min(engagement_during_tts):.3f} - {max(engagement_during_tts):.3f}"
                )

            return tts_start, tts_end, engagement_during_tts

        except Exception as e:
            self.logger.error(f"Error during TTS engagement tracking: {e}")
            self.tts_tracking_active = False
            return time.time(), time.time(), []

    async def process_turn(self, audio_data: bytes) -> TurnData:
        """Process a single conversation turn with proper engagement tracking."""
        async with self.turn_lock:
            # Cancel any auto-advance timer
            self.cancel_auto_advance_timer()

            turn_start = time.time()

            # 1. Transcribe user speech
            if audio_data:
                self.logger.info("Transcribing user speech...")
                user_text = await self.stt.transcribe(audio_data)
            else:
                user_text = ""
            user_spoke = len(user_text.strip()) > 0

            if "update_transcript" in self.ui_callbacks:
                self.ui_callbacks["update_transcript"](
                    f"User: {user_text if user_spoke else '[Silent]'}"
                )

            # 2. Get current engagement (before response)
            engagement_before = self.engagement_scorer.current_engagement

            # Build context vector before selecting strategy
            context_vector = self.bandit_agent._build_context_vector()

            # 3. Bandit agent selects strategy
            strategy = self.bandit_agent.select_strategy()

            self.logger.info(
                f"Selected strategy: {strategy.tone}/{strategy.topic}/{strategy.emotion}/{strategy.hook}"
            )

            # Update UI immediately to show selected strategy before TTS
            if "update_strategy" in self.ui_callbacks:
                self.ui_callbacks["update_strategy"]((strategy, None))

            # 4. Generate GPT response
            self.logger.info("Generating response...")
            assistant_text = await self.gpt.generate_response(user_text, strategy)

            if "update_transcript" in self.ui_callbacks:
                self.ui_callbacks["update_transcript"](f"Assistant: {assistant_text}")

            # 5. Speak response with proper engagement tracking
            self.logger.info("Speaking response with engagement tracking...")
            tts_start, tts_end, engagement_during_tts = (
                await self._speak_with_engagement_tracking(assistant_text, strategy)
            )

            # 6. Calculate reward using proper TTS engagement data
            session_duration = time.time() - getattr(
                self, "session_start_time", time.time()
            )
            reward = self._calculate_adaptive_reward(
                engagement_before,
                engagement_during_tts,
                tts_end - tts_start,
                user_spoke,
                session_duration,
            )

            # Calculate final engagement for logging
            engagement_after = (
                np.mean(engagement_during_tts)
                if engagement_during_tts
                else self.engagement_scorer.current_engagement
            )

            # Let strategy learn from this example
            strategy.add_example(assistant_text, engagement_after - engagement_before)

            # 7. Update bandit agent
            self.bandit_agent.update(strategy, context_vector, reward)

            # Add turn to context history
            self.bandit_agent.context.add_turn(
                user_text, assistant_text, strategy, engagement_after
            )

            # Send update to UI for visualization
            if "update_strategy" in self.ui_callbacks:
                self.ui_callbacks["update_strategy"]((strategy, reward))

            # Update state
            self.last_engagement = engagement_after
            self.turn_count += 1

            # Start auto-advance timer
            self.start_auto_advance_timer()

            # Create turn data
            turn_data = TurnData(
                user_text=user_text,
                user_spoke=user_spoke,
                strategy=strategy,
                assistant_text=assistant_text,
                engagement_before=engagement_before,
                engagement_during_tts=engagement_during_tts,
                engagement_after=engagement_after,
                reward=reward,
                duration=time.time() - turn_start,
            )

            # Enhanced logging
            self.logger.log_turn(
                {
                    "turn": self.turn_count,
                    "user_text": user_text,
                    "user_spoke": user_spoke,
                    "strategy": strategy.to_prompt(),
                    "assistant_text": assistant_text,
                    "engagement_before": engagement_before,
                    "engagement_during_tts_samples": len(engagement_during_tts),
                    "tts_analysis": self._calculate_tts_engagement_score(
                        engagement_during_tts, tts_end - tts_start
                    ),
                    # ADD THIS LINE
                    "engagement_after": engagement_after,
                    "reward": reward,
                    "duration": turn_data.duration,
                    "tts_duration": tts_end - tts_start,
                }
            )

            return turn_data

    # [Rest of the methods remain the same as before...]

    async def run_session(self, client):
        """Run the main conversation session."""
        try:
            self.session_start_time = time.time()
            self.event_loop = asyncio.get_event_loop()

            # Start EEG streaming
            await self.eeg_manager.start_streaming(client)
            self.is_running = True

            # Initial greeting with engagement tracking
            strategy = self.bandit_agent.select_strategy()
            greeting = await self.gpt.generate_response("", strategy)

            if "update_transcript" in self.ui_callbacks:
                self.ui_callbacks["update_transcript"](f"Assistant: {greeting}")

            await self._speak_with_engagement_tracking(greeting, strategy)

            # Start auto-advance timer after greeting
            self.start_auto_advance_timer()

            # Main conversation loop
            while self.is_running:
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Session error: {e}")
        finally:
            await self.eeg_manager.stop_streaming()

    def start_auto_advance_timer(self):
        """Start a timer to auto-advance the conversation."""
        if self.auto_advance_task and not self.auto_advance_task.done():
            self.auto_advance_task.cancel()

        if self.is_running:
            self.auto_advance_task = asyncio.create_task(self._auto_advance_timer())

    def cancel_auto_advance_timer(self):
        """Cancel any running auto-advance timer."""
        if self.auto_advance_task and not self.auto_advance_task.done():
            self.auto_advance_task.cancel()
            self.auto_advance_task = None

    async def _auto_advance_timer(self):
        """Wait for the timeout and queue a silent turn."""
        try:
            for seconds_left in range(self.auto_advance_timeout, 0, -1):
                if "update_countdown" in self.ui_callbacks:
                    self.ui_callbacks["update_countdown"](seconds_left)
                await asyncio.sleep(1)

            if "update_countdown" in self.ui_callbacks:
                self.ui_callbacks["update_countdown"](0)

            self.logger.info("Auto-advancing conversation (user silent)")
            empty_audio = b""
            asyncio.create_task(self.process_turn(empty_audio))
        except asyncio.CancelledError:
            if "update_countdown" in self.ui_callbacks:
                self.ui_callbacks["update_countdown"](0)
        finally:
            self.auto_advance_task = None

    def stop(self):
        """Stop the conversation session."""
        self.is_running = False
        self.cancel_auto_advance_timer()
        self.tts.stop()

    def cleanup(self):
        """Clean up resources."""
        self.cancel_auto_advance_timer()
        if self.turn_count > 0:
            self.print_session_summary()

        self.stt.cleanup()
        self.tts.cleanup()

        # Save bandit agent
        from pathlib import Path

        save_path = Path("models")
        save_path.mkdir(exist_ok=True)

        try:
            self.bandit_agent.save(save_path / "contextual_bandit.pkl")
            print("Bandit state saved successfully")
        except Exception as e:
            print(f"Failed to save bandit state: {e}")

        # Save session summary
        summary = self.get_session_summary()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = save_path / f"session_summary_{timestamp}.json"

        import json

        def serialize_object(obj):
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            elif hasattr(obj, "tone"):
                return {
                    "tone": obj.tone,
                    "topic": obj.topic,
                    "emotion": obj.emotion,
                    "hook": obj.hook,
                    "index": obj.index,
                }
            return str(obj)

        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=serialize_object)
            print(f"Session summary saved to: {summary_path}")
        except Exception as e:
            print(f"Failed to save session summary: {e}")

    def get_session_summary(self) -> Dict:
        """Generate comprehensive session summary."""
        try:
            performance_summary = self.bandit_agent.get_performance_summary()
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            performance_summary = {"error": str(e)}

        return {
            "session_info": {
                "total_turns": self.turn_count,
                "session_duration": time.time()
                - getattr(self, "session_start_time", time.time()),
                "final_engagement": self.last_engagement,
            },
            "bandit_performance": performance_summary,
            "engagement_stats": {
                "current_engagement": self.engagement_scorer.current_engagement,
                "baseline_collected": getattr(
                    self.engagement_scorer, "baseline_collected", False
                ),
                "adaptive_baseline_initialized": getattr(
                    self.engagement_scorer, "baseline_initialized", False
                ),
            },
        }

    def print_session_summary(self):
        """Print detailed session summary to console."""
        summary = self.get_session_summary()

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)

        # Session info
        session_info = summary["session_info"]
        print(f"Total Turns: {session_info['total_turns']}")
        print(f"Session Duration: {session_info['session_duration']:.1f} seconds")
        print(f"Final Engagement: {session_info['final_engagement']:.3f}")

        # Bandit Performance
        rl_perf = summary["bandit_performance"]
        if "error" not in rl_perf:
            print(
                f"\nAVERAGE RECENT REWARD: {rl_perf.get('average_recent_reward', 'N/A')}"
            )

            # Component analysis
            print("\n🎯 COMPONENT PERFORMANCE:")
            for component, data in rl_perf.get("components", {}).items():
                print(f"\n📊 {component.upper()}:")
                print(
                    f"   Best Choice: {data.get('best_choice', 'N/A')} (score: {data.get('best_score', 0):.3f})"
                )
                usage_stats = data.get("usage_stats", {})
                sorted_arms = sorted(
                    usage_stats.items(),
                    key=lambda x: x[1].get("average_reward", 0),
                    reverse=True,
                )
                print("   Top Performers:")
                for i, (arm, stats) in enumerate(sorted_arms[:3]):
                    recent_avg = stats.get(
                        "recent_average", stats.get("average_reward", 0)
                    )
                    print(
                        f"     {i+1}. {arm}: {recent_avg:.3f} recent avg ({stats.get('usage_count', 0)} uses)"
                    )

            restart_stats = rl_perf.get("restart_stats", {})
            print("\n🔄 ADAPTIVE RESTARTS:")
            print(f"   Total Restarts: {restart_stats.get('total_restarts', 0)}")
            print(f"   Last Restart: Step {restart_stats.get('last_restart_step', 0)}")
        else:
            print(f"\nRL Performance Error: {rl_perf['error']}")

        print("=" * 60)
