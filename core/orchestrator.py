"""Main orchestrator with proper engagement tracking during TTS."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

from config import AUTO_ADVANCE_TIMEOUT_SEC
from therapy.session_manager import TherapeuticSessionManager
from utils.logger import ConversationLogger
from audio.speech_to_text import SpeechToText
from audio.text_to_speech import TextToSpeech
from conversation.gpt_wrapper import GPTConversation
from eeg.device_manager import EEGDeviceManager
from eeg.turn_based_engagement_scorer import TurnBasedEngagementScorer
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
    emotion_before: float
    emotion_after: float
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
        self.engagement_scorer = TurnBasedEngagementScorer()
        self.event_loop = None

        # Therapeutic session management
        self.therapeutic_manager = TherapeuticSessionManager()
        self.therapy_mode = True
        self.last_emotion = 0.5

        # Initialize Contextual Bandit agent
        self.bandit_agent = ContextualBanditAgent(context_window_size=50)
        self.bandit_agent.set_strategy_space(self.therapy_mode)

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

        # Track if a turn is actively being processed
        self.turn_in_progress = False
        self.ai_speaking = False

        # UI callbacks
        self.ui_callbacks = ui_callbacks or {}

        # Set EEG data callback
        self.eeg_manager.data_callback = self._on_eeg_data

    def _on_eeg_data(self, ch1_samples, ch2_samples):
        """Process incoming EEG data for turn-based engagement tracking."""
        # Add data to current turn if tracking is active
        self.engagement_scorer.add_eeg_chunk(ch1_samples, ch2_samples)

        # Update UI if callback provided (use current engagement)
        if "update_engagement" in self.ui_callbacks:
            current_engagement = self.engagement_scorer.get_current_engagement()
            self.ui_callbacks["update_engagement"](current_engagement)

        if "update_emotion" in self.ui_callbacks:
            current_emotion = self.engagement_scorer.get_current_emotion()
            self.ui_callbacks["update_emotion"](current_emotion)

        if "update_eeg" in self.ui_callbacks:
            self.ui_callbacks["update_eeg"]((ch1_samples, ch2_samples))

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
            engagement_after: float,
            emotion_before: float,  # NEW parameter
            emotion_after: float,  # NEW parameter
            tts_duration: float,
            user_spoke: bool,
            session_duration: float,
            user_text: str,
            assistant_text: str,
            context_type: str = "normal",
    ) -> float:
        """
        Calculate an adaptive reward that considers both engagement and emotion.
        It rewards productive focus (high engagement + neutral/positive emotion)
        and penalizes agitation (high engagement + negative emotion).
        """

        # --- Part 1: Core Engagement Reward (same as before) ---
        engagement_delta = engagement_after - engagement_before
        reward = engagement_delta * 10.0  # Scale to meaningful range

        # Absolute engagement level bonuses/penalties
        if engagement_after > 0.7:
            reward += 0.5  # High engagement bonus
        elif engagement_after > 0.5:
            reward += 0.2  # Medium engagement bonus

        if engagement_after < 0.3:
            reward -= 0.5  # Low engagement penalty
        elif engagement_after < 0.35:
            reward -= 0.2  # Below-medium penalty

        # --- Part 2: Emotional Quality Modulation (The New Logic) ---

        # We define "agitation" as an increase in engagement while emotion becomes negative.
        # We define "flow/insight" as an increase in engagement while emotion becomes positive.
        # We define "soothing/regulation" as moving from a negative to a neutral/positive state.

        # A) Penalize Agitation: High engagement is bad if the user feels bad.
        if engagement_after > 0.6 and emotion_after < 0.4:
            # This is the key signature of being provoked or distressed.
            # We heavily penalize this state.
            reward -= 1.5  # Strong penalty for creating a negative, high-arousal state.
            print(f"📉 Penalty: Agitation detected (Eng: {engagement_after:.2f}, Emo: {emotion_after:.2f})")

        # B) Reward Productive Flow/Insight: High engagement is great if the user feels good.
        if engagement_after > 0.7 and emotion_after > 0.6:
            # This is the ideal "flow" state for deep, productive work.
            reward += 1.0  # Strong bonus for achieving a positive, focused state.
            print(f"📈 Bonus: Flow/Insight detected (Eng: {engagement_after:.2f}, Emo: {emotion_after:.2f})")

        # C) Reward Successful Co-Regulation:
        if emotion_before < 0.45 and emotion_after > 0.5:
            # The agent successfully soothed the user or helped them regulate.
            # This is a highly valuable therapeutic action.
            reward += 1.2  # Very strong bonus for emotional regulation.
            print(f"📈 Bonus: Successful emotional regulation (Emo: {emotion_before:.2f} -> {emotion_after:.2f})")

        # --- Part 3: Interaction & Contextual Bonuses (same as before) ---

        # User interaction bonus
        if user_spoke:
            response_quality = self._assess_response_quality(user_text, assistant_text)
            reward += 0.15 * response_quality

        # Context-specific bonuses
        if context_type == "auto_advance" and engagement_after > 0.6:
            reward += 0.3
        elif context_type == "cold_start" and engagement_after > 0.4:
            reward += 0.2

        # Store delta for debugging (can be removed later)
        if not hasattr(self, 'delta_history'):
            self.delta_history = []
        self.delta_history.append(engagement_delta)
        if len(self.delta_history) > 20:
            self.delta_history.pop(0)

        print(f"🏆 Final Reward: {reward:.3f}")
        return reward

    def _assess_response_quality(self, user_text: str, assistant_text: str) -> float:
        """Simple heuristic to judge response quality."""
        if not assistant_text.strip():
            return 0.0
        if '?' in user_text and '?' not in assistant_text:
            return 0.5
        return 1.0

    async def _speak_with_engagement_tracking(
            self, text: str, strategy,
            user_emotion: float, user_engagement: float
    ) -> Tuple[float, float, float, float]:
        """Speak with emotion-adaptive TTS and track results."""

        try:
            self.engagement_scorer.start_turn()
            self.ai_speaking = True

            engagement_before = self.engagement_scorer.get_current_engagement()
            emotion_before = self.engagement_scorer.get_current_emotion()

            tts_start = time.time()

            try:
                await self.tts.speak(
                    text,
                    strategy,
                    user_emotion,
                    user_engagement,
                    voice="Ava Song",
                    streaming_mode="true_streaming",
                )
                tts_end = time.time()
                tts_duration = tts_end - tts_start

                # Normal turn ending
                engagement_after, emotion_after = self.engagement_scorer.end_turn(tts_duration)

            except Exception as e:
                tts_end = time.time()
                tts_duration = tts_end - tts_start

                if getattr(self.tts, "interrupted", False):
                    self.logger.info("TTS was interrupted, collecting partial metrics")
                    engagement_after, emotion_after = self.engagement_scorer.end_turn_early(tts_duration)
                else:
                    self.logger.error(f"TTS error: {e}")
                    engagement_after, emotion_after = self.engagement_scorer.end_turn(tts_duration)

            self.logger.info(
                f"Adaptive TTS: {engagement_before:.3f}->{engagement_after:.3f}, "
                f"emotion {emotion_before:.3f}->{emotion_after:.3f}"
            )

            return tts_start, tts_end, engagement_after, emotion_after

        except Exception as e:
            self.logger.error(f"Error during TTS engagement tracking: {e}")
            return time.time(), time.time(), self.engagement_scorer.get_current_engagement(), self.engagement_scorer.get_current_emotion()
        finally:
            self.ai_speaking = False

    def interrupt_ai_speech(self):
        """Interrupt AI speech and collect metrics properly."""
        if self.ai_speaking:
            self.logger.info("Interrupting AI speech...")
            self.ai_speaking = False  # Clear flag immediately
            self.tts.stop()

    async def process_turn(self, audio_data: bytes) -> TurnData:
        """Process a single conversation turn with turn-based engagement tracking."""

        # Prevent concurrent turn processing
        if self.turn_in_progress:
            self.logger.warning("Turn already in progress, skipping")
            return

        self.turn_in_progress = True
        try:
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
                await asyncio.sleep(0.2)
    
            # 2. Get current engagement and emotion before response
            engagement_before = self.engagement_scorer.get_current_engagement()
            emotion_before = self.engagement_scorer.get_current_emotion()
    
            if self.therapy_mode:
                thera_context = self.therapeutic_manager.get_session_summary()
                session_phase = thera_context["session_phase"]
                target_concept = thera_context["current_target"]
            else:
                session_phase = "non_therapeutic"
                target_concept = None

            # Build context object before selecting strategy
            context_object = self.bandit_agent._build_context_vector(session_phase, target_concept)

            print("context vector:", context_object)

            # 3. Bandit agent selects strategy
            strategy = self.bandit_agent.select_strategy(
                session_phase=session_phase,
                target_concept=target_concept,
            )
    
            self.logger.info(
                f"Selected strategy: {strategy.tone}/{strategy.topic}/{strategy.emotion}/{strategy.hook}"
            )
    
            # Update UI immediately to show selected strategy before TTS
            if "update_strategy" in self.ui_callbacks:
                self.ui_callbacks["update_strategy"]((strategy, None))
                await asyncio.sleep(0.5)  # Small delay to ensure update is processed
    
            # 4. Generate GPT response with current engagement and emotion for risk detection
            self.logger.info("Generating response...")
            assistant_text = await self.gpt.generate_response(
                user_text,
                strategy,
                turn_count=self.turn_count + 1,
                current_engagement=engagement_before,  # Pass current engagement
                current_emotion=emotion_before,  # Pass current emotion
            )
    
            if "update_transcript" in self.ui_callbacks:
                self.ui_callbacks["update_transcript"](f"Assistant: {assistant_text}")
    
            # 5. Speak response with turn-based engagement tracking
            self.logger.info("Speaking response with turn-based engagement tracking...")
            tts_start, tts_end, engagement_after, emotion_after = (
                await self._speak_with_engagement_tracking(
                    assistant_text, strategy, emotion_before, engagement_before
                )
            )
    
            # 6. Calculate reward using turn-based engagement data
            session_duration = time.time() - getattr(
                self, "session_start_time", time.time()
            )
            reward = self._calculate_adaptive_reward(
                engagement_before=engagement_before,
                engagement_after=engagement_after,
                emotion_before=emotion_before,        # PASS the new parameter
                emotion_after=emotion_after,          # PASS the new parameter
                tts_duration=tts_end - tts_start,
                user_spoke=user_spoke,
                session_duration=session_duration,
                user_text=user_text,
                assistant_text=assistant_text,
                context_type=self.bandit_agent.classify_current_context(), # Use new classify method
            )

            if self.therapy_mode:
                therapeutic_analysis = await self.therapeutic_manager.process_therapeutic_turn(
                    user_text, assistant_text, engagement_after, emotion_after  # ADD emotion_after
                )
    
                print(f"📊 Concepts found: {therapeutic_analysis['concepts_found']}")
                print(f"🎯 Hot concepts: {therapeutic_analysis['hot_concepts']}")
                print(
                    f"🔄 Session phase: {therapeutic_analysis['session_phase']} (turn {therapeutic_analysis['turn_in_phase']})")
    
    
    
            # Let strategy learn from this example
            strategy.add_example(assistant_text, engagement_after - engagement_before)
    
            # 7. Update bandit agent
            self.bandit_agent.update(strategy, context_object, reward)
    
            # Add turn to context history
            self.bandit_agent.context.add_turn(
                user_text, assistant_text, strategy, engagement_after
            )
    
            # Send FINAL update to UI with reward
            if "update_strategy" in self.ui_callbacks:
                self.ui_callbacks["update_strategy"]((strategy, reward))
                self.logger.info(f"UI update sent: strategy={strategy.tone}, reward={reward:.3f}")
                await asyncio.sleep(0.5)  # Small delay to ensure update is processed
    
    
            # Update state
            self.last_engagement = engagement_after
            self.last_emotion = emotion_after
            self.turn_count += 1
    
            # Create turn data (updated for turn-based approach)
            turn_data = TurnData(
                user_text=user_text,
                user_spoke=user_spoke,
                strategy=strategy,
                assistant_text=assistant_text,
                engagement_before=engagement_before,
                engagement_during_tts=[engagement_after],  # Single value in list for compatibility
                engagement_after=engagement_after,
                emotion_before=emotion_before,
                emotion_after=emotion_after,
                reward=reward,
                duration=time.time() - turn_start,
            )
    
    
            # Enhanced logging (updated for turn-based approach)
            # Enhanced logging (updated for turn-based approach)
            self.logger.log_turn(
                {
                    "turn": self.turn_count,
                    "user_text": user_text,
                    "user_spoke": user_spoke,
                    "strategy": strategy.to_prompt(),
                    "assistant_text": assistant_text,
                    "engagement_before": engagement_before,
                    "engagement_after": engagement_after,
                    "engagement_delta": engagement_after - engagement_before,
                    "emotion_before": emotion_before,  # NEW
                    "emotion_after": emotion_after,  # NEW
                    "emotion_delta": emotion_after - emotion_before,  # NEW
                    "reward": reward,
                    "duration": turn_data.duration,
                    "tts_duration": tts_end - tts_start,
                    "turn_based_scoring": True,  # Flag to indicate new scoring method
                }
            )
    
    
            return turn_data
        
        finally:
            # Clear turn-in-progress flag and restart auto-advance timer
            self.turn_in_progress = False
            self.start_auto_advance_timer()

    async def run_session(self, client):
        """Run the main conversation session."""
        try:
            self.session_start_time = time.time()
            self.event_loop = asyncio.get_event_loop()

            # Start EEG streaming
            await self.eeg_manager.start_streaming(client)
            self.is_running = True

            # Initial greeting with engagement tracking
            strategy = self.bandit_agent.select_strategy(
                session_phase="exploration",
                target_concept=None,
            )
            greeting = await self.gpt.generate_response("", strategy)

            if "update_transcript" in self.ui_callbacks:
                self.ui_callbacks["update_transcript"](f"Assistant: {greeting}")

            await self._speak_with_engagement_tracking(
                greeting, strategy, 0.5, 0.5
            )

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
                await asyncio.sleep(0.1)

            # Check if we can auto-advance (not interrupted or already processing)
            if (
                self.auto_advance_task
                and not self.auto_advance_task.cancelled()
                and not self.turn_in_progress
                and not self.ai_speaking
            ):
                self.logger.info("Auto-advancing conversation (user silent)")
                # Create task instead of await to prevent blocking
                asyncio.create_task(self.process_turn(b""))

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
            self.save_session_visualizations()  # Add this line
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

    def save_session_visualizations(self):
        """Save session visualizations as images."""
        from pathlib import Path
        import time

        # Create visualizations directory
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)

        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save word cloud if UI has concept widget
        if hasattr(self, 'ui_callbacks') and 'save_word_cloud' in self.ui_callbacks:
            try:
                word_cloud_path = viz_dir / f"word_cloud_{timestamp}.png"
                self.ui_callbacks['save_word_cloud'](str(word_cloud_path))
            except Exception as e:
                print(f"Failed to save word cloud: {e}")

        # Save concept statistics as JSON
        if self.therapy_mode and hasattr(self, 'therapeutic_manager'):
            try:
                stats_path = viz_dir / f"concept_stats_{timestamp}.json"
                self._save_concept_statistics(stats_path)
            except Exception as e:
                print(f"Failed to save concept statistics: {e}")

    def _save_concept_statistics(self, filepath: Path):
        """Save detailed concept statistics as JSON."""
        import json

        concept_tracker = self.therapeutic_manager.concept_tracker
        stats = {
            "session_timestamp": filepath.stem.split('_')[-2:],
            "total_turns": self.turn_count,
            "concepts": {}
        }

        for concept, engagement_scores in concept_tracker.concept_activations.items():
            emotion_scores = concept_tracker.concept_emotions.get(concept, [])

            stats["concepts"][concept] = {
                "engagement_scores": engagement_scores,
                "emotion_scores": emotion_scores,
                "total_mentions": len(engagement_scores),
                "avg_engagement": float(np.mean(engagement_scores)),
                "avg_emotion": float(np.mean(emotion_scores)) if emotion_scores else 0.5,
                "emotional_intensity": float(
                    np.mean([abs(e - 0.5) * 2 for e in emotion_scores])) if emotion_scores else 0.0,
                "engagement_trend": float(engagement_scores[-1] - engagement_scores[0]) if len(
                    engagement_scores) > 1 else 0.0
            }

        # Sort by importance for easier reading
        stats["concepts"] = dict(sorted(
            stats["concepts"].items(),
            key=lambda x: x[1]["avg_engagement"] * (1 + x[1]["emotional_intensity"]),
            reverse=True
        ))

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📊 Concept statistics saved to: {filepath}")
