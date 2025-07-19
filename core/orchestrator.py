"""Main orchestrator that coordinates all components."""

import asyncio
import time
from typing import Optional, Callable, Dict
from dataclasses import dataclass

import numpy as np

from config import AUTO_ADVANCE_TIMEOUT_SEC
from utils.logger import ConversationLogger
from audio.speech_to_text import SpeechToText
from audio.text_to_speech import TextToSpeech
from conversation.gpt_wrapper import GPTConversation
from eeg.device_manager import EEGDeviceManager
from eeg.engagement_scorer import EngagementScorer
from rl.hierarchical_bandit import HierarchicalBanditAgent


@dataclass
class TurnData:
    """Data for a single conversation turn."""
    user_text: str
    user_spoke: bool
    strategy: 'Strategy'
    assistant_text: str
    engagement_before: float
    engagement_after: float
    reward: float
    duration: float


class ConversationOrchestrator:
    """Orchestrates the entire conversation flow."""

    def __init__(self, ui_callbacks: Optional[dict] = None):
        # Initialize all components
        self.logger = ConversationLogger()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.gpt = GPTConversation()
        self.eeg_manager = EEGDeviceManager()
        self.engagement_scorer = EngagementScorer()
        self.event_loop = None

        # Initialize Hierarchical Bandit agent
        self.bandit_agent = HierarchicalBanditAgent()

        # Load saved bandit state
        from pathlib import Path
        models_path = Path("models")
        models_path.mkdir(exist_ok=True)  # Ensure directory exists
        bandit_file = models_path / "hierarchical_bandit.json"
        try:
            self.bandit_agent.load(bandit_file)
            print(f"Loaded bandit state from {bandit_file}")
        except Exception as e:
            print(f"Could not load bandit state: {e}")

        # State tracking
        self.last_engagement = 0.5
        self.turn_count = 0
        self.is_running = False

        # Enhanced reward calculation for adaptive engagement
        self.engagement_history = []
        self.reward_baseline = 0.0
        self.reward_std = 1.0

        # Auto-advance timer
        self.auto_advance_task: Optional[asyncio.Task] = None
        self.auto_advance_timeout = AUTO_ADVANCE_TIMEOUT_SEC

        # UI callbacks
        self.ui_callbacks = ui_callbacks or {}

        # Set EEG data callback
        self.eeg_manager.data_callback = self._on_eeg_data

    def _on_eeg_data(self, ch1_samples, ch2_samples):
        """Process incoming EEG data."""
        import numpy as np
        engagement = self.engagement_scorer.process_chunk(
            np.array(ch1_samples),
            np.array(ch2_samples)
        )

        # Update UI if callback provided
        if 'update_engagement' in self.ui_callbacks:
            self.ui_callbacks['update_engagement'](engagement)

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

    def _calculate_adaptive_reward(self, engagement_before: float, engagement_after: float,
                                 user_spoke: bool, session_duration: float) -> float:
        """Calculate reward adapted for the new engagement scorer."""

        # Basic engagement change
        engagement_delta = engagement_after - engagement_before

        # Store engagement history for adaptive normalization
        self.engagement_history.append(engagement_delta)
        if len(self.engagement_history) > 50:  # Keep last 50 deltas
            self.engagement_history.pop(0)

        # Adaptive reward baseline and scaling
        if len(self.engagement_history) > 10:
            self.reward_baseline = np.mean(self.engagement_history)
            self.reward_std = max(np.std(self.engagement_history), 0.01)

        # Normalized engagement change (z-score)
        normalized_delta = (engagement_delta - self.reward_baseline) / self.reward_std

        # Base reward from normalized engagement change
        reward = normalized_delta

        # Bonuses and penalties

        # 1. Absolute engagement level bonus (higher engagement = better)
        if engagement_after > 0.6:
            reward += 0.3 * (engagement_after - 0.6)
        elif engagement_after < 0.4:
            reward -= 0.2 * (0.4 - engagement_after)

        # 2. User interaction bonus
        if user_spoke:
            reward += 0.2

        # 3. Session progression bonus (later improvements matter more)
        progression_multiplier = min(1.0 + session_duration / 300, 1.5)  # Up to 1.5x after 5 minutes
        reward *= progression_multiplier

        # 4. Consistency bonus (reward stable high engagement)
        if len(self.engagement_history) > 5:
            recent_engagement = [self.last_engagement, engagement_after]
            if all(e > 0.6 for e in recent_engagement[-3:]):
                reward += 0.1  # Bonus for sustained high engagement

        # 5. Large drops penalty
        if engagement_delta < -0.1:
            reward -= 0.3

        # Clip final reward
        reward = np.clip(reward, -2.0, 2.0)

        return reward

    async def process_turn(self, audio_data: bytes) -> TurnData:
        """Process a single conversation turn."""
        # Cancel any auto-advance timer
        self.cancel_auto_advance_timer()

        turn_start = time.time()

        # 1. Transcribe user speech (skip API call for silence)
        if audio_data:
            self.logger.info("Transcribing user speech...")
            user_text = await self.stt.transcribe(audio_data)
        else:
            user_text = ""
        user_spoke = len(user_text.strip()) > 0

        if 'update_transcript' in self.ui_callbacks:
            self.ui_callbacks['update_transcript'](f"User: {user_text if user_spoke else '[Silent]'}")

        # 2. Get current engagement
        engagement_before = self.engagement_scorer.current_engagement

        # 3. Bandit agent selects strategy
        strategy = self.bandit_agent.select_strategy()

        self.logger.info(
            f"Selected strategy: {strategy.tone}/{strategy.topic}/{strategy.emotion}/{strategy.hook}"
        )

        # 4. Generate GPT response
        self.logger.info("Generating response...")
        assistant_text = await self.gpt.generate_response(user_text, strategy)

        if 'update_transcript' in self.ui_callbacks:
            self.ui_callbacks['update_transcript'](f"Assistant: {assistant_text}")

        # 5. Speak response with strategy and track engagement
        self.logger.info("Speaking response...")
        tts_start, tts_end = await self.tts.speak(assistant_text, strategy)

        # 6. Get engagement during TTS
        engagement_after = self.engagement_scorer.get_segment_engagement(
            tts_start, tts_end, self.eeg_manager
        )

        # 7. Calculate adaptive reward
        session_duration = time.time() - getattr(self, 'session_start_time', time.time())
        reward = self._calculate_adaptive_reward(
            engagement_before, engagement_after, user_spoke, session_duration
        )

        # Let strategy learn from this example
        strategy.add_example(assistant_text, engagement_after - engagement_before)

        # 8. Update bandit agent
        self.bandit_agent.update(strategy, reward)

        # Send update to UI for visualization
        if 'update_strategy' in self.ui_callbacks:
            self.ui_callbacks['update_strategy']((strategy, reward))

        # Update state
        self.last_engagement = engagement_after
        self.turn_count += 1

        # Start auto-advance timer now that TTS finished
        self.start_auto_advance_timer()

        # Create turn data
        turn_data = TurnData(
            user_text=user_text,
            user_spoke=user_spoke,
            strategy=strategy,
            assistant_text=assistant_text,
            engagement_before=engagement_before,
            engagement_after=engagement_after,
            reward=reward,
            duration=time.time() - turn_start
        )

        # Log turn
        self.logger.log_turn({
            'turn': self.turn_count,
            'user_text': user_text,
            'user_spoke': user_spoke,
            'strategy': strategy.to_prompt(),
            'assistant_text': assistant_text,
            'engagement_before': engagement_before,
            'engagement_after': engagement_after,
            'reward': reward,
            'duration': turn_data.duration
        })

        return turn_data

    async def run_session(self, client):
        """Run the main conversation session."""
        try:
            self.session_start_time = time.time()
            self.event_loop = asyncio.get_event_loop()

            # Start EEG streaming
            await self.eeg_manager.start_streaming(client)
            self.is_running = True

            # Initial greeting
            strategy = self.bandit_agent.select_strategy()
            greeting = await self.gpt.generate_response("", strategy)

            if 'update_transcript' in self.ui_callbacks:
                self.ui_callbacks['update_transcript'](f"Assistant: {greeting}")

            await self.tts.speak(greeting, strategy)

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
                if 'update_countdown' in self.ui_callbacks:
                    self.ui_callbacks['update_countdown'](seconds_left)
                await asyncio.sleep(1)

            if 'update_countdown' in self.ui_callbacks:
                self.ui_callbacks['update_countdown'](0)

            self.logger.info("Auto-advancing conversation (user silent)")
            empty_audio = b""
            asyncio.create_task(self.process_turn(empty_audio))
        except asyncio.CancelledError:
            if 'update_countdown' in self.ui_callbacks:
                self.ui_callbacks['update_countdown'](0)
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
            self.bandit_agent.save(save_path / "hierarchical_bandit.json")
            print("Bandit state saved successfully")
        except Exception as e:
            print(f"Failed to save bandit state: {e}")

        # Save session summary to file
        summary = self.get_session_summary()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = save_path / f"session_summary_{timestamp}.json"

        import json

        def serialize_object(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'tone'):
                return {
                    'tone': obj.tone,
                    'topic': obj.topic,
                    'emotion': obj.emotion,
                    'hook': obj.hook,
                    'index': obj.index
                }
            return str(obj)

        try:
            with open(summary_path, 'w') as f:
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
                "session_duration": time.time() - getattr(self, 'session_start_time', time.time()),
                "final_engagement": self.last_engagement
            },
            "bandit_performance": performance_summary,
            "engagement_stats": {
                "current_engagement": self.engagement_scorer.current_engagement,
                "baseline_collected": getattr(self.engagement_scorer, 'baseline_collected', False),
                "adaptive_baseline_initialized": getattr(self.engagement_scorer, 'baseline_initialized', False)
            }
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
            print(f"\nAVERAGE RECENT REWARD: {rl_perf.get('average_recent_reward', 'N/A')}")

            # Component analysis
            print("\n🎯 COMPONENT PERFORMANCE:")
            for component, data in rl_perf.get('components', {}).items():
                print(f"\n📊 {component.upper()}:")
                print(f"   Best Choice: {data.get('best_choice', 'N/A')} (score: {data.get('best_score', 0):.3f})")
                usage_stats = data.get('usage_stats', {})
                sorted_arms = sorted(usage_stats.items(), key=lambda x: x[1].get('average_reward', 0), reverse=True)
                print("   Top Performers:")
                for i, (arm, stats) in enumerate(sorted_arms[:3]):
                    print(f"     {i+1}. {arm}: {stats.get('average_reward', 0):.3f} avg ({stats.get('usage_count', 0)} uses, {stats.get('success_rate', 0):.1%} success)")

            restart_stats = rl_perf.get('restart_stats', {})
            print("\n🔄 ADAPTIVE RESTARTS:")
            print(f"   Total Restarts: {restart_stats.get('total_restarts', 0)}")
            print(f"   Last Restart: Step {restart_stats.get('last_restart_step', 0)}")
        else:
            print(f"\nRL Performance Error: {rl_perf['error']}")

        print("=" * 60)