"""Main orchestrator that coordinates all components."""

import asyncio
import time
from typing import Optional, Callable, Dict
from dataclasses import dataclass

import numpy as np

from config import RLConfig, AUTO_ADVANCE_TIMEOUT_SEC
from utils.logger import ConversationLogger
from audio.speech_to_text import SpeechToText
from audio.text_to_speech import TextToSpeech
from conversation.gpt_wrapper import GPTConversation
from eeg.device_manager import EEGDeviceManager
from eeg.engagement_scorer import EngagementScorer
from rl.agent import QLearningAgent
from rl.strategy import StrategySpace


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

        # Initialize RL components
        self.strategy_space = StrategySpace(subset_size=100)
        self.rl_agent = QLearningAgent(self.strategy_space)

        # Load any saved strategy memories
        from pathlib import Path
        memory_file = Path("models") / "strategy_memory.json"
        self.strategy_space.load_memory(memory_file)

        # State tracking
        self.last_strategy_idx = 0
        self.last_engagement = 0.5
        self.turn_count = 0
        self.is_running = False

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

    async def process_turn(self, audio_data: bytes) -> TurnData:
        """Process a single conversation turn."""
        # Cancel any auto-advance timer
        self.cancel_auto_advance_timer()

        turn_start = time.time()

        # 1. Transcribe user speech
        self.logger.info("Transcribing user speech...")
        user_text = await self.stt.transcribe(audio_data)
        user_spoke = len(user_text.strip()) > 0

        if 'update_transcript' in self.ui_callbacks:
            self.ui_callbacks['update_transcript'](f"User: {user_text if user_spoke else '[Silent]'}")

        # 2. Get current engagement
        engagement_before = self.engagement_scorer.current_engagement

        # 3. RL agent selects strategy
        state_idx = self.rl_agent.state_to_index(
            self.last_strategy_idx,
            self.last_engagement,
            user_spoke
        )
        strategy_idx = self.rl_agent.choose_action(state_idx)
        strategy = self.strategy_space.get_strategy(strategy_idx)

        self.logger.info(f"Selected strategy: {strategy}")

        # 4. Generate GPT response
        self.logger.info("Generating response...")
        assistant_text = await self.gpt.generate_response(user_text, strategy)

        if 'update_transcript' in self.ui_callbacks:
            self.ui_callbacks['update_transcript'](f"Assistant: {assistant_text}")

        # 5. Speak response with strategy and track engagement
        self.logger.info("Speaking response...")
        tts_start, tts_end = await self.tts.speak(assistant_text, strategy)  # Pass strategy to TTS

        # 6. Get engagement during TTS
        engagement_after = self.engagement_scorer.get_segment_engagement(
            tts_start, tts_end, self.eeg_manager
        )

        # 7. Calculate reward
        reward = (engagement_after - engagement_before) * 100  # Scale up
        reward = np.tanh(reward)  # Squash to [-1, 1] but preserve sign

        # Let strategy learn from this example
        strategy.add_example(assistant_text, engagement_after - engagement_before)

        # 8. Update RL agent
        next_state_idx = self.rl_agent.state_to_index(
            strategy_idx,
            engagement_after,
            user_spoke
        )
        self.rl_agent.update(state_idx, strategy_idx, reward, next_state_idx)

        # Update state
        self.last_strategy_idx = strategy_idx
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
            self.session_start_time = time.time()  # Add this line
            self.event_loop = asyncio.get_event_loop()

            # Start EEG streaming
            await self.eeg_manager.start_streaming(client)
            self.is_running = True

            # Initial greeting
            strategy = self.strategy_space.get_random_strategy()
            greeting = await self.gpt.generate_response("", strategy)

            if 'update_transcript' in self.ui_callbacks:
                self.ui_callbacks['update_transcript'](f"Assistant: {greeting}")

            await self.tts.speak(greeting, strategy)  # Pass strategy to TTS

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
        """Wait for the timeout and process an empty turn."""
        try:
            for seconds_left in range(self.auto_advance_timeout, 0, -1):
                if 'update_countdown' in self.ui_callbacks:
                    self.ui_callbacks['update_countdown'](seconds_left)
                await asyncio.sleep(1)

            if 'update_countdown' in self.ui_callbacks:
                self.ui_callbacks['update_countdown'](0)

            self.logger.info("Auto-advancing conversation (user silent)")
            empty_audio = b""
            await self.process_turn(empty_audio)
        except asyncio.CancelledError:
            if 'update_countdown' in self.ui_callbacks:
                self.ui_callbacks['update_countdown'](0)
            pass

    def stop(self):
        """Stop the conversation session."""
        self.is_running = False
        self.cancel_auto_advance_timer()
        self.tts.stop()

    # Update the cleanup method
    def cleanup(self):
        """Clean up resources."""
        self.cancel_auto_advance_timer()
        # Print session summary before cleanup
        if self.turn_count > 0:
            self.print_session_summary()

        self.stt.cleanup()
        self.tts.cleanup()

        # Save RL agent
        from pathlib import Path
        save_path = Path("models")
        save_path.mkdir(exist_ok=True)
        self.rl_agent.save(save_path / "q_table.pkl")

        # Persist strategy examples
        self.strategy_space.save_memory(save_path / "strategy_memory.json")

        # Save session summary to file
        summary = self.get_session_summary()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = save_path / f"session_summary_{timestamp}.json"

        import json

        # Custom serializer for Strategy objects
        def serialize_object(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'tone'):  # Strategy object
                return {
                    'tone': obj.tone,
                    'topic': obj.topic,
                    'emotion': obj.emotion,
                    'hook': obj.hook,
                    'index': obj.index
                }
            return str(obj)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=serialize_object)

        print(f"\nSession summary saved to: {summary_path}")

    def get_session_summary(self) -> Dict:
        """Generate comprehensive session summary."""
        performance_summary = self.rl_agent.get_performance_summary()

        return {
            "session_info": {
                "total_turns": self.turn_count,
                "session_duration": time.time() - getattr(self, 'session_start_time', time.time()),
                "final_engagement": self.last_engagement
            },
            "rl_performance": performance_summary,
            "engagement_stats": {
                "current_engagement": self.engagement_scorer.current_engagement,
                "baseline_collected": getattr(self.engagement_scorer, 'baseline_collected', False)
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

        # RL Performance
        rl_perf = summary["rl_performance"]
        if "error" not in rl_perf:
            print(f"\nTOTAL REWARD: {rl_perf['total_reward']:.2f}")
            print(f"AVERAGE REWARD: {rl_perf['average_reward']:.3f}")

            # Best strategy - Fixed to use dot notation
            best = rl_perf["best_strategy"]
            print(f"\n🏆 WINNING STRATEGY:")
            print(f"   Strategy: {best['strategy'].tone} tone, {best['strategy'].topic} topic")
            print(f"             {best['strategy'].emotion} emotion, {best['strategy'].hook} hook")
            print(f"   Average Reward: {best['average_reward']:.3f}")
            print(f"   Used {best['usage_count']} times")

            # Top strategies - Fixed to use dot notation
            print(f"\n🏅 TOP 5 STRATEGIES:")
            for i, strategy_info in enumerate(rl_perf["top_strategies"][:5], 1):
                s = strategy_info["strategy"]
                print(f"   {i}. {s.tone}/{s.topic}/{s.emotion}/{s.hook}")
                print(f"      Avg Reward: {strategy_info['average_reward']:.3f} "
                      f"(used {strategy_info['usage_count']} times)")

            # Learning progress
            learning = rl_perf["learning_progress"]
            print(f"\n📈 LEARNING PROGRESS:")
            print(f"   Early Average Reward: {learning['early_average_reward']:.3f}")
            print(f"   Recent Average Reward: {learning['recent_average_reward']:.3f}")
            print(f"   Improvement: {learning['improvement']:.3f}")

            # Exploration stats
            exploration = rl_perf["exploration_stats"]
            print(f"\n🔍 EXPLORATION:")
            print(f"   Strategies Tried: {exploration['strategies_tried']}/{exploration['total_strategies']}")
            print(f"   Final Epsilon: {exploration['final_epsilon']:.3f}")

        print("=" * 60)