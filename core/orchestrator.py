"""Main orchestrator that coordinates all components."""

import asyncio
import time
from typing import Optional, Callable
from dataclasses import dataclass

from config import RLConfig
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
        self.event_loop = None  # Add this line

        # Initialize RL components
        self.strategy_space = StrategySpace(subset_size=100)
        self.rl_agent = QLearningAgent(self.strategy_space)

        # State tracking
        self.last_strategy_idx = 0
        self.last_engagement = 0.5
        self.turn_count = 0
        self.is_running = False

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

        # 5. Speak response and track engagement
        self.logger.info("Speaking response...")
        tts_start, tts_end = await self.tts.speak(assistant_text)

        # 6. Get engagement during TTS
        engagement_after = self.engagement_scorer.get_segment_engagement(
            tts_start, tts_end, self.eeg_manager
        )

        # 7. Calculate reward
        reward = engagement_after - engagement_before
        reward = max(-1.0, min(1.0, reward))  # Clip to [-1, 1]

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
            self.event_loop = asyncio.get_event_loop()

            # Start EEG streaming
            await self.eeg_manager.start_streaming(client)
            self.is_running = True

            # Initial greeting
            strategy = self.strategy_space.get_random_strategy()
            greeting = await self.gpt.generate_response("", strategy)

            if 'update_transcript' in self.ui_callbacks:
                self.ui_callbacks['update_transcript'](f"Assistant: {greeting}")

            await self.tts.speak(greeting)

            # Main conversation loop
            while self.is_running:
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Session error: {e}")
        finally:
            await self.eeg_manager.stop_streaming()

    def stop(self):
        """Stop the conversation session."""
        self.is_running = False
        self.tts.stop()

    def cleanup(self):
        """Clean up resources."""
        self.stt.cleanup()
        self.tts.cleanup()

        # Save RL agent
        from pathlib import Path
        save_path = Path("models")
        save_path.mkdir(exist_ok=True)
        self.rl_agent.save(save_path / "q_table.pkl")