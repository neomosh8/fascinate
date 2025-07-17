# main.py
import asyncio
import time
import os
from typing import Optional
import numpy as np

# Import our modules (assuming they're in the same directory)
from eeg_engagement import EngagementProcessor
from strategy_system import StrategyGenerator, CommunicationStrategy
from rl_agent import EngagementRL
from webrtc_interface import OpenAIRealtimeClient

# Import the neocore EEG client
import sys

sys.path.append('.')  # Add current directory to path
try:
    from neocore_client import EEGStreamer, find_device
except ImportError:
    print("Warning: neocore_client not found. EEG will be simulated.")
    EEGStreamer = None


class ConversationalAI:
    """Main orchestrator for the EEG-driven conversational AI system"""

    def __init__(self, openai_api_key: str, user_name: str = "friend"):
        self.openai_api_key = openai_api_key
        self.user_name = user_name

        # Initialize components
        self.engagement_processor = EngagementProcessor()
        self.strategy_generator = StrategyGenerator()
        self.rl_agent = EngagementRL(self.strategy_generator.get_strategy_count())
        self.realtime_client = OpenAIRealtimeClient(openai_api_key)

        # State tracking
        self.conversation_turn = 0
        self.current_strategy: Optional[CommunicationStrategy] = None
        self.last_engagement = 0.5
        self.conversation_active = False

        # EEG simulation mode if hardware not available
        self.simulate_eeg = EEGStreamer is None
        self.eeg_streamer = None

        # Setup callbacks
        self.engagement_processor.set_engagement_callback(self._on_engagement_update)
        self.realtime_client.set_transcript_callback(self._on_ai_transcript)

        # Load existing RL model if available
        try:
            self.rl_agent.load_model("engagement_model.pth")
        except:
            print("No existing RL model found, starting fresh")

    async def initialize(self):
        """Initialize all components"""
        print("🧠 Initializing EEG-Driven Conversational AI")
        print("=" * 50)

        # Connect to OpenAI
        print("Connecting to OpenAI Realtime API...")
        if not await self.realtime_client.connect():
            raise Exception("Failed to connect to OpenAI")

        # Initialize EEG
        if not self.simulate_eeg:
            print("Initializing EEG connection...")
            await self._init_eeg()
        else:
            print("⚠️  Running in EEG simulation mode")
            # Start simulated EEG data
            asyncio.create_task(self._simulate_eeg_data())

        print("✅ System initialized successfully!")

    async def _init_eeg(self):
        """Initialize real EEG connection"""
        try:
            # Find EEG device
            device_address = await find_device()

            # Create custom EEG streamer that feeds our engagement processor
            class CustomEEGStreamer(EEGStreamer):
                def __init__(self, engagement_processor):
                    super().__init__()
                    self.engagement_processor = engagement_processor

                def notification_handler(self, sender: int, data: bytearray):
                    try:
                        if len(data) < 6:
                            return
                        from neocore_client import parse_eeg_packet
                        ch1_samples, ch2_samples = parse_eeg_packet(data[2:])

                        # Feed to engagement processor instead of plotter
                        self.engagement_processor.add_eeg_data(ch1_samples, ch2_samples)

                    except Exception as e:
                        print(f"EEG data parsing error: {e}")

            self.eeg_streamer = CustomEEGStreamer(self.engagement_processor)

            # Start EEG streaming in background
            asyncio.create_task(self.eeg_streamer.stream_data(device_address))

        except Exception as e:
            print(f"EEG initialization failed: {e}")
            print("Falling back to simulation mode")
            self.simulate_eeg = True
            asyncio.create_task(self._simulate_eeg_data())

    async def _simulate_eeg_data(self):
        """Simulate EEG data for testing"""
        while True:
            # Generate realistic-looking EEG data
            samples_per_chunk = 27

            # Base signal with some noise
            t = np.linspace(0, samples_per_chunk / 250, samples_per_chunk)

            # Simulate different engagement levels based on conversation state
            base_engagement = 0.6 if self.conversation_active else 0.4

            # Add some alpha (10Hz) and beta (20Hz) components
            ch1_data = (
                               base_engagement * np.sin(2 * np.pi * 10 * t) +  # Alpha
                               (base_engagement * 1.5) * np.sin(2 * np.pi * 20 * t) +  # Beta
                               0.1 * np.random.randn(samples_per_chunk)  # Noise
                       ) * 1000  # Scale to microvolts

            ch2_data = ch1_data + 0.05 * np.random.randn(samples_per_chunk) * 1000

            self.engagement_processor.add_eeg_data(ch1_data.tolist(), ch2_data.tolist())

            await asyncio.sleep(0.1)  # ~10Hz update rate

    def _on_engagement_update(self, engagement: float):
        """Handle engagement updates from EEG processor"""
        print(f"📊 Engagement: {engagement:.2f}")

        # Update RL agent if conversation is active
        if self.conversation_active and self.conversation_turn > 0:
            self._update_rl_agent(engagement)

    def _update_rl_agent(self, current_engagement: float):
        """Update RL agent with new engagement data"""
        engagement_change = self.engagement_processor.get_engagement_change()

        # Create state
        state = self.rl_agent.get_state(
            current_engagement=current_engagement,
            engagement_change=engagement_change,
            conversation_length=self.conversation_turn
        )

        # Calculate reward and update
        if hasattr(self.rl_agent, 'current_state') and self.rl_agent.last_action is not None:
            reward = self.rl_agent.calculate_reward(engagement_change, current_engagement)

            # Store experience
            self.rl_agent.store_experience(
                self.rl_agent.current_state,
                self.rl_agent.last_action,
                reward,
                state,
                done=False
            )

            # Update performance tracking
            self.rl_agent.update_performance(self.rl_agent.last_action, reward)

            # Train the agent
            self.rl_agent.train_step()

            print(f"🤖 RL Update - Reward: {reward:.2f}, Epsilon: {self.rl_agent.epsilon:.3f}")

        # Update state
        self.rl_agent.current_state = state
        self.rl_agent.engagement_history.append(current_engagement)
        self.last_engagement = current_engagement

    def _on_ai_transcript(self, text: str):
        """Handle AI transcript updates"""
        if text.strip():
            print(f"🤖 AI: {text}")

    async def start_conversation(self):
        """Start the main conversation loop"""
        print("\n🎙️  Starting conversation...")
        print("The AI will begin. Speak naturally when it's your turn.")
        print("Press Ctrl+C to stop.\n")

        self.conversation_active = True

        try:
            # Initial AI greeting with first strategy
            await self._ai_turn_with_strategy()

            # Main conversation loop
            while self.conversation_active:
                await asyncio.sleep(1)  # Keep the loop running

        except KeyboardInterrupt:
            print("\n\n🛑 Conversation stopped by user")
        finally:
            await self._cleanup()

    async def _ai_turn_with_strategy(self):
        """AI speaks with selected strategy"""
        self.conversation_turn += 1

        # Get current state for RL agent
        current_engagement = self.engagement_processor.current_engagement
        engagement_change = self.engagement_processor.get_engagement_change()

        state = self.rl_agent.get_state(
            current_engagement=current_engagement,
            engagement_change=engagement_change,
            conversation_length=self.conversation_turn
        )

        # RL agent selects strategy
        action = self.rl_agent.select_action(state)
        self.rl_agent.last_action = action
        self.rl_agent.current_state = state

        # Get strategy and create prompt
        strategy = self.strategy_generator.get_strategy_by_index(action)
        self.current_strategy = strategy

        print(f"\n🎯 Turn {self.conversation_turn} Strategy:")
        print(f"   Tone: {strategy.tone}")
        print(f"   Topic: {strategy.topic}")
        print(f"   Emotion: {strategy.emotion}")
        print(f"   Hook: {strategy.hook}")
        print()

        # Update AI instructions with strategy
        strategy_prompt = strategy.to_prompt(self.user_name)
        if self.conversation_turn == 1:
            # Initial greeting
            strategy_prompt += "\nThis is the beginning of the conversation. Give a warm greeting and start the conversation naturally."

        await self.realtime_client.update_instructions(strategy_prompt)

        # Trigger AI response
        await self.realtime_client.create_response()

        # Schedule next turn after delay (simulate conversation flow)
        asyncio.create_task(self._schedule_next_turn())

    async def _schedule_next_turn(self):
        """Schedule next AI turn after user interaction"""
        # Wait for user to speak and AI to respond
        await asyncio.sleep(15)  # Adjust based on conversation pace

        if self.conversation_active:
            await self._ai_turn_with_strategy()

    async def _cleanup(self):
        """Clean up resources"""
        self.conversation_active = False

        # Save RL model
        try:
            self.rl_agent.save_model("engagement_model.pth")
            print("💾 RL model saved")
        except Exception as e:
            print(f"Failed to save RL model: {e}")

        # Disconnect from OpenAI
        await self.realtime_client.disconnect()
        print("🔌 Disconnected from OpenAI")

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Main execution
async def main():
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    USER_NAME = "friend"  # Can be customized

    # Create and run the conversational AI
    ai = ConversationalAI(OPENAI_API_KEY, USER_NAME)

    try:
        await ai.initialize()
        await ai.start_conversation()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())