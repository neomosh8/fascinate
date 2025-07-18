# main.py - Complete implementation
import asyncio
import time
import os
from typing import Optional
import numpy as np
from engagement_tracker import TimeAlignedEngagementTracker

# Import our modules
from eeg_engagement import EngagementProcessor
from strategy_system import StrategyGenerator, CommunicationStrategy
from rl_agent_adaptive import AdaptiveEngagementRL
from webrtc_interface import OpenAIRealtimeClient
from udio_handler_sounddevice import AudioHandler
from custom_eeg_streamer import CustomEEGStreamer

# For EEG device discovery
try:
    import sys
    sys.path.append('.')
    from neocore_client import find_device
except ImportError:
    print("Warning: neocore_client not found. EEG will be simulated.")
    find_device = None


class ConversationalAI:
    """Main orchestrator with time-aligned engagement tracking"""

    def __init__(self, openai_api_key: str, user_name: str = "friend"):
        self.openai_api_key = openai_api_key
        self.user_name = user_name

        # Initialize components
        self.engagement_processor = EngagementProcessor()
        self.strategy_generator = StrategyGenerator()
        self.rl_agent = AdaptiveEngagementRL(self.strategy_generator.get_strategy_count())
        self.realtime_client = OpenAIRealtimeClient(openai_api_key)
        self.audio_handler = AudioHandler()

        # NEW: Time-aligned engagement tracker
        self.engagement_tracker = TimeAlignedEngagementTracker(audio_delay_estimate=0.3)

        # State tracking
        self.conversation_turn = 0
        self.current_strategy: Optional[CommunicationStrategy] = None
        self.current_strategy_index = None
        self.last_engagement = 0.5
        self.conversation_active = False
        self.ai_speaking = False
        self.user_speaking = False

        # EEG simulation mode if hardware not available
        self.simulate_eeg = find_device is None
        self.eeg_streamer = None

        # Setup callbacks
        self.engagement_processor.set_engagement_callback(self._on_engagement_update)
        self.realtime_client.set_transcript_callback(self._on_ai_transcript)
        self.realtime_client.set_audio_callback(self._on_ai_audio)
        self.realtime_client.set_audio_complete_callback(
            lambda: self.engagement_tracker.on_audio_playback_complete()
        )
        # NEW: Set up time-aligned RL updates
        self.engagement_tracker.set_rl_update_callback(self._on_time_aligned_rl_update)

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

        # Initialize audio with event loop and callback
        print("Initializing audio system...")
        self.audio_handler.set_event_loop(asyncio.get_event_loop())
        self.audio_handler.set_audio_send_callback(self._send_audio_to_openai)  # Add this
        self.audio_handler.start_recording()
        self.audio_handler.start_playback()

        # Start monitoring task
        asyncio.create_task(self._monitor_audio_debug())

        # Initialize EEG
        if not self.simulate_eeg:
            print("Initializing EEG connection...")
            await self._init_eeg()
        else:
            print("⚠️  Running in EEG simulation mode")
            asyncio.create_task(self._simulate_eeg_data())

        print("✅ System initialized successfully!")

    async def _init_eeg(self):
        """Initialize real EEG connection"""
        try:
            # Find EEG device
            device_address = await find_device()

            # Create custom EEG streamer that feeds our engagement processor
            self.eeg_streamer = CustomEEGStreamer(self.engagement_processor)

            # Start EEG streaming in background
            asyncio.create_task(self.eeg_streamer.stream_data(device_address))

        except Exception as e:
            print(f"EEG initialization failed: {e}")
            print("Falling back to simulation mode")
            self.simulate_eeg = True
            asyncio.create_task(self._simulate_eeg_data())

    async def _send_audio_to_openai(self, audio_data: bytes):
        """Send audio data to OpenAI - only when appropriate"""
        # Only send when not AI speaking and conversation is active
        if self.conversation_active and not self.ai_speaking:
            await self.realtime_client.send_audio_chunk(audio_data)
            return True
        return False

    async def _monitor_audio_debug(self):
        """Monitor audio system for debugging"""
        while self.conversation_active:
            await asyncio.sleep(10)  # Every 10 seconds

            stats = self.audio_handler.get_debug_stats()
            print(f"🎤 Audio Debug: captured={stats['chunks_captured']}, "
                  f"sent={stats['chunks_sent']}, "
                  f"last_audio={stats['last_audio_ago']:.1f}s ago" if stats['last_audio_ago'] else "never")

    async def _simulate_eeg_data(self):
        """Simulate EEG data for testing"""
        while True:
            samples_per_chunk = 27
            t = np.linspace(0, samples_per_chunk/250, samples_per_chunk)

            # Vary engagement based on conversation state and user speaking
            base_engagement = 0.6 if self.conversation_active else 0.4
            if self.user_speaking:
                base_engagement += 0.2  # Higher engagement when user speaks

            ch1_data = (
                base_engagement * np.sin(2 * np.pi * 10 * t) +
                (base_engagement * 1.5) * np.sin(2 * np.pi * 20 * t) +
                0.1 * np.random.randn(samples_per_chunk)
            ) * 1000

            ch2_data = ch1_data + 0.05 * np.random.randn(samples_per_chunk) * 1000

            self.engagement_processor.add_eeg_data(ch1_data.tolist(), ch2_data.tolist())

            await asyncio.sleep(0.1)

    async def _monitor_audio_status(self):
        """Monitor audio buffer status"""
        while self.conversation_active:
            await asyncio.sleep(2)  # Check every 2 seconds

            if hasattr(self.audio_handler, 'get_buffer_status'):
                status = self.audio_handler.get_buffer_status()
                if status.get('buffered_duration', 0) > 0.1:  # Only show if there's significant audio
                    print(f"\n🔊 Buffer: {status['buffered_duration']:.1f}s")

    def _on_engagement_update(self, engagement: float):
        """Handle engagement updates - now feeds into time tracker"""
        print(f"📊 Engagement: {engagement:.2f}")

        # Add to time-aligned tracker instead of immediate RL update
        self.engagement_tracker.add_engagement_measurement(engagement)

        # Keep for other purposes
        self.last_engagement = engagement

    def _on_time_aligned_rl_update(self, strategy_index: int, engagement: float,
                                   engagement_change: float, session):
        """Handle RL update with immediate adaptation"""
        if not self.conversation_active:
            return

        # Calculate reward
        reward = self.rl_agent.calculate_reward(engagement_change, engagement)

        # IMMEDIATE UPDATE - agent adapts right away
        self.rl_agent.update_performance(strategy_index, reward, engagement)

        # Store experience and train
        state = self.rl_agent.get_state(
            current_engagement=engagement,
            engagement_change=engagement_change,
            conversation_length=self.conversation_turn
        )

        if hasattr(self.rl_agent, 'current_state'):
            self.rl_agent.store_experience(
                self.rl_agent.current_state,
                strategy_index,
                reward,
                state,
                done=False
            )

            # Train immediately for fast adaptation
            self.rl_agent.train_step()

        self.rl_agent.current_state = state

        # Show detailed feedback
        stats = self.rl_agent.get_strategy_stats()
        print(f"🤖 ADAPTIVE RL Update - Strategy: {strategy_index}, "
              f"Reward: {reward:.2f}, Hot strategies: {stats['hot_list']}")

    def _on_ai_audio(self, audio_data: bytes):
        """Handle AI audio output - track timing"""
        # Play audio
        self.audio_handler.play_audio(audio_data)

        # Track audio chunk for timing
        self.engagement_tracker.on_audio_chunk_received()

        self.ai_speaking = True

    def _on_ai_transcript(self, text: str):
        """Handle AI transcript updates"""
        if text.strip():
            print(f"{text}", end="", flush=True)

    async def start_conversation(self):
        """Start the main conversation loop"""
        print("\n🎙️  Starting conversation...")
        print("The AI will begin. Speak naturally when it's your turn.")
        print("The conversation will continue for multiple turns.")
        print("Press Ctrl+C to stop.\n")

        self.conversation_active = True

        try:
            # Initial AI greeting with first strategy
            await self._ai_turn_with_strategy()

            # Main conversation loop - keep running indefinitely
            while self.conversation_active:
                # Check for user speech events from OpenAI
                await asyncio.sleep(1)

                # Auto-advance conversation every 30 seconds if no interaction
                if hasattr(self, 'last_ai_response_time'):
                    if time.time() - self.last_ai_response_time > 30:
                        print("\n⏰ Auto-advancing conversation...")
                        await self._ai_turn_with_strategy()

        except KeyboardInterrupt:
            print("\n\n🛑 Conversation stopped by user")
        finally:
            await self._cleanup()

        # main.py - Add strategy monitoring


    async def _ai_turn_with_strategy(self):
        """AI speaks with selected strategy - with timing tracking"""
        self.conversation_turn += 1
        self.ai_speaking = True

        current_engagement = self.engagement_processor.current_engagement
        engagement_change = self.engagement_processor.get_engagement_change()

        state = self.rl_agent.get_state(
            current_engagement=current_engagement,
            engagement_change=engagement_change,
            conversation_length=self.conversation_turn
        )

        action = self.rl_agent.select_action(state)
        strategy = self.strategy_generator.get_strategy_by_index(action)

        # Store for tracking
        self.current_strategy = strategy
        self.current_strategy_index = action

        # NEW: Start tracking this strategy
        self.engagement_tracker.start_new_strategy(action, strategy)

        print(f"\n🎯 Turn {self.conversation_turn} Strategy:")
        print(f"   Index: {action}")  # Show the actual index
        print(f"   Tone: {strategy.tone}")
        print(f"   Topic: {strategy.topic}")
        print(f"   Emotion: {strategy.emotion}")
        print(f"   Hook: {strategy.hook}")

        # Show strategy stats every few turns with error handling
        if self.conversation_turn % 3 == 0:
            try:
                stats = self.rl_agent.get_strategy_stats()
                print(f"📈 Strategy Stats: {stats['strategies_tried']}/{stats['total_strategies']} tried, "
                      f"{stats['strategies_unused']} unused")
            except KeyError as e:
                print(f"📈 Strategy Stats: Error accessing {e}")
            except Exception as e:
                print(f"📈 Strategy Stats: Error {e}")
        print()

        strategy_prompt = strategy.to_prompt(self.user_name)
        if self.conversation_turn == 1:
            strategy_prompt += "\nThis is the beginning of the conversation. Give a warm greeting and start the conversation naturally. Keep it conversational and ask a question to engage the user."
        else:
            strategy_prompt += f"\nThis is turn {self.conversation_turn} of our conversation. Build on the previous discussion and keep the conversation flowing naturally."

        await self.realtime_client.update_instructions(strategy_prompt)
        await self.realtime_client.create_response()

        self.last_ai_response_time = time.time()

        # Wait a bit then mark AI as not speaking
        await asyncio.sleep(3)
        self.ai_speaking = False

    async def _schedule_next_turn(self):
        """Schedule next AI turn after user interaction"""
        await asyncio.sleep(15)  # Adjust based on conversation pace

        if self.conversation_active:
            await self._ai_turn_with_strategy()

    async def _cleanup(self):
        """Clean up resources"""
        self.conversation_active = False

        # Save engagement plot
        self.engagement_processor.save_engagement_plot("conversation_engagement.png")

        # Save RL model
        try:
            self.rl_agent.save_model("engagement_model.pth")
            print("💾 RL model saved")
        except Exception as e:
            print(f"Failed to save RL model: {e}")

        # Clean up audio
        self.audio_handler.cleanup()

        # Disconnect from OpenAI
        await self.realtime_client.disconnect()
        print("🔌 Disconnected from OpenAI")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Main execution
async def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    USER_NAME = "friend"

    ai = ConversationalAI(OPENAI_API_KEY, USER_NAME)

    try:
        await ai.initialize()
        await ai.start_conversation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())