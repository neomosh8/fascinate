#!/usr/bin/env python3
"""
EEG-Driven Conversational AI with RL
Main entry point for the application.
"""

import asyncio
import sys
import os
from bleak import BleakClient

# Add imports
from core.orchestrator import ConversationOrchestrator
from ui.app_window import ConversationUI
from utils.logger import ConversationLogger


async def main():
    """Main application entry point."""
    logger = ConversationLogger()
    logger.info("Starting EEG Conversation RL System")

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1

    if not os.getenv("ELEVENLABS_API_KEY"):
        print("Error: ELEVENLABS_API_KEY environment variable not set")
        return 1

    # Get EEG device MAC if provided
    eeg_mac = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if ':' in arg:
                eeg_mac = arg.upper()
                break

    # Initialize orchestrator
    orchestrator = ConversationOrchestrator()

    # Initialize EEG connection
    if not await orchestrator.initialize(eeg_mac):
        print("Failed to initialize. Exiting.")
        return 1

    # Create and run UI
    def run_ui():
        ui = ConversationUI(orchestrator)

        # Run async session in background
        async def session_runner():
            async with BleakClient(orchestrator.eeg_manager.device_address) as client:
                await orchestrator.run_session(client)

        # Start session in background thread
        import threading
        def run_async_session():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(session_runner())

        session_thread = threading.Thread(target=run_async_session)
        session_thread.daemon = True
        session_thread.start()

        # Run UI
        ui.run()

        # Cleanup
        orchestrator.cleanup()

    # Run UI in main thread
    run_ui()

    logger.info("Application terminated")
    return 0


if __name__ == "__main__":
    try:
        # Run with asyncio
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)