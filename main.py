#!/usr/bin/env python3
"""
EEG-Driven Conversational AI with RL
Main entry point with pygame UI.
"""

import asyncio
import sys
import os

# Add imports
from core.orchestrator import ConversationOrchestrator
from ui.pygame_ui import create_pygame_ui  # New pygame UI
from utils.logger import ConversationLogger


async def main():
    """Main application entry point."""
    logger = ConversationLogger()
    logger.info("Starting EEG Conversation RL System with Pygame UI")

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
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

    # Create and run pygame UI
    create_pygame_ui(orchestrator)

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