"""Logging utilities for debugging and monitoring."""

import logging
import json
from datetime import datetime
from pathlib import Path


class ConversationLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{timestamp}.json"
        self.session_data = {
            "start_time": datetime.now().isoformat(),
            "turns": []
        }

        # Setup Python logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("EEG_RL")

    def log_turn(self, turn_data: dict):
        """Log a conversation turn with all relevant data."""
        self.session_data["turns"].append({
            "timestamp": datetime.now().isoformat(),
            **turn_data
        })
        self._save_session()

    def _save_session(self):
        """Save session data to file."""
        with open(self.session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def log_session_summary(self, summary_data: dict):
        """Log session summary with performance metrics."""
        self.session_data["summary"] = summary_data
        self.session_data["end_time"] = datetime.now().isoformat()
        self._save_session()

        # Also create a separate summary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.log_dir / f"summary_{timestamp}.json"

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
