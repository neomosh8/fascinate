"""Configuration and constants for the EEG Conversation RL system."""

import os
from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (now loaded from .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model configurations
WHISPER_MODEL = "gpt-4o-transcribe"
GPT_MODEL = "gpt-4.1"
TTS_MODEL = "gpt-4o-mini-tts"  # Updated to use OpenAI TTS
TTS_VOICE = "coral"  # Default OpenAI voice

# EEG Configuration
EEG_SAMPLE_RATE = 250  # Hz
ENGAGEMENT_WINDOW_SEC = 3  # seconds for EWMA smoothing
BETA_BAND = (13, 30)  # Hz for beta band (attention/engagement)

# RL Configuration
@dataclass
class RLConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.6
    epsilon_initial: float = 0.2
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.05

# Communication Strategy Components
TONES = ["playful", "naughty", "informational", "bossy", "aggressive", "sarcastic"]
TOPICS = ["politics", "facts", "story", "controversial", "dad joke", "flirting"]
EMOTIONS = ["happy", "sad", "serious", "scared", "whisper", "shout out", "laughter"]
HOOKS = ["hey [name]", "you know what?", "are you with me?", "listen", "look"]


# UI Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRAPH_UPDATE_INTERVAL = 50  # ms

# Audio Configuration
AUDIO_SAMPLE_RATE = 22050
AUDIO_CHUNK_SIZE = 1024
MAX_RECORDING_DURATION = 30  # seconds

# Token limits
MAX_GPT_TOKENS = 180
MAX_CONVERSATION_TURNS = 100