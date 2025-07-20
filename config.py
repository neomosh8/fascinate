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
    epsilon_initial: float = 0.9  # Start higher
    epsilon_decay: float = 0.999  # Decay slower
    epsilon_min: float = 0.15     # Explore longer

    # UCB parameters
    use_ucb: bool = True
    ucb_confidence: float = 2.0

# Communication Strategy Components
TONES = ["playful", "naughty", "informational", "bossy", "aggressive", "sarcastic", 'calm','kind']
TOPICS = ["politics", "facts", "story", "controversial", "nerds", "gaming","gen z", "boomer","high iq", "low iq", "autistic"]
EMOTIONS = ["happy", "sad", "serious", "scared", "whisper", "shouting", "laughter","flirting"]
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
MAX_GPT_TOKENS = 100
MAX_CONVERSATION_TURNS = 100

# Auto-advance timeout
AUTO_ADVANCE_TIMEOUT_SEC = 5  # seconds to wait before auto advancing

# Contextual Bandit Configuration
@dataclass
class ContextualBanditConfig:
    context_window_size: int = 5
    num_candidates: int = 20
    ucb_confidence: float = 2.0
    max_strategy_memory: int = 100
    min_experience_threshold: int = 3
    similarity_top_k: int = 5
