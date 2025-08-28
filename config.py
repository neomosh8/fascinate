"""Configuration and constants for the EEG Conversation RL system."""

import os
from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (now loaded from .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3")

# Model configurations
WHISPER_MODEL = "gpt-4o-transcribe"
GPT_MODEL = "gpt-5"
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
# TONES = ["playful", "naughty", "informational", "aggressive", "sarcastic", 'calm','kind', "confident","empathic"]
# TOPICS = ["facts", "story", "controversial", "nerdy","gen z", "boomer","high iq", "low iq", "autistic","professional"]
# EMOTIONS = ["happy", "sad", "serious", "scared","worry",  "whisper", "angry", "laughter","flirting","thoughtful"]
# HOOKS = ["hey [name]", "you know what?", "are you with me?", "listen", "look", "Oh my god", " i can  not believe it!","can you beleive it?", "not gonna lie"]

# Communication Strategy Components
TONES = ["playful", "confident", "kind", "sarcastic", "informational"]
TOPICS = ["story", "facts", "controversial", "gen z", "professional"]
EMOTIONS = ["happy", "serious", "thoughtful", "angry", "flirting"]
HOOKS = ["hey maya", "you know what?", "listen", "can you believe it?", "not gonna lie"]


# UI Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRAPH_UPDATE_INTERVAL = 50  # ms

# Audio Configuration
AUDIO_SAMPLE_RATE = 22050
AUDIO_CHUNK_SIZE = 1024
MAX_RECORDING_DURATION = 120  # seconds

# Token limits
# Token limits - Dynamic scaling
MIN_GPT_TOKENS = 130        # Minimum tokens during warmup
MAX_GPT_TOKENS = 180       # Maximum tokens after full ramp-up
WARMUP_TURNS = 10         # Turns to stay at minimum
MAX_TURN = 30             # Turn where maximum is reached
MAX_CONVERSATION_TURNS = 100


# Auto-advance timeout
AUTO_ADVANCE_TIMEOUT_SEC = 120  # seconds to wait before auto advancing
HUME_API_KEY = os.getenv("HUME_API_KEY")

# TTS Engine Selection
TTS_ENGINE = os.getenv("TTS_ENGINE", "openai")  # "openai", "hume", or "elevenlabs"
# Contextual Bandit Configuration
@dataclass
class ContextualBanditConfig:
    context_window_size: int = 5
    num_candidates: int = 20
    ucb_confidence: float = 2.0
    max_strategy_memory: int = 100
    min_experience_threshold: int = 3
    similarity_top_k: int = 5
def calculate_dynamic_tokens(turn_count: int) -> int:
    """Calculate dynamic token limit based on turn count."""
    if turn_count <= WARMUP_TURNS:
        return MIN_GPT_TOKENS
    elif turn_count >= MAX_TURN:
        return MAX_GPT_TOKENS
    else:
        # Linear interpolation between min and max
        progress = (turn_count - WARMUP_TURNS) / (MAX_TURN - WARMUP_TURNS)
        token_range = MAX_GPT_TOKENS - MIN_GPT_TOKENS
        return int(MIN_GPT_TOKENS + (progress * token_range))


# Therapeutic Strategy Components
# --- NEW: Define Modalities with Descriptions for Embedding ---
# This dictionary is the new "brain" for selecting and describing approaches.
# The descriptions are crafted to be both human-readable and semantically rich for embedding.
THERAPEUTIC_MODALITIES = {
    "ifs": {
        "description": (
            "an Internal Family Systems (IFS) approach. Gently personify the user's feelings "
            "as 'parts' of them (e.g., 'a part of you that feels tension,' 'the playful part'). "
            "Explore the positive intention behind each part, especially the protective ones. "
            "The goal is to foster curiosity and compassion for their inner world."
        )
    },
    "somatic": {
        "description": (
            "a Somatic (body-based) approach. Guide the user to notice the physical sensations "
            "connected to their emotions (e.g., 'Where do you feel that in your body?'). "
            "Focus on tracking sensations like heat, tension, or energy without judgment. "
            "The goal is to help them process emotions through the body."
        )
    },
    "narrative": {
        "description": (
            "a Narrative Therapy approach. Focus on the 'story' the user is telling about the problem. "
            "Help them externalize the problem (e.g., 'What has The Anxiety been telling you?'). "
            "Look for 'exceptions' or moments when the problem wasn't in charge. "
            "The goal is to help them see themselves as separate from the problem and re-author their story."
        )
    },
    "cbt": {
        "description": (
            "a Cognitive Behavioral Therapy (CBT) approach. Focus on the link between a thought, a feeling, and a behavior. "
            "Gently help the user identify the specific thoughts that arise with the target feeling. "
            "You might explore the evidence for that thought or consider alternative perspectives. "
            "The goal is to challenge and reframe unhelpful thought patterns."
        )
    },
    "solution_focused": {
        "description": (
            "a Solution-Focused Brief Therapy (SFBT) approach. Keep the focus on the future and potential solutions. "
            "Use questions that presuppose change, like the 'Miracle Question' (e.g., 'If this feeling was gone tomorrow, "
            "what would be the first small thing you'd notice?'). Look for the user's existing strengths and resources."
        )
    }
}

# The list of approaches is now derived from the keys of the dictionary.
THERAPEUTIC_APPROACHES = list(THERAPEUTIC_MODALITIES.keys())
THERAPEUTIC_TONES = ["empathetic", "validating", "curious", "gentle", "reflective", "supportive"]
EXPLORATION_DOMAINS = ["childhood", "relationships", "dreams", "fears", "identity", "career", "family"]
THERAPEUTIC_HOOKS = ["Help me understand...", "I'm curious about...", "Tell me more about...", "What comes up for you...", "It sounds like...", "I'm noticing..."]



@dataclass
class TherapyConfig:
    exploration_turns: int = 3
    exploitation_turns: int = 5
    activation_threshold: float = 0.7
    min_concept_mentions: int = 2
