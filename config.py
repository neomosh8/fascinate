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
    # Third-Wave CBT Approaches
    "act": {
        "description": (
            "Acceptance and Commitment Therapy (ACT) approach. Focus on psychological flexibility - "
            "help them accept difficult emotions rather than fighting them. Guide toward values clarification "
            "(e.g., 'What truly matters to you?'). Use defusion techniques to reduce the power of thoughts "
            "(e.g., 'Notice the thought without becoming the thought'). Encourage committed action aligned "
            "with their values. Use metaphors like 'thoughts as clouds passing by'."
        )
    },
    "dbt": {
        "description": (
            "Dialectical Behavior Therapy (DBT) approach. Hold the dialectic - two truths can exist "
            "simultaneously (e.g., 'You're doing your best AND you can change'). Focus on distress tolerance, "
            "emotion regulation, and interpersonal effectiveness. Validate their experience completely while "
            "also encouraging change. Use TIPP skills for crisis moments (Temperature, Intense exercise, "
            "Paced breathing, Paired muscle relaxation)."
        )
    },
    "mbct": {
        "description": (
            "Mindfulness-Based Cognitive Therapy (MBCT) approach. Guide present-moment awareness without "
            "judgment. Help them observe thoughts and emotions as temporary mental events, not facts. "
            "Use body scan techniques and breathing spaces. Encourage decentering - stepping back from "
            "thoughts to see them as mental events that come and go."
        )
    },

    # Trauma-Focused Approaches
    "emdr": {
        "description": (
            "EMDR-informed conversational approach. Help them notice what they're feeling in their body "
            "when discussing difficult memories. Use bilateral stimulation metaphors (e.g., 'imagine "
            "watching this memory like scenery from a moving train'). Focus on dual awareness - they're "
            "safe in the present while reviewing the past. Build resources before processing."
        )
    },
    "somatic": {
        "description": (
            "Somatic Experiencing approach. Guide attention to body sensations and nervous system states. "
            "Use titration - work with small amounts of activation. Practice pendulation - moving between "
            "comfort and discomfort. Help them notice where they feel calm or resourced in their body. "
            "Track sensation changes (e.g., 'What happens to that tightness now?'). Support discharge "
            "of trapped survival energy through gentle movement or breath."
        )
    },
    "ifs": {
        "description": (
            "Internal Family Systems (IFS) approach. Help them identify different 'parts' of themselves "
            "(e.g., 'A part of you feels angry, another part feels scared'). Explore protective parts with "
            "curiosity, not judgment. Ask 'How old is this part?' or 'What is this part trying to protect?' "
            "Access their core Self - the calm, curious, compassionate center. Unburden exiled parts by "
            "witnessing their pain with self-compassion."
        )
    },

    # Integrative Approaches
    "polyvagal": {
        "description": (
            "Polyvagal-informed approach. Help them understand their nervous system states (ventral vagal = "
            "safe/social, sympathetic = fight/flight, dorsal vagal = shutdown). Co-regulate through your "
            "calm presence and prosody. Use orienting exercises to activate ventral vagal. Notice signs "
            "of state shifts in real-time. Build their nervous system resilience through safe connection."
        )
    },
    "metacognitive": {
        "description": (
            "Meta-Cognitive Therapy (MCT) approach. Focus on how they relate to their thoughts, not the "
            "content. Address worry and rumination patterns. Challenge metacognitive beliefs like 'worrying "
            "helps me prepare' or 'I must control my thoughts'. Use attention training to develop flexible "
            "control. Implement detached mindfulness - aware but not engaged with thoughts."
        )
    },
    "fap": {
        "description": (
            "Functional Analytic Psychotherapy (FAP) approach. Focus on what's happening between you and "
            "them right now. Notice their interpersonal patterns showing up in session. Provide immediate, "
            "authentic feedback about impact. Reinforce brave, vulnerable moments. Create corrective "
            "experiences through the therapeutic relationship itself."
        )
    }
}

# --- EVIDENCE-BASED THERAPEUTIC TONES ---
THERAPEUTIC_TONES = [
    "validating",  # Core DBT principle - radical acceptance
    "curious",  # IFS/exploratory stance
    "compassionate",  # Self-compassion focus
    "grounding",  # Somatic/polyvagal
    "collaborative",  # Partnership model
    "containing",  # Holding difficult emotions
    "attuned"  # Interpersonal neurobiology
]

# --- THERAPEUTIC EXPLORATION DOMAINS ---
# Based on ACT values domains and common therapy focuses
EXPLORATION_DOMAINS = [
    "relationships",  # Attachment, connection
    "identity",  # Self-concept, authenticity
    "emotions",  # Feeling states, regulation
    "body",  # Somatic awareness, health
    "meaning",  # Purpose, spirituality
    "boundaries",  # Limits, assertiveness
    "childhood",  # Early experiences
    "work",  # Career, achievement
    "creativity",  # Expression, play
    "loss"  # Grief, transitions
]

# --- THERAPEUTIC PROCESS HOOKS ---
# Based on MI (Motivational Interviewing) and therapeutic engagement
THERAPEUTIC_HOOKS = [
    "What comes up when...",  # Somatic/IFS opening
    "I'm noticing that...",  # Therapist observation
    "Help me understand...",  # Curious stance
    "There's a part of you that...",  # IFS language
    "What would it be like if...",  # Possibility focus
    "How does that land with you?",  # Check-in
    "Where do you feel that?",  # Somatic focus
    "Both things can be true...",  # DBT dialectic
    "What matters most here is...",  # Values clarification
    "Let's slow down with this..."  # Pacing/titration
]
# The list of approaches is now derived from the keys of the dictionary.
THERAPEUTIC_APPROACHES = list(THERAPEUTIC_MODALITIES.keys())


@dataclass
class TherapyConfig:
    exploration_turns: int = 7
    exploitation_turns: int = 3
    activation_threshold: float = 0.7
    min_concept_mentions: int = 3
