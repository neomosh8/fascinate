"""GPT conversation management with z-score risk detection and exploit phase optimization."""

import asyncio
from typing import List, Dict, Optional, Any
from openai import OpenAI
import numpy as np

from config import calculate_dynamic_tokens
from config import GPT_MODEL, MAX_GPT_TOKENS, TTS_ENGINE
from rl.strategy import Strategy


class GPTConversation:
    """Manages conversation with GPT including statistical risk detection."""

    def __init__(self):
        self.client = OpenAI()
        self.conversation_history: List[Dict[str, str]] = []

        # Risk detection tracking with sufficient history for z-score
        self.engagement_history: List[float] = []
        self.emotion_history: List[float] = []
        self.min_history_for_zscore = 5  # Need at least 5 points for meaningful stats

    def _calculate_zscore_risk(self, current_engagement: float, current_emotion: float) -> Optional[str]:
        """
        Statistical risk detection using z-scores > 3 standard deviations.
        Returns repair instructions if statistically significant risk detected.
        """
        if len(self.engagement_history) < self.min_history_for_zscore:
            return None

        risk_instructions: List[str] = []

        eng_array = np.array(self.engagement_history)
        emo_array = np.array(self.emotion_history)

        # Engagement z-score (looking for drops)
        if len(eng_array) > 1:
            eng_mean = float(np.mean(eng_array))
            eng_std = float(np.std(eng_array))

            if eng_std > 0.01:
                eng_zscore = (current_engagement - eng_mean) / eng_std
                if eng_zscore < -3.0:
                    risk_instructions.append(
                        ""
                        "IMMEDIATE COURSE CORRECTION NEEDED: "
                        "- Something is clearly not working - change approach immediately "
                        "- Ask directly: 'Should we talk about something else?' "
                        "- Switch to completely different topic or style "
                        "- Be more engaging/energetic or try appropriate humor "
                        "- Let them fully control conversation direction"
                    )

        # Emotion z-score
        if len(emo_array) > 1:
            emo_mean = float(np.mean(emo_array))
            emo_std = float(np.std(emo_array))

            if emo_std > 0.01:
                emo_z = abs(current_emotion - emo_mean) / emo_std

                if emo_z > 3.0:
                    if current_emotion < 0.3:
                        risk_instructions.append(
                            f"⚠️ STATISTICALLY SIGNIFICANT NEGATIVE EMOTION (z={emo_z:.2f}): "
                            "IMMEDIATE REPAIR NEEDED: "
                            "- You likely triggered something - acknowledge this possibility "
                            "- Ask if you said something that felt off "
                            "- Offer genuine apology if appropriate "
                            "- Completely change approach or let them guide "
                            "- Be much more gentle and validating"
                        )
                    elif current_emotion > 0.7:
                        risk_instructions.append(
                            f"✅ STATISTICALLY SIGNIFICANT POSITIVE EMOTION (z={emo_z:.2f}): "
                            "AMPLIFY THIS SUCCESS: "
                            "- This approach is working exceptionally well "
                            "- Continue exactly what you just did "
                            "- Build momentum on this positive connection "
                            "- Match and slightly amplify their positive energy"
                        )

        return "\n".join(risk_instructions) if risk_instructions else None

    def _get_model_parameters(self, strategy: Strategy) -> Dict[str, Any]:
        """
        Get model parameters based on exploration vs exploitation phase
        using Responses API fields.
        """
        is_exploration = getattr(strategy, "exploration_mode", True)

        common: Dict[str, Any] = {
            "model": GPT_MODEL,
            # reasoning summary is supported in Responses with reasoning models
            # keep summary set to auto for lightweight metadata
            "reasoning": {"summary": "auto"},
        }

        if is_exploration:
            # Exploration: creative and diverse, minimal reasoning effort
            return {
                **common,
                "reasoning": {"effort": "minimal", "summary": "auto"},
                # leave temperature/top_p to model defaults during exploration
            }
        else:
            # Exploitation: focused and consistent
            return {
                **common,
                "reasoning": {"effort": "medium", "summary": "auto"},
            }

    def _as_response_messages(self, system_prompt: str, user_input: Optional[str]) -> List[Dict[str, Any]]:
        def part(t: str, block_type: str) -> Dict[str, str]:
            return {"type": block_type, "text": t}

        msgs: List[Dict[str, Any]] = []

        # System message as 'input_text' or could be default 'input_text'
        msgs.append({
            "role": "system",
            "content": [part(system_prompt, "input_text")]
        })

        for msg in self.conversation_history[-10:]:
            block_type = "input_text" if msg["role"] == "user" else "output_text"
            msgs.append({
                "role": msg["role"],
                "content": [part(msg["content"], block_type)]
            })

        if user_input:
            msgs.append({
                "role": "user",
                "content": [part(user_input, "input_text")]
            })
        else:
            msgs.append({
                "role": "user",
                "content": [part("[User remained silent, that's okay, continue]", "input_text")]
            })

        return msgs

    async def generate_response(self,
                                user_input: str,
                                strategy: Strategy,
                                turn_count: int = 1,
                                current_engagement: float = 0.5,
                                current_emotion: float = 0.5,
                                additional_context: Optional[str] = None) -> str:
        """Generate assistant response with z-score risk detection and phase-adaptive parameters."""

        # Statistical risk detection
        risk_instructions = self._calculate_zscore_risk(current_engagement, current_emotion)

        # Update tracking histories with rolling window
        self.engagement_history.append(current_engagement)
        self.emotion_history.append(current_emotion)
        if len(self.engagement_history) > 20:
            self.engagement_history.pop(0)
        if len(self.emotion_history) > 20:
            self.emotion_history.pop(0)

        # Dynamic token cap for exploration
        max_tokens = calculate_dynamic_tokens(turn_count)

        # Phase-aware parameters
        model_params = self._get_model_parameters(strategy)

        # Build phase context
        phase_context = ""
        if hasattr(strategy, "exploration_mode"):
            if strategy.exploration_mode:
                phase_context = "\n🔍 EXPLORATION PHASE: Be curious, try different approaches, discover what resonates."
            else:
                phase_context = f"\n🎯 EXPLOITATION PHASE: Focus deeply on {getattr(strategy, 'target_concept', 'the core topic')}. Build systematically on what's working."

        # System prompt body with special handling for the first message
        if turn_count == 0 or (
            additional_context and "first message" in additional_context.lower()
        ):
            system_prompt = f"""
{strategy.to_prompt_with_memory()}
{phase_context}

IMPORTANT: This is the FIRST message of the session. You should:
- Introduce yourself warmly
- Set a comfortable, welcoming tone
- Briefly explain you're here to have a supportive conversation
- Ask an open-ended question to start the dialogue
- Keep it brief and natural

Example opening: "Hey there, I'm here to chat with you today. How are you doing?"
"""
        else:
            system_prompt = f"""
{strategy.to_prompt_with_memory()}
{phase_context}

CONVERSATION FLOW RULES:
- NEVER repeat questions the user has already answered
- Build organically on what they just shared - go deeper into THEIR content
- If user shows frustration/boredom, immediately shift approach
- Track emotional state: if engagement drops, change topic or style
- Use varied language - avoid formulaic therapeutic phrases
- When user shares something meaningful, explore THAT rather than switching topics

RESPONSE STYLE:
- If user resists: Back off and let them lead
- Use natural speech patterns with occasional glitches like [pause], [hmm], [sigh], [umm]
- keep it short and don't bombared with too many questiosn at one, 1 or 2 at most

MEMORY CHECK: Before responding, consider:
1. What has the user already told me about this topic?
2. What did they just emphasize or show emotion about?
3. Are they showing frustration with my approach?
4. How can I build on their last response rather than starting fresh?
"""

        if TTS_ENGINE.lower() == "elevenlabs":
            system_prompt += (
                "Use  audio tags to adjust emotional delivery and create natural, empathetic dialogue.\n\n"
                "VOICE-RELATED EXPRESSIONS:\n"
                "These guide tone and emotional nuance:\n"
                "  [whispers], [sighs], [gentle laugh], [softly], [exhales]\n"
                "  [reassuring], [calm], [concerned], [empathetic], [encouraging]\n"
                "Example:\n"
                '  [softly] It sounds like you’ve been carrying this weight for a long time.\n'
                '  [reassuring] You are safe here, and we can take this one step at a time.\n\n'
                "PUNCTUATION & DELIVERY:\n"
                "Use punctuation to shape pacing and emphasis:\n"
                "  - Ellipses (…) create reflective pauses.\n"
                "  - CAPITALIZATION signals stronger emphasis.\n"
                "  - Commas and periods maintain natural rhythm.\n"
                "Example:\n"
                '  [sighs] … It can feel overwhelming at first, but remember: HEALING takes time.\n\n'
                "THERAPEUTIC TONE EXAMPLES:\n"
                "1. Empathetic Listening:\n"
                '   [concerned] I hear how painful this has been for you.\n'
                '   … Would you like to share more about when these feelings started?\n\n'
                "2. Gentle Encouragement:\n"
                "   [encouraging] You’ve made so much progress already.\n"
                "   [soft laugh] Sometimes we don’t notice our own growth until we pause to reflect.\n\n"
                "3. Mindfulness/Meditation Cue:\n"
                "   [whispers] Take a slow breath in … and let it go.\n"
                "   [calm] Notice the weight leaving your body as you exhale.\n\n"
                "BEST PRACTICES:\n"
                "• Match tags to the therapeutic context — a meditative voice shouldn’t shout.\n"
                "• Use tags sparingly, to enhance but not overwhelm natural flow.\n"
                "• Test voices with tags, as some effects vary depending on the chosen voice.\n"
            )

        if risk_instructions:
            system_prompt += f"\n\n🚨 STATISTICAL ALERT - IMMEDIATE PRIORITY:\n{risk_instructions}\n"

        # Prepare messages for Responses API
        messages = self._as_response_messages(system_prompt, user_input)

        # Maintain your local history for future context
        if user_input:
            self.conversation_history.append({"role": "user", "content": user_input})

        # Perform the API call
        try:
            loop = asyncio.get_event_loop()

            def _generate():
                api_params = {
                    **model_params,
                    "input": messages,
                }
                # Only cap output tokens in exploration mode
                # if getattr(strategy, "exploration_mode", True):
                #     api_params["max_output_tokens"] = max_tokens  # Responses API uses this field

                return self.client.responses.create(**api_params)

            completion = await loop.run_in_executor(None, _generate)

            # Prefer convenience property, with safe fallbacks
            response_text: Optional[str] = getattr(completion, "output_text", None)
            if not response_text:
                try:
                    # New Responses API returns a structured output array
                    # Find the first text item
                    for item in getattr(completion, "output", []):
                        if item.get("type") == "message":
                            for part in item["content"]:
                                if part.get("type") == "output_text":
                                    response_text = part.get("text")
                                    break
                        if response_text:
                            break
                except Exception:
                    # Last resort
                    response_text = str(completion)

            if not response_text:
                response_text = "I could not parse a reply from the model."

            # Save assistant reply to history
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Logging
            phase = "EXPLORATION" if getattr(strategy, "exploration_mode", True) else "EXPLOITATION"
            print(f"Turn {turn_count}: {phase} mode")
            if risk_instructions:
                print(f"Statistical repair attempted: E:{current_engagement:.3f} Em:{current_emotion:.3f}")

            return response_text

        except Exception as e:
            print(f"GPT error: {e}")
            return "I'm having trouble responding right now. Could you try again?"

    def reset_conversation(self):
        """Reset conversation history and risk tracking."""
        self.conversation_history = []
        self.engagement_history = []
        self.emotion_history = []

    def get_risk_stats(self) -> Dict[str, Any]:
        """Get current risk detection statistics for debugging."""
        if len(self.engagement_history) < self.min_history_for_zscore:
            return {"status": "insufficient_data", "history_length": len(self.engagement_history)}

        eng_array = np.array(self.engagement_history)
        emo_array = np.array(self.emotion_history)

        return {
            "status": "active",
            "history_length": len(self.engagement_history),
            "engagement_stats": {
                "mean": float(np.mean(eng_array)),
                "std": float(np.std(eng_array)),
                "current": float(eng_array[-1]) if len(eng_array) > 0 else None,
            },
            "emotion_stats": {
                "mean": float(np.mean(emo_array)),
                "std": float(np.std(emo_array)),
                "current": float(emo_array[-1]) if len(emo_array) > 0 else None,
            },
        }
