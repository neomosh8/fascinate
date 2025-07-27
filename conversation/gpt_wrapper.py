"""GPT conversation management with z-score risk detection and exploit phase optimization."""

import asyncio
from typing import List, Dict, Optional
from openai import OpenAI
import numpy as np

from config import calculate_dynamic_tokens
from config import GPT_MODEL, MAX_GPT_TOKENS
from rl.strategy import Strategy


class GPTConversation:
    """Manages conversation with GPT including statistical risk detection."""

    def __init__(self):
        self.client = OpenAI()
        self.conversation_history: List[Dict[str, str]] = []

        # Risk detection tracking with sufficient history for z-score
        self.engagement_history = []
        self.emotion_history = []
        self.min_history_for_zscore = 5  # Need at least 5 points for meaningful stats

    def _calculate_zscore_risk(self, current_engagement: float, current_emotion: float) -> Optional[str]:
        """
        Statistical risk detection using z-scores > 3 standard deviations.
        Returns repair instructions if statistically significant risk detected.
        """
        if len(self.engagement_history) < self.min_history_for_zscore:
            return None

        risk_instructions = []

        # Calculate z-scores for engagement and emotion
        eng_array = np.array(self.engagement_history)
        emo_array = np.array(self.emotion_history)

        # Engagement z-score (looking for drops, so negative z-score is concerning)
        if len(eng_array) > 1:
            eng_mean = np.mean(eng_array)
            eng_std = np.std(eng_array)

            if eng_std > 0.01:  # Avoid division by zero for very stable values
                eng_zscore = (current_engagement - eng_mean) / eng_std

                if eng_zscore < -3.0:  # Statistically significant drop
                    risk_instructions.append(
                        f""
                        "IMMEDIATE COURSE CORRECTION NEEDED: "
                        "- Something is clearly not working - change approach immediately "
                        "- Ask directly: 'Should we talk about something else?' "
                        "- Switch to completely different topic or style "
                        "- Be more engaging/energetic or try appropriate humor "
                        "- Let them fully control conversation direction"
                    )

        # Emotion z-score (looking for extreme changes in either direction)
        if len(emo_array) > 1:
            emo_mean = np.mean(emo_array)
            emo_std = np.std(emo_array)

            if emo_std > 0.01:
                emo_zscore = abs(current_emotion - emo_mean) / emo_std

                if emo_zscore > 3.0:  # Statistically significant change
                    if current_emotion < 0.3:
                        risk_instructions.append(
                            f"⚠️ STATISTICALLY SIGNIFICANT NEGATIVE EMOTION (z={emo_zscore:.2f}): "
                            "IMMEDIATE REPAIR NEEDED: "
                            "- You likely triggered something - acknowledge this possibility "
                            "- Ask if you said something that felt off "
                            "- Offer genuine apology if appropriate "
                            "- Completely change approach or let them guide "
                            "- Be much more gentle and validating"
                        )
                    elif current_emotion > 0.7:
                        risk_instructions.append(
                            f"✅ STATISTICALLY SIGNIFICANT POSITIVE EMOTION (z={emo_zscore:.2f}): "
                            "AMPLIFY THIS SUCCESS: "
                            "- This approach is working exceptionally well "
                            "- Continue exactly what you just did "
                            "- Build momentum on this positive connection "
                            "- Match and slightly amplify their positive energy"
                        )

        return "\n".join(risk_instructions) if risk_instructions else None

    def _get_model_parameters(self, strategy: Strategy) -> Dict:
        """
        Get model parameters based on exploration vs exploitation phase.
        Exploitation phase uses more focused, deterministic parameters.
        """
        # Check if this is a therapeutic strategy with exploration mode
        is_exploration = True  # Default assumption

        if hasattr(strategy, 'exploration_mode'):
            is_exploration = strategy.exploration_mode

        if is_exploration:
            # Exploration phase: more creative, diverse responses
            return {
                "model": GPT_MODEL,
                "temperature": 0.8,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        else:
            # Exploitation phase: more focused, consistent responses

            return {
                "model": "o4-mini",
                "reasoning_effort":"medium"
            }

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

        # Update tracking history
        self.engagement_history.append(current_engagement)
        self.emotion_history.append(current_emotion)

        # Keep rolling window of last 20 values for z-score calculation
        if len(self.engagement_history) > 20:
            self.engagement_history.pop(0)
            self.emotion_history.pop(0)

        # Calculate dynamic token limit
        max_tokens = calculate_dynamic_tokens(turn_count)

        # Get model parameters based on exploration/exploitation phase
        model_params = self._get_model_parameters(strategy)

        # Build messages
        messages = []

        # Build system prompt with phase-aware instructions
        phase_context = ""
        if hasattr(strategy, 'exploration_mode'):
            if strategy.exploration_mode:
                phase_context = "\n🔍 EXPLORATION PHASE: Be curious, try different approaches, discover what resonates."
            else:
                phase_context = f"\n🎯 EXPLOITATION PHASE: Focus deeply on {getattr(strategy, 'target_concept', 'the core topic')}. Build systematically on what's working."

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
- Token limit: {max_tokens}

MEMORY CHECK: Before responding, consider:
1. What has the user already told me about this topic?
2. What did they just emphasize or show emotion about?
3. Are they showing frustration with my approach?
4. How can I build on their last response rather than starting fresh?
"""

        # Add risk instructions if statistically significant risk detected
        if risk_instructions:
            system_prompt += f"\n\n🚨 STATISTICAL ALERT - IMMEDIATE PRIORITY:\n{risk_instructions}\n"
            print(f"⚠️ Statistically significant risk detected - added repair instructions")

        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # Add conversation history (keep last 50 turns for context)
        for msg in self.conversation_history[-50:]:
            messages.append(msg)

        # Add user input
        if user_input:
            messages.append({
                "role": "user",
                "content": user_input
            })
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
        else:
            # User was silent
            messages.append({
                "role": "user",
                "content": "[User remained silent, that's okay, continue]"
            })

        # Generate response with phase-adaptive parameters
        try:
            loop = asyncio.get_event_loop()

            def _generate():
                # Build API call parameters
                api_params = {
                    "messages": messages,
                    **model_params
                }

                # Only add max_tokens for exploration mode
                if hasattr(strategy, 'exploration_mode') and strategy.exploration_mode:
                    api_params["max_tokens"] = max_tokens

                return self.client.chat.completions.create(**api_params)
            completion = await loop.run_in_executor(None, _generate)
            response = completion.choices[0].message.content

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Enhanced logging
            phase = "EXPLORATION" if getattr(strategy, 'exploration_mode', True) else "EXPLOITATION"
            print(f"Turn {turn_count}: {phase} mode")

            if risk_instructions:
                print(f"🔧f Statistical repair attempted: E:{current_engagement:.3f} Em:{current_emotion:.3f}")

            return response

        except Exception as e:
            print(f"GPT error: {e}")
            return "I'm having trouble responding right now. Could you try again?"

    def reset_conversation(self):
        """Reset conversation history and risk tracking."""
        self.conversation_history = []
        self.engagement_history = []
        self.emotion_history = []

    def get_risk_stats(self) -> Dict:
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
                "current": float(eng_array[-1]) if len(eng_array) > 0 else None
            },
            "emotion_stats": {
                "mean": float(np.mean(emo_array)),
                "std": float(np.std(emo_array)),
                "current": float(emo_array[-1]) if len(emo_array) > 0 else None
            }
        }