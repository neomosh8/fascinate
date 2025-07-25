"""GPT conversation management."""

import asyncio
from typing import List, Dict, Optional
from openai import OpenAI

from config import calculate_dynamic_tokens
from config import GPT_MODEL, MAX_GPT_TOKENS
from rl.strategy import Strategy


class GPTConversation:
    """Manages conversation with GPT."""

    def __init__(self):
        self.client = OpenAI()
        self.conversation_history: List[Dict[str, str]] = []

    async def generate_response(self,
                                user_input: str,
                                strategy: Strategy,
                                turn_count: int = 1,  # Add turn_count parameter
                                additional_context: Optional[str] = None) -> str:
        """Generate assistant response with given strategy and dynamic token limit."""

        # Calculate dynamic token limit
        max_tokens = calculate_dynamic_tokens(turn_count)

        # Build messages
        messages = []

        # Add strategy as system message
        messages.append({
            "role": "system",
            "content": f"""
{strategy.to_prompt_with_memory()}
Keep responses concise but allow for more detail as conversation progresses.
keep in the Token limit: {max_tokens}
Put normal talking glitches in bracket, like [laughter], [sigh], [umm], [ahhh], [uhhh], [pause], [clears throat], [cough], [hmm], [gulp] etc.."""
        })

        # Add conversation history (keep last 5 turns for context)
        for msg in self.conversation_history[-10:]:
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

        # Generate response with dynamic token limit
        try:
            loop = asyncio.get_event_loop()

            def _generate():
                return self.client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,  # Use dynamic token limit
                    temperature=0.8
                )

            completion = await loop.run_in_executor(None, _generate)

            response = completion.choices[0].message.content

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Optional: Log token usage for debugging
            print(f"Turn {turn_count}: Used {max_tokens} max tokens")

            return response

        except Exception as e:
            print(f"GPT error: {e}")
            return "I'm having trouble responding right now. Could you try again?"

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []