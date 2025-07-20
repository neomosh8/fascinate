"""GPT conversation management."""

import asyncio
from typing import List, Dict, Optional
from openai import OpenAI

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
                                additional_context: Optional[str] = None) -> str:
        """Generate assistant response with given strategy."""

        # Build messages
        messages = []

        # Add strategy as system message
        messages.append({
            "role": "system",
            "content": f"""You are a conversational assistant designed to keep users engaged.
{strategy.to_prompt_with_memory()}

Keep responses concise .
put normal talking glitches in bracket, like [laughter] [smirk] [cough] [ahhhh] [emmmm] etc.."""
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
                "content": "[User remained silent]"
            })

        # Generate response
        try:
            loop = asyncio.get_event_loop()

            def _generate():
                return self.client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=messages,
                    max_tokens=MAX_GPT_TOKENS,
                    temperature=0.8
                )

            completion = await loop.run_in_executor(None, _generate)

            response = completion.choices[0].message.content

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            print("%%%%%%%%",strategy.to_prompt_with_memory())
            return response

        except Exception as e:
            print(f"GPT error: {e}")
            return "I'm having trouble responding right now. Could you try again?"

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []