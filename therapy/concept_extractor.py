"""Concept extraction using OpenAI API."""

import asyncio
import json
from typing import List
from openai import OpenAI


class ConceptExtractor:
    """Extract psychological concepts from conversation text using OpenAI."""

    def __init__(self):
        self.client = OpenAI()

    async def extract_concepts(self, text: str) -> List[str]:
        """Extract psychological concepts from text using GPT."""

        prompt = (
            "You are a psychological concept extractor. Analyze the following text and"
            " extract key psychological concepts, themes, and emotionally significant topics.\n\n"
            "Return ONLY a JSON object with 'concepts' array. Focus on:\n"
            "- Relationships (mother, father, partner, family, etc.)\n"
            "- Emotions (anger, fear, shame, joy, etc.)\n"
            "- Life domains (work, childhood, health, money, etc.)\n"
            "- Psychological themes (control, trust, abandonment, success, failure, etc.)\n\n"
            "Text to analyze: \"{text}\"\n\n"
            "Return format: {{'concepts': ['concept1', 'concept2']}}\n"
            "Maximum 10 concepts."
        )

        try:
            loop = asyncio.get_event_loop()

            def _extract():
                completion = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "developer",
                            "content": "You are a psychological concept extraction specialist. Return only JSON objects.",
                        },
                        {"role": "user", "content": prompt.format(text=text)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                return completion.choices[0].message.content

            result = await loop.run_in_executor(None, _extract)
            concepts_data = json.loads(result)
            return concepts_data.get("concepts", [])

        except Exception as e:
            print(f"Concept extraction error: {e}")
            return []
