# webrtc_interface.py
import asyncio
import json
import ssl

import aiohttp
import base64

import certifi
import numpy as np
from typing import Callable, Optional
import logging


class OpenAIRealtimeClient:
    """WebRTC client for OpenAI Realtime API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.websocket = None
        self.audio_callback: Optional[Callable] = None
        self.transcript_callback: Optional[Callable] = None
        self.is_speaking = False
        self.conversation_active = False

    async def connect(self):
        """Connect to OpenAI Realtime API"""
        try:
            # For development, we'll use WebSocket instead of full WebRTC
            # This is simpler for prototype and testing
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())

            self.session = aiohttp.ClientSession()
            self.websocket = await self.session.ws_connect(url, ssl=ssl_ctx,headers=headers)

            print("Connected to OpenAI Realtime API")

            # Start listening for server events
            asyncio.create_task(self._listen_for_events())

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def _listen_for_events(self):
        """Listen for server events"""
        async for msg in self.websocket:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                    await self._handle_server_event(event)
                except Exception as e:
                    print(f"Error handling server event: {e}")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket error: {msg.data}")
                break

    async def _handle_server_event(self, event: dict):
        """Handle server events"""
        event_type = event.get('type')

        if event_type == 'session.created':
            print("Session created")
            await self._configure_session()

        elif event_type == 'response.audio.delta':
            # Handle audio output from AI
            if self.audio_callback:
                audio_data = base64.b64decode(event.get('delta', ''))
                self.audio_callback(audio_data)

        elif event_type == 'response.audio_transcript.delta':
            # Handle transcript
            if self.transcript_callback:
                text = event.get('delta', '')
                self.transcript_callback(text)

        elif event_type == 'input_audio_buffer.speech_started':
            self.is_speaking = True
            print("User started speaking")

        elif event_type == 'input_audio_buffer.speech_stopped':
            self.is_speaking = False
            print("User stopped speaking")

        elif event_type == 'response.done':
            print("AI response complete")

        elif event_type == 'error':
            print(f"Server error: {event}")

    async def _configure_session(self):
        """Configure the session settings"""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "You are a helpful AI assistant engaging in conversation.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                }
            }
        }

        await self._send_event(config)

    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to the API"""
        if not self.websocket:
            return

        # Convert audio to base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }

        await self._send_event(event)

    async def update_instructions(self, strategy_prompt: str):
        """Update AI instructions with new strategy"""
        config = {
            "type": "session.update",
            "session": {
                "instructions": strategy_prompt
            }
        }

        await self._send_event(config)

    async def create_response(self):
        """Trigger AI response generation"""
        event = {
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"]
            }
        }

        await self._send_event(event)

    async def _send_event(self, event: dict):
        """Send event to server"""
        if self.websocket:
            await self.websocket.send_str(json.dumps(event))

    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """Set callback for audio output"""
        self.audio_callback = callback

    def set_transcript_callback(self, callback: Callable[[str], None]):
        """Set callback for transcript"""
        self.transcript_callback = callback

    async def disconnect(self):
        """Disconnect from API"""
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()