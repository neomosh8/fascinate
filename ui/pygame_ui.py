"""Pygame-based UI with sphere visualization and conversation interface."""

import pygame
import pygame.freetype
import math
import time
import random
import numpy as np
from scipy.fft import fft, fftfreq
from collections import deque
from typing import Optional, List, Dict, Tuple
import re
import asyncio
import threading
import queue

from config import WINDOW_WIDTH, WINDOW_HEIGHT
from core.orchestrator import ConversationOrchestrator
from ui.bandit_visualizer import BanditVisualizationDashboard


class EngagementWidget:
    """Apple Watch style ECG engagement display."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.engagement_history = deque(maxlen=width)
        self.time_counter = 0

    def update(self, engagement: float):
        """Update engagement reading."""
        self.engagement_history.append(engagement)
        self.time_counter += 1

    def draw(self, screen: pygame.Surface):
        """Draw ECG-style engagement plot."""
        self.surface.fill((0, 0, 0, 180))  # Semi-transparent background

        if len(self.engagement_history) > 1:
            # Draw grid lines
            for i in range(0, self.rect.height, 20):
                pygame.draw.line(self.surface, (0, 50, 0), (0, i), (self.rect.width, i), 1)
            for i in range(0, self.rect.width, 20):
                pygame.draw.line(self.surface, (0, 50, 0), (i, 0), (i, self.rect.height), 1)

            # Draw engagement line
            points = []
            for i, engagement in enumerate(self.engagement_history):
                x = i
                y = int(self.rect.height - (engagement * self.rect.height))
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.surface, (0, 255, 100), False, points, 2)

                # Add glow effect
                for i in range(3):
                    glow_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
                    pygame.draw.lines(glow_surface, (0, 255, 100, 50), False, points, 4 + i*2)
                    self.surface.blit(glow_surface, (0, 0))

        # Draw current value text
        if self.engagement_history:
            current_val = self.engagement_history[-1]
            font = pygame.freetype.Font(None, 16)
            text = f"{current_val:.3f}"
            font.render_to(self.surface, (5, 5), text, (0, 255, 100))

        screen.blit(self.surface, self.rect)


class FadingText:
    """Text with fade in/out effects."""

    def __init__(self, text: str, font: pygame.freetype.Font, color: Tuple[int, int, int], x: int, y: int):
        self.text = text
        self.font = font
        self.color = color
        self.x = x
        self.y = y
        self.alpha = 0
        self.target_alpha = 255
        self.fade_speed = 8
        self.active = True

        # Pre-render text
        text_surface, _ = self.font.render(self.text, self.color)
        self.surface = text_surface.convert_alpha()

    def fade_in(self):
        """Start fading in."""
        self.target_alpha = 255

    def fade_out(self):
        """Start fading out."""
        self.target_alpha = 0

    def update(self):
        """Update fade animation."""
        if self.alpha < self.target_alpha:
            self.alpha = min(self.alpha + self.fade_speed, self.target_alpha)
        elif self.alpha > self.target_alpha:
            self.alpha = max(self.alpha - self.fade_speed, self.target_alpha)

        if self.target_alpha == 0 and self.alpha == 0:
            self.active = False

    def draw(self, screen: pygame.Surface):
        """Draw faded text."""
        if self.active and self.alpha > 0:
            # Create a copy and set alpha
            temp_surface = self.surface.copy()
            temp_surface.set_alpha(self.alpha)

            # Center text
            text_rect = temp_surface.get_rect()
            text_rect.centerx = self.x
            text_rect.centery = self.y

            screen.blit(temp_surface, text_rect)


class SphereVisualization:
    """3D sphere visualization with trending words."""

    def __init__(self, center_x: int, center_y: int, grid_cols: int = 60, grid_rows: int = 30):
        self.center_x = center_x
        self.center_y = center_y
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        # Font setup
        self.font_size = 12
        self.font = pygame.freetype.Font(None, self.font_size)

        # Get character dimensions
        char_surface, char_rect = self.font.render('@', (255, 255, 255))
        self.char_width = char_rect.width
        self.char_height = char_rect.height

        # Sphere parameters
        self.r_base = min(grid_cols, grid_rows) / 3.5
        self.amp = 0.8
        self.freq = 0.05
        self.correction = self.char_height / self.char_width

        # Trending words
        self.trending_words = ["AI", "conversation", "engagement", "learning"]
        self.directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        self.letter_map = {}
        self.last_update = time.time()
        self.start_time = self.last_update

        # Shading characters
        self.shades = ' .:-=+*#%@'

        # Audio properties
        self.bass = 0.0

    def update_trending_words(self, words: List[str]):
        """Update trending words from AI response."""
        self.trending_words = words
        self.generate_letter_map()

    def set_audio_level(self, level: float):
        """Set audio level for sphere deformation."""
        self.bass = level

    def generate_letter_map(self):
        """Generate letter placement map."""
        self.letter_map = {}
        selected_words = random.sample(self.trending_words, min(3, len(self.trending_words)))

        for word in selected_words:
            for _ in range(6):  # 2 placements per word
                attempts = 0
                while attempts < 50:
                    i_start = random.randint(0, self.grid_rows - 1)
                    j_start = random.randint(0, self.grid_cols - 1)
                    di, dj = random.choice(self.directions)

                    # Check if word fits
                    can_fit = True
                    for k in range(len(word)):
                        ii = i_start + k * di
                        jj = j_start + k * dj
                        if not (0 <= ii < self.grid_rows and 0 <= jj < self.grid_cols):
                            can_fit = False
                            break

                    if can_fit:
                        # Place letters
                        for k in range(len(word)):
                            ii = i_start + k * di
                            jj = j_start + k * dj
                            self.letter_map[(ii, jj)] = word[k]
                        break

                    attempts += 1

    def draw(self, screen: pygame.Surface):
        """Draw the 3D sphere."""
        current_time = time.time()
        t = current_time - self.start_time

        # Update letter map periodically
        if current_time - self.last_update > 5.0:
            self.generate_letter_map()
            self.last_update = current_time

        # Calculate sphere radius with animation
        r = self.r_base * (1 + self.amp * ((math.sin(2 * math.pi * self.freq * t) + 1) / 2))

        # Create grids
        grid = [[' ' for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        bright_grid = [[0.0 for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        # Fill sphere
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                x = j - self.grid_cols / 2.0
                y = (i - self.grid_rows / 2.0) * self.correction
                dist2 = x**2 + y**2

                # Audio deformation
                if dist2 > 0:
                    angle = math.atan2(y, x)
                else:
                    angle = 0

                deform = 0.3 * self.bass * (math.sin(6 * angle + 2 * t) + math.sin(4 * angle - t)) / 2.0
                r_effective = r * (1 + deform)

                if dist2 < r_effective**2:
                    z = math.sqrt(r_effective**2 - dist2)

                    # Simple lighting
                    pos_norm = math.sqrt(x**2 + y**2 + z**2)
                    if pos_norm > 0:
                        lighting = (z / pos_norm + 1) / 2
                    else:
                        lighting = 0.5

                    # Add some texture
                    brightness = lighting * (0.8 + 0.2 * math.sin(x * 0.5) * math.cos(y * 0.5))
                    brightness = max(0.0, min(1.0, brightness))

                    bright_grid[i][j] = brightness
                    index = int(brightness * (len(self.shades) - 1))
                    grid[i][j] = self.shades[index]

        # Overlay letters
        for (i, j), letter in self.letter_map.items():
            x = j - self.grid_cols / 2.0
            y = (i - self.grid_rows / 2.0) * self.correction
            dist2 = x**2 + y**2

            if dist2 > 0:
                angle = math.atan2(y, x)
            else:
                angle = 0

            deform = 0.3 * self.bass * (math.sin(6 * angle + 2 * t) + math.sin(4 * angle - t)) / 2.0
            r_effective = r * (1 + deform)

            if dist2 < r_effective**2:
                grid[i][j] = letter

        # Render the grid
        start_x = self.center_x - (self.grid_cols * self.char_width) // 2
        start_y = self.center_y - (self.grid_rows * self.char_height) // 2

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                char = grid[i][j]
                if char != ' ':
                    bright = bright_grid[i][j]

                    # Matrix green color with intensity
                    main_color = (0, int(100 + 155 * bright), 0)
                    glow_color = (0, int(50 + 100 * bright), 0)

                    x_pos = start_x + j * self.char_width
                    y_pos = start_y + i * self.char_height

                    # Render glow
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        glow_surface, _ = self.font.render(char, glow_color)
                        glow_surface.set_alpha(60)
                        screen.blit(glow_surface, (x_pos + dx, y_pos + dy))

                    # Render main character
                    char_surface, _ = self.font.render(char, main_color)
                    screen.blit(char_surface, (x_pos, y_pos))


class PygameConversationUI:
    """Main pygame UI for EEG conversation system."""

    def __init__(self, orchestrator: ConversationOrchestrator):
        self.orchestrator = orchestrator

        # Initialize pygame
        pygame.init()
        pygame.freetype.init()
        pygame.mixer.init()

        # Setup display
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('EEG-Driven Conversation with RL')

        # Colors
        self.bg_color = (10, 15, 20)
        self.text_color = (200, 255, 200)
        self.button_color = (30, 60, 40)
        self.button_hover_color = (50, 100, 60)
        self.button_active_color = (100, 50, 50)

        # Fonts
        self.font_large = pygame.freetype.Font(None, 24)
        self.font_medium = pygame.freetype.Font(None, 18)
        self.font_small = pygame.freetype.Font(None, 14)

        # UI Components
        self.sphere = SphereVisualization(self.screen_width // 2, 250)
        self.engagement_widget = EngagementWidget(self.screen_width - 220, 20, 200, 120)

        # Bandit visualization dashboard
        self.bandit_dashboard = BanditVisualizationDashboard(self.screen_width, self.screen_height)
        self.show_dashboard = True

        # Track latest strategy performance
        self.latest_strategy = None
        self.latest_reward = None

        # Message display
        self.current_message = None
        self.message_y = 450
        self.messages = deque(maxlen=5)

        # Buttons
        self.speak_button_rect = pygame.Rect(self.screen_width // 2 - 100, 500, 200, 60)
        self.summary_button_rect = pygame.Rect(self.screen_width // 2 - 150, 600, 100, 30)
        self.stop_button_rect = pygame.Rect(self.screen_width // 2 + 50, 600, 100, 30)

        # State
        self.is_recording = False
        self.space_pressed = False
        self.button_hover = None
        self.clock = pygame.time.Clock()
        self.running = True

        # Thread-safe queue for updates
        self.update_queue = queue.Queue()

        # Bind orchestrator callbacks
        self.orchestrator.ui_callbacks = {
            'update_engagement': self._queue_engagement_update,
            'update_transcript': self._queue_transcript_update,
            'update_countdown': self._queue_countdown_update,
            'update_strategy': self._queue_strategy_update,
        }

        # Audio management
        self.current_audio_file = None
        self.audio_level = 0.0

    def _queue_engagement_update(self, engagement: float):
        """Queue engagement update (thread-safe)."""
        self.update_queue.put(('engagement', engagement))

    def _queue_transcript_update(self, text: str):
        """Queue transcript update (thread-safe)."""
        self.update_queue.put(('transcript', text))

    def _queue_countdown_update(self, seconds_left: int):
        """Queue countdown update (thread-safe)."""
        self.update_queue.put(('countdown', seconds_left))

    def _queue_strategy_update(self, data):
        """Queue strategy update for visualization."""
        self.update_queue.put(('strategy_update', data))

    def extract_words_from_text(self, text: str) -> List[str]:
        """Extract interesting words from AI response."""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                       'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                       'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                       'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'}

        # Extract words, clean them up
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        interesting_words = [w for w in words if w not in common_words and len(w) > 3]

        # Return up to 8 random words
        return random.sample(interesting_words, min(8, len(interesting_words))) if interesting_words else ['conversation']

    def add_message(self, text: str, is_user: bool = False):
        """Add a new message with fade effect."""
        # Fade out current message
        if self.current_message:
            self.current_message.fade_out()

        # Create new message
        color = (150, 200, 255) if is_user else (200, 255, 150)
        prefix = "You: " if is_user else "AI: "
        full_text = prefix + text

        # Wrap text if too long
        if len(full_text) > 80:
            full_text = full_text[:77] + "..."

        new_message = FadingText(
            full_text,
            self.font_medium,
            color,
            self.screen_width // 2,
            self.message_y
        )
        new_message.fade_in()

        self.messages.append(self.current_message)
        self.current_message = new_message

        # Extract words for sphere if AI message
        if not is_user:
            words = self.extract_words_from_text(text)
            if words:
                self.sphere.update_trending_words(words)

    def handle_speak_button_press(self):
        """Handle speak button press."""
        if not self.is_recording:
            self.is_recording = True
            self.orchestrator.cancel_auto_advance_timer()
            self.orchestrator.stt.start_recording()

    def handle_speak_button_release(self):
        """Handle speak button release."""
        if self.is_recording:
            self.is_recording = False
            audio_data = self.orchestrator.stt.stop_recording()

            # Submit to async event loop
            if hasattr(self.orchestrator, 'event_loop') and self.orchestrator.event_loop:
                asyncio.run_coroutine_threadsafe(
                    self._process_turn_async(audio_data),
                    self.orchestrator.event_loop
                )

    async def _process_turn_async(self, audio_data):
        """Process turn in async context."""
        try:
            await self.orchestrator.process_turn(audio_data)
        except Exception as e:
            print(f"Turn processing error: {e}")

    def draw_button(self, rect: pygame.Rect, text: str, font: pygame.freetype.Font,
                   active: bool = False, hover: bool = False):
        """Draw a button with hover and active states."""
        if active:
            color = self.button_active_color
        elif hover:
            color = self.button_hover_color
        else:
            color = self.button_color

        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.text_color, rect, 2, border_radius=8)

        # Center text
        text_surface, text_rect = font.render(text, self.text_color)
        text_rect.center = rect.center
        self.screen.blit(text_surface, text_rect)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.orchestrator.stop()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.space_pressed:
                        self.space_pressed = True
                        self.handle_speak_button_press()
                elif event.key == pygame.K_TAB:
                    self.show_dashboard = not self.show_dashboard

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    if self.space_pressed:
                        self.space_pressed = False
                        self.handle_speak_button_release()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.speak_button_rect.collidepoint(event.pos):
                        self.handle_speak_button_press()
                    elif self.summary_button_rect.collidepoint(event.pos):
                        self.show_session_summary()
                    elif self.stop_button_rect.collidepoint(event.pos):
                        self.running = False
                        self.orchestrator.stop()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.speak_button_rect.collidepoint(event.pos):
                    self.handle_speak_button_release()

            elif event.type == pygame.MOUSEMOTION:
                # Update hover states
                self.button_hover = None
                if self.speak_button_rect.collidepoint(event.pos):
                    self.button_hover = 'speak'
                elif self.summary_button_rect.collidepoint(event.pos):
                    self.button_hover = 'summary'
                elif self.stop_button_rect.collidepoint(event.pos):
                    self.button_hover = 'stop'

    def process_updates(self):
        """Process queued updates."""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()

                if update_type == 'engagement':
                    self.engagement_widget.update(data)

                elif update_type == 'transcript':
                    if data.startswith("User:"):
                        text = data[5:].strip()
                        if text and text != '[Silent]':
                            self.add_message(text, is_user=True)
                    elif data.startswith("Assistant:"):
                        text = data[10:].strip()
                        self.add_message(text, is_user=False)

                elif update_type == 'countdown':
                    pass  # Could add countdown display

                elif update_type == 'strategy_update':
                    self.latest_strategy, self.latest_reward = data
                    if self.show_dashboard:
                        self.bandit_dashboard.update(
                            self.orchestrator.bandit_agent,
                            self.latest_strategy,
                            self.latest_reward,
                        )

        except queue.Empty:
            pass

    def show_session_summary(self):
        """Show session summary in console (could be enhanced)."""
        summary = self.orchestrator.get_session_summary()
        text = self._format_summary_text(summary)
        print(text)

    def _format_summary_text(self, summary: dict) -> str:
        """Format summary dictionary as readable text."""
        text = "SESSION SUMMARY\n" + "=" * 50 + "\n\n"

        # Session info (keep existing)
        session_info = summary["session_info"]
        text += f"Total Turns: {session_info['total_turns']}\n"
        text += f"Session Duration: {session_info['session_duration']:.1f} seconds\n"
        text += f"Final Engagement: {session_info['final_engagement']:.3f}\n\n"

        # Bandit Performance
        bandit_perf = summary["bandit_performance"]
        text += f"AVERAGE RECENT REWARD: {bandit_perf['average_recent_reward']:.3f}\n\n"

        # Component analysis
        text += "🎯 COMPONENT PERFORMANCE:\n"
        for component, data in bandit_perf['components'].items():
            text += f"\n📊 {component.upper()}:\n"
            text += f"   Best Choice: {data['best_choice']} (score: {data['best_score']:.3f})\n"
            usage_stats = data['usage_stats']
            sorted_arms = sorted(usage_stats.items(), key=lambda x: x[1]['average_reward'], reverse=True)
            text += f"   Top Performers:\n"
            for i, (arm, stats) in enumerate(sorted_arms[:3]):
                text += f"     {i+1}. {arm}: {stats['average_reward']:.3f} avg "
                text += f"({stats['usage_count']} uses, {stats['success_rate']:.1%} success)\n"

        # Restart information
        restart_stats = bandit_perf['restart_stats']
        text += f"\n🔄 ADAPTIVE RESTARTS:\n"
        text += f"   Total Restarts: {restart_stats['total_restarts']}\n"
        text += f"   Last Restart: Step {restart_stats['last_restart_step']}\n"

        return text

    def update_audio_level(self):
        """Update audio level for sphere visualization."""
        # Simple audio level simulation - could be enhanced with actual audio analysis
        if pygame.mixer.music.get_busy():
            self.audio_level = random.uniform(0.3, 0.8)
        else:
            self.audio_level *= 0.95  # Decay when no audio

        self.sphere.set_audio_level(self.audio_level)

    def draw(self):
        """Main draw function."""
        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw sphere visualization
        self.sphere.draw(self.screen)

        # Draw engagement widget
        self.engagement_widget.draw(self.screen)

        # Draw messages
        for message in self.messages:
            if message and message.active:
                message.update()
                message.draw(self.screen)

        if self.current_message:
            self.current_message.update()
            self.current_message.draw(self.screen)

        # Draw buttons
        self.draw_button(
            self.speak_button_rect,
            "Recording..." if self.is_recording else "Hold to Speak (Space)",
            self.font_medium,
            active=self.is_recording,
            hover=self.button_hover == 'speak'
        )

        self.draw_button(
            self.summary_button_rect,
            "Summary",
            self.font_small,
            hover=self.button_hover == 'summary'
        )

        self.draw_button(
            self.stop_button_rect,
            "Stop",
            self.font_small,
            hover=self.button_hover == 'stop'
        )

        # Draw engagement value at top
        if hasattr(self.orchestrator.engagement_scorer, 'current_engagement'):
            engagement_text = f"Engagement: {self.orchestrator.engagement_scorer.current_engagement:.3f}"
            self.font_small.render_to(self.screen, (20, 20), engagement_text, self.text_color)

        # Draw bandit dashboard if enabled
        if self.show_dashboard:
            self.bandit_dashboard.draw(self.screen, self.orchestrator.bandit_agent)
            hint_text = "Press TAB to toggle dashboard"
            self.font_small.render_to(
                self.screen,
                (self.screen_width - 200, self.screen_height - 20),
                hint_text,
                (150, 150, 150),
            )

        # Draw instructions
        instructions = [
            "Hold SPACE or click button to speak",
            "Watch the sphere react to the conversation",
            "Green plot shows real-time engagement"
        ]

        for i, instruction in enumerate(instructions):
            self.font_small.render_to(
                self.screen,
                (20, self.screen_height - 80 + i * 20),
                instruction,
                (150, 150, 150)
            )

        pygame.display.flip()

    def run(self):
        """Main UI loop."""
        # Initial greeting
        self.add_message("Hello! I'm ready to chat. Hold space to speak.", is_user=False)

        while self.running:
            self.handle_events()
            self.process_updates()
            self.update_audio_level()
            self.draw()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()


# Update the main.py to use the new UI
def create_pygame_ui(orchestrator):
    """Create and run the pygame UI."""
    ui = PygameConversationUI(orchestrator)

    # Run async session in background
    async def session_runner():
        from bleak import BleakClient
        async with BleakClient(orchestrator.eeg_manager.device_address) as client:
            await orchestrator.run_session(client)

    # Start session in background thread
    def run_async_session():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(session_runner())

    session_thread = threading.Thread(target=run_async_session)
    session_thread.daemon = True
    session_thread.start()

    # Run UI
    ui.run()

    # Cleanup
    orchestrator.cleanup()