"""Pygame-based UI with sphere visualization and conversation interface."""

import pygame
import pygame.freetype
import math
import time
import random
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
from collections import deque
from typing import Optional, List, Dict, Tuple
import re
import asyncio
import threading
import queue
import logging

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    EEG_SAMPLE_RATE,
    BETA_BAND,
)
from core.orchestrator import ConversationOrchestrator
from ui.bandit_visualizer import BanditVisualizationDashboard
import math
import random
from typing import Dict, List, Tuple, Optional

FONT_PATH = "DejaVuSans.ttf"      # relative path to the TTF you downloaded

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
        """Render the engagement plot on a transparent surface with a semi-transparent grid of dots and border."""
        # Fully clear (transparent RGBA)
        self.surface.fill((0, 0, 0, 0))

        # Semi-transparent dot field
        dot_spacing = 10
        dot_radius = 1
        dot_color = (0, 160, 0, 128)  # 50 % alpha

        for y in range(0, self.rect.height, dot_spacing):
            for x in range(0, self.rect.width, dot_spacing):
                pygame.draw.circle(self.surface, dot_color, (x, y), dot_radius)

        # Engagement waveform
        if len(self.engagement_history) > 1:
            points = [
                (i, int(self.rect.height - v * self.rect.height))
                for i, v in enumerate(self.engagement_history)
            ]
            if len(points) > 1:
                pygame.draw.lines(self.surface, (0, 255, 100), False, points, 2)

                # Soft glow
                for i in range(3):
                    glow = pygame.Surface(
                        (self.rect.width, self.rect.height), pygame.SRCALPHA
                    )
                    pygame.draw.lines(glow, (0, 255, 100, 50), False, points, 4 + i * 2)
                    self.surface.blit(glow, (0, 0))

        # Current engagement value text
        if self.engagement_history:
            font = pygame.freetype.Font(None, 16)
            font.render_to(
                self.surface,
                (5, 5),
                f"{self.engagement_history[-1]:.3f}",
                (0, 255, 100),
            )

        # Semi-transparent light-green border
        border_color = (120, 255, 120, 128)  # 50 % alpha
        pygame.draw.rect(
            self.surface,
            border_color,
            self.surface.get_rect(),
            width=2,
            border_radius=8,
        )

        # Blit to main screen
        screen.blit(self.surface, self.rect)


class EEGPlotWidget:
    """Realtime plot of raw or filtered EEG signal."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.raw_ch1 = deque(maxlen=EEG_SAMPLE_RATE * 3)
        self.raw_ch2 = deque(maxlen=EEG_SAMPLE_RATE * 3)
        self.filt_ch1 = deque(maxlen=EEG_SAMPLE_RATE * 3)
        self.filt_ch2 = deque(maxlen=EEG_SAMPLE_RATE * 3)

        nyq = EEG_SAMPLE_RATE / 2
        sos = signal.butter(
            4,
            [BETA_BAND[0] / nyq, BETA_BAND[1] / nyq],
            btype="band",
            output="sos",
        )
        self.sos = sos
        self.zi_ch1 = signal.sosfilt_zi(sos)
        self.zi_ch2 = signal.sosfilt_zi(sos)

        self.visible = False
        self.mode = "raw"  # 'raw' or 'filtered'

    def add_data(self, ch1: List[float], ch2: List[float]):
        self.raw_ch1.extend(ch1)
        self.raw_ch2.extend(ch2)

        ch1_arr = np.array(ch1)
        ch2_arr = np.array(ch2)
        filt1, self.zi_ch1 = signal.sosfilt(self.sos, ch1_arr, zi=self.zi_ch1)
        filt2, self.zi_ch2 = signal.sosfilt(self.sos, ch2_arr, zi=self.zi_ch2)
        self.filt_ch1.extend(filt1)
        self.filt_ch2.extend(filt2)

    def draw(self, screen: pygame.Surface):
        if not self.visible:
            return

        self.surface.fill((0, 0, 0, 180))
        data1 = (
            list(self.raw_ch1)
            if self.mode == "raw"
            else list(self.filt_ch1)
        )
        data2 = (
            list(self.raw_ch2)
            if self.mode == "raw"
            else list(self.filt_ch2)
        )

        if len(data1) < 2:
            screen.blit(self.surface, self.rect)
            return

        max_val = max(
            max(np.abs(data1)) if data1 else 1,
            max(np.abs(data2)) if data2 else 1,
            1,
        )

        step = max(1, len(data1) // self.rect.width)
        points1 = []
        points2 = []
        for i in range(0, len(data1), step):
            x = int(i / len(data1) * self.rect.width)
            y1 = int(self.rect.height / 2 - (data1[i] / max_val) * (self.rect.height / 2))
            y2 = int(self.rect.height / 2 - (data2[i] / max_val) * (self.rect.height / 2))
            points1.append((x, y1))
            points2.append((x, y2))

        if len(points1) > 1:
            pygame.draw.lines(self.surface, (0, 255, 0), False, points1, 1)
        if len(points2) > 1:
            pygame.draw.lines(self.surface, (255, 0, 0), False, points2, 1)

        label = "RAW" if self.mode == "raw" else "FILTERED"
        font = pygame.freetype.Font(None, 14)
        font.render_to(self.surface, (5, 5), label, (200, 200, 200))

        pygame.draw.rect(self.surface, (180, 255, 180), self.surface.get_rect(), 1)

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
        self.font = pygame.freetype.Font(FONT_PATH, self.font_size)

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
        # self.shades = ' .:-=+*#%@'
        self.shades = ' .○⊙⊗◍◉●'

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

        # Use orchestrator logger if available
        self.logger = getattr(self.orchestrator, "logger", logging.getLogger(__name__))

        # Initialize pygame
        pygame.init()
        pygame.freetype.init()
        pygame.mixer.init()

        # Setup display
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self._layout_buttons()

        pygame.display.set_caption('neocore AI Therapist - POC')

        # Colors
        self.bg_color = (10, 15, 20)
        self.text_color = (200, 255, 200)
        self.button_color = (30, 60, 40)
        self.button_hover_color = (50, 100, 60)
        self.button_active_color = (100, 50, 50)

        self.border_color = (180, 255, 180)  # light-green outline
        self.mic_icon = "🎤"  # U+1F3A4

        # Fonts
        self.font_large = pygame.freetype.Font(None, 24)
        self.font_medium = pygame.freetype.Font(None, 18)
        self.font_small = pygame.freetype.Font(None, 14)

        # UI Components
        self.sphere = SphereVisualization(self.screen_width // 2, 250)
        self.engagement_widget = EngagementWidget(self.screen_width - 220, 20, 200, 120)
        self.eeg_widget = EEGPlotWidget(20, 20, 300, 120)

        self.concept_widget = ConceptVisualizationWidget(
            self.screen_width - 450, 150, 430, 400
        )
        # Bandit visualization dashboard
        self.bandit_dashboard = BanditVisualizationDashboard(self.screen_width, self.screen_height)
        self.show_dashboard = True

        # Track latest strategy performance
        self.latest_strategy = None
        self.latest_reward = None

        # Message display
        self.current_message = None
        self.message_y = 540
        self.messages = deque(maxlen=5)


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
            'update_eeg': self._queue_eeg_update,
        }

        # Audio management
        self.current_audio_file = None
        self.audio_level = 0.0

        self.orchestrator.ui_callbacks.update({
            'save_word_cloud': self.save_word_cloud,
        })

    def _layout_buttons(self):
        """Create the Speak, Summary and Stop buttons as one centered stack."""
        # Main button position and size
        speak_w, speak_h = 260, 70
        speak_x = self.screen_width // 2 - speak_w // 2
        speak_y = 620  # was 500: lower = further down

        # Build Speak
        self.speak_button_rect = pygame.Rect(speak_x, speak_y, speak_w, speak_h)

        # Summary and Stop share 105 % of Speak width
        total_w = int(speak_w * 1.05)
        child_w = total_w // 2
        child_h = 30
        left = self.speak_button_rect.centerx - total_w // 2
        child_y = self.speak_button_rect.bottom + 14  # gap under Speak

        self.summary_button_rect = pygame.Rect(left, child_y, child_w, child_h)
        self.stop_button_rect = pygame.Rect(left + child_w, child_y, child_w, child_h)

        # Guide text centred under Stop
        self.guide_pos = (
            self.speak_button_rect.centerx,
            self.stop_button_rect.bottom + 12
        )

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

    def _queue_eeg_update(self, data):
        """Queue raw EEG samples for plotting."""
        self.update_queue.put(('eeg', data))

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
        color =(200, 255, 210) if is_user else (163, 255, 181)
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
        """Handle speak button press with interruption support."""
        if not self.is_recording:
            if self.orchestrator.ai_speaking:
                self.logger.info("Interrupting AI to let user speak")
                self.orchestrator.interrupt_ai_speech()
                pygame.time.wait(100)

            self.is_recording = True
            self.orchestrator.cancel_auto_advance_timer()
            # Small delay to ensure cancellation takes effect
            pygame.time.wait(50)
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
            # Give UI time to process the strategy update
            await asyncio.sleep(0.1)  # ← ADD THIS
        except Exception as e:
            print(f"Turn processing error: {e}")

    def draw_button(
            self,
            rect: pygame.Rect,
            text: str,
            font: pygame.freetype.Font,
            *,
            icon: str | None = None,
            filled: bool = True,
            active: bool = False,
            hover: bool = False,
    ):
        # Optional solid fill
        if filled:
            if active:
                fill = self.button_active_color
            elif hover:
                fill = self.button_hover_color
            else:
                fill = self.button_color
            pygame.draw.rect(self.screen, fill, rect, border_radius=14)

        # 70 % transparent light-green border
        border_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        border_color = (180, 255, 180, 178)  # alpha 178 ≈ 70 %
        pygame.draw.rect(border_surf, border_color, border_surf.get_rect(),
                         width=2, border_radius=14)  # rounder radius (14)
        self.screen.blit(border_surf, rect.topleft)

        # Render icon and text
        if icon:
            icon_surf, icon_rect = font.render(icon, self.text_color)
            label_surf, label_rect = font.render(text, self.text_color)

            gap = 8
            total_w = icon_rect.width + gap + label_rect.width
            start_x = rect.centerx - total_w // 2
            centre_y = rect.centery

            icon_rect.topleft = (start_x, centre_y - icon_rect.height // 2)
            label_rect.topleft = (start_x + icon_rect.width + gap,
                                  centre_y - label_rect.height // 2)

            self.screen.blit(icon_surf, icon_rect)
            self.screen.blit(label_surf, label_rect)
        else:
            label_surf, label_rect = font.render(text, self.text_color)
            label_rect.center = rect.center
            self.screen.blit(label_surf, label_rect)

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
                        # This will now handle interruption automatically
                        self.handle_speak_button_press()
                elif event.key == pygame.K_q:  # NEW
                    self.concept_widget.toggle_visibility()
                elif event.key == pygame.K_TAB:
                    self.show_dashboard = not self.show_dashboard
                elif event.key == pygame.K_p:
                    if self.eeg_widget.visible and self.eeg_widget.mode == "raw":
                        self.eeg_widget.visible = False
                    else:
                        self.eeg_widget.mode = "raw"
                        self.eeg_widget.visible = True
                elif event.key == pygame.K_f:
                    if self.eeg_widget.visible and self.eeg_widget.mode == "filtered":
                        self.eeg_widget.visible = False
                    else:
                        self.eeg_widget.mode = "filtered"
                        self.eeg_widget.visible = True

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
        updates_processed = 0
        max_updates_per_frame = 100  # Prevent queue overflow

        try:
            while updates_processed < max_updates_per_frame:
                update_type, data = self.update_queue.get_nowait()
                updates_processed += 1

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
                    strategy, reward = data
                    self.latest_strategy = strategy
                    self.latest_reward = reward

                    # Always update dashboard when we get a strategy update
                    if self.show_dashboard and strategy is not None:
                        self.bandit_dashboard.update(
                            self.orchestrator.bandit_agent,
                            strategy,
                            reward,
                        )
                        # Log for debugging
                        if reward is not None:
                            print(f"Dashboard updated: {strategy.tone} with reward {reward:.3f}")

                elif update_type == 'eeg':
                    ch1, ch2 = data
                    self.eeg_widget.add_data(ch1, ch2)

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
        # Check both pygame audio AND orchestrator AI speaking state
        if pygame.mixer.music.get_busy() or self.orchestrator.ai_speaking:
            self.audio_level = random.uniform(0.3, 1)
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
        self.eeg_widget.draw(self.screen)
        # NEW: Update and draw concept widget
        if hasattr(self.orchestrator, 'therapeutic_manager'):
            concept_tracker = self.orchestrator.therapeutic_manager.concept_tracker
            self.concept_widget.update(concept_tracker)
            self.concept_widget.draw(self.screen, concept_tracker)
        # Draw messages
        for message in self.messages:
            if message and message.active:
                message.update()
                message.draw(self.screen)

        if self.current_message:
            self.current_message.update()
            self.current_message.draw(self.screen)

        # main “Hold to Speak” button keeps its fill
        self.draw_button(
            self.speak_button_rect,
            "Recording..." if self.is_recording else "Hold to Speak (Space)",
            self.font_medium,
            icon=self.mic_icon,  # ← mic in front
            active=self.is_recording,
            hover=self.button_hover == 'speak',
            filled=True,
        )

        # child buttons: outline only
        self.draw_button(
            self.summary_button_rect,
            "Summary",
            self.font_small,
            hover=self.button_hover == 'summary',
            filled=False,
        )

        self.draw_button(
            self.stop_button_rect,
            "Stop",
            self.font_small,
            hover=self.button_hover == 'stop',
            filled=False,
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

        guide_text = "Hold SPACE or click to speak"
        text_surf, text_rect = self.font_small.render(guide_text, (150, 150, 150))
        text_rect.center = self.guide_pos
        self.screen.blit(text_surf, text_rect)

        pygame.display.flip()

    def run(self):
        """Main UI loop."""
        # Initial greeting
        self.add_message("Hello! I'm ready to chat. Hold space to speak.", is_user=False)

        while self.running:
            self.handle_events()

            # Process updates MORE frequently
            self.process_updates()  # ← Already called once
            pygame.time.wait(10)  # ← ADD small wait
            self.process_updates()  # ← Call again

            self.update_audio_level()
            self.draw()
            self.clock.tick(60)

        pygame.quit()

    def save_word_cloud(self, filepath: str):
        """Save the current word cloud visualization."""
        if hasattr(self.orchestrator, 'therapeutic_manager'):
            concept_tracker = self.orchestrator.therapeutic_manager.concept_tracker
            self.concept_widget.save_as_image(filepath, concept_tracker)
        else:
            print("⚠️ No therapeutic manager available for word cloud")


class ConceptVisualizationWidget:
    """Real-time concept tracking with emotional word cloud."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.visible = False

        # Fonts for different sizes
        self.font_large = pygame.freetype.Font(FONT_PATH, 24)
        self.font_medium = pygame.freetype.Font(FONT_PATH, 18)
        self.font_small = pygame.freetype.Font(FONT_PATH, 14)
        self.font_tiny = pygame.freetype.Font(FONT_PATH, 12)

        # Animation
        self.pulse_timer = 0
        self.fade_alpha = 0
        self.target_alpha = 0

        # Word positioning
        self.word_positions = {}
        self.last_update_time = 0

    def toggle_visibility(self):
        """Toggle widget visibility with fade animation."""
        self.visible = not self.visible
        self.target_alpha = 255 if self.visible else 0

    def get_concept_color(self, avg_engagement: float, avg_emotion: float, emotional_intensity: float) -> Tuple[
        int, int, int]:
        """Get color based ONLY on emotion: red=negative, gray=neutral, green=positive."""

        if avg_emotion < 0.4:  # Negative emotion
            # Scale red intensity based on how negative (0.4 to 0.0 maps to light red to bright red)
            intensity = int(255 * (0.4 - avg_emotion) / 0.4)  # 0.4→0, 0.0→255
            return (255, max(50, 255 - intensity), max(50, 255 - intensity))  # Red with some green/blue

        elif avg_emotion > 0.6:  # Positive emotion
            # Scale green intensity based on how positive (0.6 to 1.0 maps to light green to bright green)
            intensity = int(255 * (avg_emotion - 0.6) / 0.4)  # 0.6→0, 1.0→255
            return (max(50, 255 - intensity), 255, max(50, 255 - intensity))  # Green with some red/blue

        else:  # Neutral emotion (0.4 to 0.6)
            return (180, 180, 180)  # Gray

    def get_font_size(self, avg_engagement: float, emotional_intensity: float) -> pygame.freetype.Font:
        """Get font size based ONLY on engagement."""

        if avg_engagement > 0.8:
            return self.font_large  # 24px
        elif avg_engagement > 0.6:
            return self.font_medium  # 18px
        elif avg_engagement > 0.4:
            return self.font_small  # 14px
        else:
            return self.font_tiny  # 12px

    def calculate_word_position(self, word: str, font: pygame.freetype.Font, attempt: int = 0) -> Tuple[int, int]:
        """Calculate non-overlapping position for word."""
        text_surface, text_rect = font.render(word, (255, 255, 255))
        word_width, word_height = text_rect.width, text_rect.height

        # Try to find non-overlapping position
        max_attempts = 20
        for _ in range(max_attempts):
            if attempt == 0:  # First placement - try center area
                x = random.randint(self.rect.width // 4, 3 * self.rect.width // 4 - word_width)
                y = random.randint(self.rect.height // 4, 3 * self.rect.height // 4 - word_height)
            else:  # Subsequent attempts - anywhere
                x = random.randint(10, self.rect.width - word_width - 10)
                y = random.randint(30, self.rect.height - word_height - 10)  # Leave space for title

            # Check for overlaps with existing words
            new_rect = pygame.Rect(x, y, word_width, word_height)
            overlap = False

            for existing_word, (ex_x, ex_y, ex_w, ex_h) in self.word_positions.items():
                existing_rect = pygame.Rect(ex_x, ex_y, ex_w, ex_h)
                if new_rect.colliderect(existing_rect):
                    overlap = True
                    break

            if not overlap:
                return x, y

        # If no non-overlapping position found, return random position
        return (random.randint(10, self.rect.width - word_width - 10),
                random.randint(30, self.rect.height - word_height - 10))

    def update(self, concept_tracker):
        """Update concept visualization with latest data."""
        current_time = time.time()
        self.pulse_timer += 1

        # Update fade animation
        if self.fade_alpha < self.target_alpha:
            self.fade_alpha = min(self.fade_alpha + 15, self.target_alpha)
        elif self.fade_alpha > self.target_alpha:
            self.fade_alpha = max(self.fade_alpha - 15, self.target_alpha)

        # FIXED: Only clean up positions for concepts that no longer exist
        # Don't constantly clear all positions
        if hasattr(concept_tracker, 'concept_activations'):
            current_concepts = set(concept_tracker.concept_activations.keys())
            # Remove positions for concepts that no longer exist
            self.word_positions = {
                concept: pos for concept, pos in self.word_positions.items()
                if concept in current_concepts
            }

    def draw(self, screen: pygame.Surface, concept_tracker):
        """Draw the concept visualization."""
        if not self.visible or self.fade_alpha <= 0:
            return

        # Clear surface
        self.surface.fill((0, 0, 0, 0))

        # Draw semi-transparent background
        bg_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        bg_surface.fill((20, 25, 30, 200))
        self.surface.blit(bg_surface, (0, 0))

        # Draw border
        border_color = (100, 255, 150, 180)
        pygame.draw.rect(self.surface, border_color, self.surface.get_rect(), width=2, border_radius=8)

        # Title
        title_font = pygame.freetype.Font(None, 16)
        title_font.render_to(self.surface, (10, 8), "CONCEPT TRACKER (Q to toggle)", (150, 255, 150))

        # Get concept data
        if not hasattr(concept_tracker, 'concept_activations'):
            # No data yet
            no_data_font = pygame.freetype.Font(None, 14)
            no_data_font.render_to(self.surface, (20, 40), "No concepts tracked yet...", (150, 150, 150))
            self._draw_legend()
        else:
            # Sort concepts by importance (engagement * intensity) for consistent ordering
            concept_items = []
            for concept, engagement_scores in concept_tracker.concept_activations.items():
                if len(engagement_scores) >= 1:
                    avg_engagement = np.mean(engagement_scores)

                    # Get emotional data
                    if hasattr(concept_tracker, 'concept_emotions') and concept in concept_tracker.concept_emotions:
                        emotion_scores = concept_tracker.concept_emotions[concept]
                        emotional_intensity = np.mean([abs(e - 0.5) * 2 for e in emotion_scores])
                    else:
                        emotional_intensity = 0.0

                    importance = avg_engagement * (1 + emotional_intensity)
                    concept_items.append((concept, importance, avg_engagement, emotional_intensity))

            # Sort by importance (most important first) for consistent ordering
            concept_items.sort(key=lambda x: x[1], reverse=True)

            # Draw concepts
            max_concepts = 15
            for i, (concept, importance, avg_engagement, emotional_intensity) in enumerate(
                    concept_items[:max_concepts]):
                # Get emotional data
                if hasattr(concept_tracker, 'concept_emotions') and concept in concept_tracker.concept_emotions:
                    emotion_scores = concept_tracker.concept_emotions[concept]
                    avg_emotion = np.mean(emotion_scores)
                else:
                    avg_emotion = 0.5

                # Get color and font
                color = self.get_concept_color(avg_engagement, avg_emotion, emotional_intensity)
                font = self.get_font_size(avg_engagement, emotional_intensity)

                # Add pulsing effect for high-intensity concepts
                if avg_engagement > 0.7 and emotional_intensity > 0.6:
                    pulse = math.sin(self.pulse_timer * 0.1) * 0.3 + 0.7
                    color = tuple(int(c * pulse) for c in color)

                # FIXED: Only calculate position if concept doesn't have one yet
                if concept not in self.word_positions:
                    text_surface, text_rect = font.render(concept, color)
                    x, y = self.calculate_word_position(concept, font, i)
                    self.word_positions[concept] = (x, y, text_rect.width, text_rect.height)

                # Draw word at its STABLE position
                x, y, w, h = self.word_positions[concept]
                font.render_to(self.surface, (x, y), concept, color)

                # Draw small stats
                engagement_scores = concept_tracker.concept_activations[concept]
                stats_text = f"e:{avg_engagement:.2f} m:{len(engagement_scores)}"
                stats_font = pygame.freetype.Font(None, 10)
                stats_font.render_to(self.surface, (x, y + h + 2), stats_text, (100, 100, 100))

            # Draw legend
            self._draw_legend()

        # Apply fade alpha
        if self.fade_alpha < 255:
            fade_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            fade_surface.fill((0, 0, 0, 255 - self.fade_alpha))
            self.surface.blit(fade_surface, (0, 0))

        # Blit to screen
        screen.blit(self.surface, self.rect)

    def _draw_legend(self):
        """Draw simplified color legend."""
        legend_y = self.rect.height - 60
        legend_font = pygame.freetype.Font(None, 10)

        legend_items = [
            ("SIZE = Engagement Level", (255, 255, 255)),
            ("Red = Negative Emotion", (255, 100, 100)),
            ("Gray = Neutral Emotion", (180, 180, 180)),
            ("Green = Positive Emotion", (100, 255, 100))
        ]

        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 12
            if i == 0:  # First item is just text
                legend_font.render_to(self.surface, (10, y_pos), label, color)
            else:
                # Draw color square
                color_rect = pygame.Rect(10, y_pos, 8, 8)
                pygame.draw.rect(self.surface, color, color_rect)
                # Draw label
                legend_font.render_to(self.surface, (22, y_pos), label, (180, 180, 180))

    def save_as_image(self, filepath: str, concept_tracker):
        pygame.freetype.init()
        """Save the current concept visualization as an image."""
        # Create a clean surface for saving
        save_surface = pygame.Surface((self.rect.width, self.rect.height))
        save_surface.fill((20, 25, 30))  # Dark background

        # Force visibility for rendering
        original_visible = self.visible
        original_alpha = self.fade_alpha
        self.visible = True
        self.fade_alpha = 255

        # Create a temporary surface to draw on
        temp_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        self.surface = temp_surface

        # Draw the visualization
        self._draw_concept_content(concept_tracker)

        # Blit to save surface
        save_surface.blit(temp_surface, (0, 0))

        # Save the image
        pygame.image.save(save_surface, filepath)

        # Restore original state
        self.visible = original_visible
        self.fade_alpha = original_alpha

        print(f"💾 Word cloud saved to: {filepath}")

    def _draw_concept_content(self, concept_tracker):
        """Helper method to draw concept content (extracted from draw method)."""
        # Clear surface
        self.surface.fill((0, 0, 0, 0))

        # Draw semi-transparent background
        bg_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        bg_surface.fill((20, 25, 30, 220))
        self.surface.blit(bg_surface, (0, 0))

        # Draw border
        border_color = (100, 255, 150, 180)
        pygame.draw.rect(self.surface, border_color, self.surface.get_rect(), width=2, border_radius=8)

        # Title with timestamp
        from datetime import datetime
        title_font = pygame.freetype.Font(None, 16)
        title_text = f"CONCEPT TRACKER - Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        title_font.render_to(self.surface, (10, 8), title_text, (150, 255, 150))

        # Get concept data
        if not hasattr(concept_tracker, 'concept_activations'):
            no_data_font = pygame.freetype.Font(None, 14)
            no_data_font.render_to(self.surface, (20, 40), "No concepts tracked during session", (150, 150, 150))
            self._draw_legend()
            return

        # Sort concepts by importance
        concept_items = []
        for concept, engagement_scores in concept_tracker.concept_activations.items():
            if len(engagement_scores) >= 1:
                avg_engagement = np.mean(engagement_scores)

                if hasattr(concept_tracker, 'concept_emotions') and concept in concept_tracker.concept_emotions:
                    emotion_scores = concept_tracker.concept_emotions[concept]
                    emotional_intensity = np.mean([abs(e - 0.5) * 2 for e in emotion_scores])
                else:
                    emotional_intensity = 0.0

                importance = avg_engagement * (1 + emotional_intensity)
                concept_items.append((concept, importance, avg_engagement, emotional_intensity))

        concept_items.sort(key=lambda x: x[1], reverse=True)

        # Draw concepts with stable positions
        max_concepts = 20  # Show more concepts in saved image
        for i, (concept, importance, avg_engagement, emotional_intensity) in enumerate(concept_items[:max_concepts]):
            # Get emotional data
            if hasattr(concept_tracker, 'concept_emotions') and concept in concept_tracker.concept_emotions:
                emotion_scores = concept_tracker.concept_emotions[concept]
                avg_emotion = np.mean(emotion_scores)
            else:
                avg_emotion = 0.5

            # Get color and font
            color = self.get_concept_color(avg_engagement, avg_emotion, emotional_intensity)
            font = self.get_font_size(avg_engagement, emotional_intensity)

            # Calculate position if not exists
            if concept not in self.word_positions:
                text_surface, text_rect = font.render(concept, color)
                x, y = self.calculate_word_position(concept, font, i)
                self.word_positions[concept] = (x, y, text_rect.width, text_rect.height)

            # Draw word
            x, y, w, h = self.word_positions[concept]
            font.render_to(self.surface, (x, y), concept, color)

            # Draw detailed stats for saved image
            engagement_scores = concept_tracker.concept_activations[concept]
            stats_text = f"eng:{avg_engagement:.2f} emo:{avg_emotion:.2f} cnt:{len(engagement_scores)}"
            stats_font = pygame.freetype.Font(None, 10)
            stats_font.render_to(self.surface, (x, y + h + 2), stats_text, (120, 120, 120))

        # Draw legend
        self._draw_legend()

        # Add session stats
        self._draw_session_stats(concept_tracker)

    def _draw_session_stats(self, concept_tracker):
        """Draw session statistics on the saved image."""
        stats_y = self.rect.height - 120
        stats_font = pygame.freetype.Font(None, 12)

        # Session summary
        total_concepts = len(concept_tracker.concept_activations)
        total_mentions = sum(len(scores) for scores in concept_tracker.concept_activations.values())

        if hasattr(concept_tracker, 'concept_emotions'):
            avg_session_emotion = np.mean([
                np.mean(emotions) for emotions in concept_tracker.concept_emotions.values()
                if len(emotions) > 0
            ]) if concept_tracker.concept_emotions else 0.5
        else:
            avg_session_emotion = 0.5

        stats_lines = [
            f"SESSION SUMMARY:",
            f"Total Concepts: {total_concepts}",
            f"Total Mentions: {total_mentions}",
            f"Avg Session Emotion: {avg_session_emotion:.2f}",
        ]

        for i, line in enumerate(stats_lines):
            color = (200, 200, 200) if i == 0 else (150, 150, 150)
            stats_font.render_to(self.surface, (10, stats_y + i * 14), line, color)

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