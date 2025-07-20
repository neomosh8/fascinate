"""
Simplified Real-time Bandit Visualization Dashboard
Drop-in replacement for ui/bandit_visualizer.py
"""

import pygame
import pygame.freetype
import numpy as np
import math
from collections import deque
from typing import Dict, List, Tuple, Optional


class RealTimeBanditDashboard:
    """Simplified comprehensive real-time bandit visualization."""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Colors
        self.bg_color = (15, 20, 25, 220)  # Semi-transparent dark
        self.text_color = (200, 255, 200)
        self.accent_color = (100, 255, 150)
        self.warning_color = (255, 150, 100)
        self.good_color = (100, 255, 100)
        self.bad_color = (255, 100, 100)

        # Fonts
        self.font_title = pygame.freetype.Font(None, 20)
        self.font_large = pygame.freetype.Font(None, 16)
        self.font_medium = pygame.freetype.Font(None, 14)
        self.font_small = pygame.freetype.Font(None, 12)

        # Layout
        self.margin = 20
        self.panel_spacing = 15

        # Data tracking
        self.reward_history = deque(maxlen=50)
        self.strategy_history = deque(maxlen=20)
        self.current_strategy = None
        self.current_reward = None
        self.last_update_time = 0

        # Animation
        self.pulse_timer = 0
        self.highlight_strategy = None
        self.highlight_timer = 0

    def update(self, bandit_agent, latest_strategy=None, latest_reward=None):
        """Update dashboard with latest data."""
        if latest_strategy:
            self.current_strategy = latest_strategy
            self.highlight_strategy = latest_strategy

            if latest_reward is None:
                # Indicate strategy is currently being spoken
                self.current_reward = "SPEAKING..."
                self.highlight_timer = 120
            else:
                self.current_reward = latest_reward
                self.reward_history.append(latest_reward)
                self.strategy_history.append((latest_strategy, latest_reward))
                self.highlight_timer = 60  # Frames to highlight

        self.pulse_timer += 1
        if self.highlight_timer > 0:
            self.highlight_timer -= 1

    def draw(self, screen: pygame.Surface, bandit_agent):
        """Draw the complete dashboard."""
        # Create semi-transparent overlay
        overlay = pygame.Surface(
            (self.screen_width, self.screen_height), pygame.SRCALPHA
        )
        overlay.fill(self.bg_color)
        screen.blit(overlay, (0, 0))

        # Calculate layout
        panel_width = (self.screen_width - 4 * self.margin) // 3
        panel_height = (self.screen_height - 4 * self.margin) // 2

        # Draw panels
        self._draw_current_strategy_panel(
            screen, self.margin, self.margin, panel_width, panel_height
        )
        self._draw_component_performance_panel(
            screen,
            self.margin * 2 + panel_width,
            self.margin,
            panel_width,
            panel_height,
            bandit_agent,
        )
        self._draw_learning_progress_panel(
            screen,
            self.margin * 3 + panel_width * 2,
            self.margin,
            panel_width,
            panel_height,
        )

        self._draw_strategy_timeline_panel(
            screen,
            self.margin,
            self.margin * 2 + panel_height,
            panel_width * 2 + self.margin,
            panel_height,
        )
        self._draw_context_panel(
            screen,
            self.margin * 3 + panel_width * 2,
            self.margin * 2 + panel_height,
            panel_width,
            panel_height,
            bandit_agent,
        )

    def _draw_panel_background(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
    ):
        """Draw panel background with title."""
        # Panel background
        panel_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, (25, 35, 45, 180), panel_rect, border_radius=8)
        pygame.draw.rect(
            screen, self.accent_color, panel_rect, width=2, border_radius=8
        )

        # Title
        self.font_title.render_to(screen, (x + 10, y + 8), title, self.accent_color)
        return y + 35  # Return content start Y

    def _draw_current_strategy_panel(
        self, screen: pygame.Surface, x: int, y: int, width: int, height: int
    ):
        """Draw current strategy selection panel."""
        content_y = self._draw_panel_background(
            screen, x, y, width, height, "CURRENT STRATEGY"
        )

        if not self.current_strategy:
            self.font_medium.render_to(
                screen,
                (x + 15, content_y + 20),
                "No strategy selected yet",
                self.text_color,
            )
            return

        # Strategy components with visual emphasis
        components = [
            ("TONE", self.current_strategy.tone),
            ("TOPIC", self.current_strategy.topic),
            ("EMOTION", self.current_strategy.emotion),
            ("HOOK", self.current_strategy.hook),
        ]

        comp_y = content_y + 10
        for i, (label, value) in enumerate(components):
            # Pulsing highlight effect
            pulse = math.sin(self.pulse_timer * 0.1) * 0.3 + 0.7
            highlight_alpha = int(255 * pulse) if self.highlight_timer > 0 else 100

            # Component background
            comp_rect = pygame.Rect(x + 10, comp_y, width - 20, 25)
            highlight_surf = pygame.Surface((width - 20, 25), pygame.SRCALPHA)
            highlight_surf.fill((100, 200, 255, highlight_alpha // 4))
            screen.blit(highlight_surf, (x + 10, comp_y))

            # Component text
            self.font_small.render_to(
                screen, (x + 15, comp_y + 2), label, (150, 150, 150)
            )

            # Value with color coding
            color = (
                self.good_color
                if isinstance(self.current_reward, (int, float))
                and self.current_reward > 0.5
                else self.text_color
            )
            self.font_medium.render_to(screen, (x + 15, comp_y + 12), value, color)

            comp_y += 30

        # Current reward or speaking indicator
        if self.current_reward == "SPEAKING...":
            reward_y = comp_y + 10
            pulse = math.sin(self.pulse_timer * 0.2) * 0.5 + 0.5
            color = (255, int(100 + 155 * pulse), 0)
            self.font_large.render_to(
                screen, (x + 15, reward_y), "\U0001f3b5 SPEAKING...", color
            )
        elif isinstance(self.current_reward, (int, float)):
            reward_y = comp_y + 10
            reward_color = (
                self.good_color if self.current_reward > 0 else self.bad_color
            )
            self.font_large.render_to(
                screen,
                (x + 15, reward_y),
                f"\U0001f4ca REWARD: {self.current_reward:.3f}",
                reward_color,
            )

    def _draw_component_performance_panel(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        width: int,
        height: int,
        bandit_agent,
    ):
        """Draw component performance with real-time bars."""
        content_y = self._draw_panel_background(
            screen, x, y, width, height, "COMPONENT PERFORMANCE"
        )

        try:
            summary = bandit_agent.get_performance_summary()
            components = summary.get("components", {})

            comp_y = content_y + 5
            bar_width = width - 40

            for comp_name in ["tone", "topic", "emotion", "hook"]:
                if comp_name not in components:
                    continue

                comp_data = components[comp_name]
                best_choice = comp_data.get("best_choice", "unknown")
                best_score = comp_data.get("best_score", 0)

                # Component name
                self.font_medium.render_to(
                    screen, (x + 15, comp_y), comp_name.upper(), self.text_color
                )

                # Performance bar
                bar_height = 8
                bar_y = comp_y + 18
                bar_rect = pygame.Rect(x + 15, bar_y, bar_width, bar_height)

                # Background
                pygame.draw.rect(screen, (50, 50, 50), bar_rect, border_radius=4)

                # Performance fill
                fill_width = int(
                    bar_width * max(0, min(1, (best_score + 1) / 2))
                )  # Normalize -1 to 1 -> 0 to 1
                if fill_width > 0:
                    fill_color = (
                        self.good_color
                        if best_score > 0.3
                        else self.warning_color if best_score > 0 else self.bad_color
                    )
                    fill_rect = pygame.Rect(x + 15, bar_y, fill_width, bar_height)
                    pygame.draw.rect(screen, fill_color, fill_rect, border_radius=4)

                # Best choice text
                choice_text = f"{best_choice} ({best_score:.2f})"
                self.font_small.render_to(
                    screen, (x + 20, bar_y + 12), choice_text, self.text_color
                )

                comp_y += 45

        except Exception as e:
            self.font_small.render_to(
                screen,
                (x + 15, content_y + 20),
                f"Error: {str(e)[:30]}...",
                self.bad_color,
            )

    def _draw_learning_progress_panel(
        self, screen: pygame.Surface, x: int, y: int, width: int, height: int
    ):
        """Draw learning progress and statistics."""
        content_y = self._draw_panel_background(
            screen, x, y, width, height, "LEARNING PROGRESS"
        )

        # Reward history graph
        if len(self.reward_history) > 1:
            graph_height = 80
            graph_y = content_y + 10

            # Graph background
            graph_rect = pygame.Rect(x + 15, graph_y, width - 30, graph_height)
            pygame.draw.rect(screen, (30, 30, 30, 100), graph_rect, border_radius=4)

            # Zero line
            zero_y = graph_y + graph_height // 2
            pygame.draw.line(
                screen, (100, 100, 100), (x + 15, zero_y), (x + width - 15, zero_y), 1
            )

            # Plot reward history
            if len(self.reward_history) > 1:
                points = []
                for i, reward in enumerate(self.reward_history):
                    px = (
                        x
                        + 15
                        + (i / max(1, len(self.reward_history) - 1)) * (width - 30)
                    )
                    py = zero_y - (
                        reward * graph_height // 4
                    )  # Scale to half graph height
                    py = max(graph_y, min(graph_y + graph_height, py))
                    points.append((px, py))

                if len(points) > 1:
                    pygame.draw.lines(screen, self.good_color, False, points, 2)

            # Statistics
            stats_y = graph_y + graph_height + 15
            recent_avg = (
                np.mean(list(self.reward_history)[-10:]) if self.reward_history else 0
            )
            total_trials = len(self.reward_history)

            self.font_small.render_to(
                screen,
                (x + 15, stats_y),
                f"Recent Avg: {recent_avg:.3f}",
                self.text_color,
            )
            self.font_small.render_to(
                screen,
                (x + 15, stats_y + 15),
                f"Total Trials: {total_trials}",
                self.text_color,
            )

            # Learning trend
            if len(self.reward_history) > 10:
                early_avg = np.mean(list(self.reward_history)[:5])
                trend = recent_avg - early_avg
                trend_color = self.good_color if trend > 0 else self.bad_color
                trend_text = f"Trend: {'↗' if trend > 0 else '↘'} {trend:+.3f}"
                self.font_small.render_to(
                    screen, (x + 15, stats_y + 30), trend_text, trend_color
                )

    def _draw_strategy_timeline_panel(
        self, screen: pygame.Surface, x: int, y: int, width: int, height: int
    ):
        """Draw recent strategy timeline."""
        content_y = self._draw_panel_background(
            screen, x, y, width, height, "STRATEGY TIMELINE"
        )

        if not self.strategy_history:
            self.font_medium.render_to(
                screen,
                (x + 15, content_y + 20),
                "No strategies tried yet",
                self.text_color,
            )
            return

        # Timeline visualization
        timeline_y = content_y + 10
        item_height = 25
        visible_items = min(6, len(self.strategy_history))

        for i, (strategy, reward) in enumerate(
            list(self.strategy_history)[-visible_items:]
        ):
            item_y = timeline_y + i * item_height

            # Strategy summary
            strategy_text = (
                f"{strategy.tone[:8]}/{strategy.topic[:8]}/{strategy.emotion[:6]}"
            )

            # Reward indicator
            reward_color = (
                self.good_color
                if reward > 0.3
                else self.warning_color if reward > 0 else self.bad_color
            )
            indicator_size = 8
            pygame.draw.circle(
                screen, reward_color, (x + 15, item_y + 10), indicator_size
            )

            # Strategy text
            text_x = x + 30
            self.font_small.render_to(
                screen, (text_x, item_y + 2), strategy_text, self.text_color
            )

            # Reward value
            reward_text = f"{reward:+.2f}"
            self.font_small.render_to(
                screen, (text_x, item_y + 13), reward_text, reward_color
            )

            # Success rate bar (mini)
            bar_x = x + width - 80
            bar_width = 60
            bar_height = 4
            bar_rect = pygame.Rect(bar_x, item_y + 8, bar_width, bar_height)
            pygame.draw.rect(screen, (50, 50, 50), bar_rect)

            if reward > -1:  # Valid reward range
                fill_width = int(bar_width * (reward + 1) / 2)
                if fill_width > 0:
                    fill_rect = pygame.Rect(bar_x, item_y + 8, fill_width, bar_height)
                    pygame.draw.rect(screen, reward_color, fill_rect)

    def _draw_context_panel(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        width: int,
        height: int,
        bandit_agent,
    ):
        """Draw context awareness and exploration stats."""
        content_y = self._draw_panel_background(
            screen, x, y, width, height, "CONTEXT & EXPLORATION"
        )

        try:
            summary = bandit_agent.get_performance_summary()

            stats_y = content_y + 10
            line_height = 18

            # Context utilization
            context_util = summary.get("context_utilization", 0)
            self.font_small.render_to(
                screen,
                (x + 15, stats_y),
                f"Context Window: {context_util}/5",
                self.text_color,
            )
            stats_y += line_height

            # Strategies explored
            strategies_tried = summary.get("strategies_tried", 0)
            total_selections = summary.get("total_selections", 0)
            self.font_small.render_to(
                screen,
                (x + 15, stats_y),
                f"Strategies Tried: {strategies_tried}",
                self.text_color,
            )
            stats_y += line_height

            self.font_small.render_to(
                screen,
                (x + 15, stats_y),
                f"Total Selections: {total_selections}",
                self.text_color,
            )
            stats_y += line_height * 2

            # Exploration vs Exploitation indicator
            if total_selections > 0:
                exploration_rate = min(
                    1.0, strategies_tried / max(1, total_selections * 0.1)
                )

                # Exploration bar
                bar_width = width - 40
                bar_height = 10
                bar_rect = pygame.Rect(x + 15, stats_y, bar_width, bar_height)
                pygame.draw.rect(screen, (40, 40, 40), bar_rect, border_radius=5)

                fill_width = int(bar_width * exploration_rate)
                if fill_width > 0:
                    fill_color = (
                        self.good_color
                        if 0.3 < exploration_rate < 0.7
                        else self.warning_color
                    )
                    fill_rect = pygame.Rect(x + 15, stats_y, fill_width, bar_height)
                    pygame.draw.rect(screen, fill_color, fill_rect, border_radius=5)

                self.font_small.render_to(
                    screen,
                    (x + 15, stats_y - 15),
                    "Exploration Balance",
                    self.text_color,
                )

                # Balance text
                balance_text = (
                    "Good"
                    if 0.3 < exploration_rate < 0.7
                    else "Low" if exploration_rate < 0.3 else "High"
                )
                self.font_small.render_to(
                    screen,
                    (x + 15, stats_y + 15),
                    f"Status: {balance_text}",
                    self.text_color,
                )

        except Exception as e:
            self.font_small.render_to(
                screen,
                (x + 15, content_y + 20),
                f"Context Error: {str(e)[:25]}...",
                self.bad_color,
            )


class BanditVisualizationDashboard:
    """Main dashboard - drop-in replacement for the complex version."""

    def __init__(self, screen_width: int, screen_height: int):
        self.dashboard = RealTimeBanditDashboard(screen_width, screen_height)

    def update(self, bandit_agent, latest_strategy=None, latest_reward=None):
        """Update dashboard with latest bandit data."""
        self.dashboard.update(bandit_agent, latest_strategy, latest_reward)

    def draw(self, screen: pygame.Surface, bandit_agent):
        """Draw the dashboard."""
        self.dashboard.draw(screen, bandit_agent)


# Backwards compatibility for any remaining component references
ComponentPerformanceChart = None
ConvergenceRadarChart = None
StrategyHeatmap = None
RestartIndicator = None
