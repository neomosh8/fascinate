"""Real-time visualization for hierarchical bandit learning."""

import pygame
import pygame.freetype
import numpy as np
import math
from collections import deque
from typing import Dict

from rl.hierarchical_bandit import HierarchicalBanditAgent


class ComponentPerformanceChart:
    """Real-time chart showing component performance evolution."""

    def __init__(self, x: int, y: int, width: int, height: int, component_name: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.component_name = component_name

        self.history_length = 50
        self.arm_histories: Dict[str, deque] = {}
        self.confidence_histories: Dict[str, deque] = {}

        self.colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
            (255, 150, 100),
            (150, 255, 100),
            (100, 150, 255),
        ]

        self.font = pygame.freetype.Font(None, 12)

    def update(self, bandit_data: Dict):
        """Update chart with new bandit data."""
        usage_stats = bandit_data.get("usage_stats", {})
        confidence_intervals = bandit_data.get("confidence_intervals", {})

        for arm, stats in usage_stats.items():
            if arm not in self.arm_histories:
                self.arm_histories[arm] = deque(maxlen=self.history_length)
                self.confidence_histories[arm] = deque(maxlen=self.history_length)

            self.arm_histories[arm].append(stats["average_reward"])

            if arm in confidence_intervals:
                low, high = confidence_intervals[arm]
                self.confidence_histories[arm].append(high - low)

    def draw(self, screen: pygame.Surface):
        self.surface.fill((20, 25, 30, 200))
        title_color = (200, 255, 200)
        self.font.render_to(self.surface, (5, 5), self.component_name.upper(), title_color)

        self._draw_grid()
        self._draw_performance_lines()
        self._draw_current_status()

        screen.blit(self.surface, self.rect)

    def _draw_grid(self):
        grid_color = (40, 50, 60)
        for i in range(5):
            y = 30 + i * (self.rect.height - 60) // 4
            pygame.draw.line(self.surface, grid_color, (10, y), (self.rect.width - 10, y), 1)
        for i in range(6):
            x = 10 + i * (self.rect.width - 20) // 5
            pygame.draw.line(self.surface, grid_color, (x, 30), (x, self.rect.height - 30), 1)

    def _draw_performance_lines(self):
        if not self.arm_histories:
            return

        chart_area = pygame.Rect(10, 30, self.rect.width - 20, self.rect.height - 60)
        for i, (arm, history) in enumerate(self.arm_histories.items()):
            if len(history) < 2:
                continue
            color = self.colors[i % len(self.colors)]
            points = []
            for j, value in enumerate(history):
                x = chart_area.left + (j / max(1, len(history) - 1)) * chart_area.width
                y = chart_area.bottom - (value * chart_area.height)
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.surface, color, False, points, 2)
                label_x = points[-1][0] + 5
                label_y = points[-1][1] - 10
                if label_x < self.rect.width - 50:
                    self.font.render_to(self.surface, (label_x, label_y), arm[:4], color)

    def _draw_current_status(self):
        if not self.arm_histories:
            return
        current_scores = {arm: hist[-1] for arm, hist in self.arm_histories.items() if hist}
        if not current_scores:
            return
        best_arm = max(current_scores, key=current_scores.get)
        worst_arm = min(current_scores, key=current_scores.get)
        y_pos = self.rect.height - 25
        self.font.render_to(self.surface, (10, y_pos), f"↑ {best_arm}", (100, 255, 100))
        self.font.render_to(self.surface, (self.rect.width - 80, y_pos), f"↓ {worst_arm}", (255, 100, 100))


class ConvergenceRadarChart:
    """Radar chart showing overall convergence state."""

    def __init__(self, x: int, y: int, radius: int):
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.components = ["tone", "topic", "emotion", "hook"]
        self.font = pygame.freetype.Font(None, 14)

    def draw(self, screen: pygame.Surface, bandit_agent: HierarchicalBanditAgent):
        scores = self._calculate_convergence_scores(bandit_agent)
        self._draw_radar_grid(screen)
        self._draw_convergence_polygon(screen, scores)
        self._draw_labels(screen, scores)

    def _calculate_convergence_scores(self, bandit_agent: HierarchicalBanditAgent) -> Dict[str, float]:
        scores = {}
        for name, bandit in bandit_agent.bandits.items():
            intervals = bandit.get_confidence_intervals()
            if intervals:
                avg_width = np.mean([hi - lo for lo, hi in intervals.values()])
                scores[name] = max(0, 1 - avg_width)
            else:
                scores[name] = 0.0
        return scores

    def _draw_radar_grid(self, screen: pygame.Surface):
        for i in range(1, 6):
            radius = int((i / 5) * self.radius)
            pygame.draw.circle(screen, (50, 60, 70), (self.center_x, self.center_y), radius, 1)
        for i, _ in enumerate(self.components):
            angle = i * 2 * math.pi / len(self.components) - math.pi / 2
            end_x = self.center_x + self.radius * math.cos(angle)
            end_y = self.center_y + self.radius * math.sin(angle)
            pygame.draw.line(screen, (70, 80, 90), (self.center_x, self.center_y), (end_x, end_y), 1)

    def _draw_convergence_polygon(self, screen: pygame.Surface, scores: Dict[str, float]):
        points = []
        for i, comp in enumerate(self.components):
            score = scores.get(comp, 0)
            angle = i * 2 * math.pi / len(self.components) - math.pi / 2
            dist = score * self.radius
            x = self.center_x + dist * math.cos(angle)
            y = self.center_y + dist * math.sin(angle)
            points.append((x, y))
        if len(points) > 2:
            pygame.draw.polygon(screen, (100, 150, 255, 100), points)
            pygame.draw.polygon(screen, (150, 200, 255), points, 2)

    def _draw_labels(self, screen: pygame.Surface, scores: Dict[str, float]):
        for i, comp in enumerate(self.components):
            score = scores.get(comp, 0)
            angle = i * 2 * math.pi / len(self.components) - math.pi / 2
            dist = self.radius + 20
            x = self.center_x + dist * math.cos(angle)
            y = self.center_y + dist * math.sin(angle)
            if score > 0.7:
                color = (100, 255, 100)
            elif score > 0.4:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            text = f"{comp}\n{score:.2f}"
            self.font.render_to(screen, (x - 25, y - 10), text, color)


class StrategyHeatmap:
    """Heatmap showing strategy combination performance."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.font = pygame.freetype.Font(None, 10)
        self.strategy_performance: Dict[str, list] = {}
        self.max_combinations = 20

    def update(self, strategy, reward: float):
        key = f"{strategy.tone[:3]}/{strategy.topic[:3]}/{strategy.emotion[:3]}"
        self.strategy_performance.setdefault(key, []).append(reward)
        if len(self.strategy_performance[key]) > 10:
            self.strategy_performance[key] = self.strategy_performance[key][-10:]

    def draw(self, screen: pygame.Surface):
        self.surface.fill((15, 20, 25, 200))
        self.font.render_to(self.surface, (5, 5), "STRATEGY COMBINATIONS", (200, 255, 200))
        if not self.strategy_performance:
            screen.blit(self.surface, self.rect)
            return
        sorted_strats = sorted(self.strategy_performance.items(), key=lambda x: np.mean(x[1]), reverse=True)
        y_offset = 25
        cell_h = 15
        for i, (key, rewards) in enumerate(sorted_strats[: self.max_combinations]):
            if y_offset + cell_h > self.rect.height - 10:
                break
            avg_reward = float(np.mean(rewards))
            usage = len(rewards)
            if avg_reward > 0.6:
                color = (100, 255, 100)
            elif avg_reward > 0.3:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            bar_w = int((avg_reward + 1) / 2 * (self.rect.width - 100))
            pygame.draw.rect(self.surface, color, (50, y_offset, bar_w, cell_h - 2))
            self.font.render_to(self.surface, (5, y_offset + 2), f"{key} ({usage})", (255, 255, 255))
            self.font.render_to(self.surface, (self.rect.width - 40, y_offset + 2), f"{avg_reward:.2f}", color)
            y_offset += cell_h
        screen.blit(self.surface, self.rect)


class RestartIndicator:
    """Visual indicator for adaptive restarts."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.font = pygame.freetype.Font(None, 12)
        self.restart_history = deque(maxlen=10)
        self.flash_timer = 0

    def add_restart(self, restart_type: str, step: int):
        self.restart_history.append((restart_type, step))
        self.flash_timer = 30

    def update(self):
        if self.flash_timer > 0:
            self.flash_timer -= 1

    def draw(self, screen: pygame.Surface):
        self.surface.fill((30, 30, 40, 200))
        title_color = (255, 200, 100) if self.flash_timer > 0 else (200, 200, 200)
        self.font.render_to(self.surface, (5, 5), "RESTARTS", title_color)
        if self.flash_timer > 0:
            alpha = int(255 * (self.flash_timer / 30))
            flash = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            flash.fill((255, 255, 0, alpha))
            self.surface.blit(flash, (0, 0))
        y = 25
        for rtype, step in list(self.restart_history)[-5:]:
            self.font.render_to(self.surface, (5, y), f"Step {step}: {rtype}", (255, 200, 100))
            y += 15
        screen.blit(self.surface, self.rect)


class BanditVisualizationDashboard:
    """Main dashboard combining all visualizations."""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        chart_w = 200
        chart_h = 120
        margin = 10
        self.tone_chart = ComponentPerformanceChart(margin, margin, chart_w, chart_h, "tone")
        self.topic_chart = ComponentPerformanceChart(margin + chart_w + margin, margin, chart_w, chart_h, "topic")
        self.emotion_chart = ComponentPerformanceChart(margin, margin + chart_h + margin, chart_w, chart_h, "emotion")
        self.hook_chart = ComponentPerformanceChart(margin + chart_w + margin, margin + chart_h + margin, chart_w, chart_h, "hook")
        radar_x = screen_width - 150
        radar_y = 150
        self.convergence_radar = ConvergenceRadarChart(radar_x, radar_y, 80)
        heatmap_y = margin + 2 * (chart_h + margin) + margin
        self.strategy_heatmap = StrategyHeatmap(margin, heatmap_y, chart_w * 2 + margin, 200)
        self.restart_indicator = RestartIndicator(screen_width - 200, margin, 180, 100)
        self.font = pygame.freetype.Font(None, 16)
        self._last_known_restart_step = 0

    def update(self, bandit_agent: HierarchicalBanditAgent, latest_strategy=None, latest_reward=None):
        summary = bandit_agent.get_performance_summary()
        components = summary.get("components", {})
        self.tone_chart.update(components.get("tone", {}))
        self.topic_chart.update(components.get("topic", {}))
        self.emotion_chart.update(components.get("emotion", {}))
        self.hook_chart.update(components.get("hook", {}))
        if latest_strategy is not None and latest_reward is not None:
            self.strategy_heatmap.update(latest_strategy, latest_reward)
        self.restart_indicator.update()
        last_restart = summary.get("restart_stats", {}).get("last_restart_step", 0)
        if last_restart > self._last_known_restart_step:
            self.restart_indicator.add_restart("adaptive", last_restart)
        self._last_known_restart_step = last_restart

    def draw(self, screen: pygame.Surface, bandit_agent: HierarchicalBanditAgent):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 50))
        screen.blit(overlay, (0, 0))
        self.font.render_to(screen, (10, self.screen_height - 30), "HIERARCHICAL BANDIT LEARNING DASHBOARD", (200, 255, 200))
        self.tone_chart.draw(screen)
        self.topic_chart.draw(screen)
        self.emotion_chart.draw(screen)
        self.hook_chart.draw(screen)
        self.convergence_radar.draw(screen, bandit_agent)
        self.strategy_heatmap.draw(screen)
        self.restart_indicator.draw(screen)
        status_y = self.screen_height - 60
        self.font.render_to(screen, (10, status_y), "Status: Learning component preferences...", (255, 255, 100))

