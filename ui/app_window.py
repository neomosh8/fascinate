"""Main application window using Tkinter."""

import tkinter as tk
from tkinter import scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import numpy as np
import asyncio
import threading
import queue

from config import WINDOW_WIDTH, WINDOW_HEIGHT, GRAPH_UPDATE_INTERVAL


class ConversationUI:
    """Main UI window for the EEG conversation system."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.root = tk.Tk()
        self.root.title("EEG-Driven Conversation with RL")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # Data for plotting
        self.engagement_history = deque(maxlen=300)  # 30 seconds at 10Hz
        self.time_history = deque(maxlen=300)
        self.current_time = 0

        # Queue for thread-safe updates
        self.update_queue = queue.Queue()

        # Setup UI components
        self._setup_ui()

        # Bind orchestrator callbacks
        self.orchestrator.ui_callbacks = {
            'update_engagement': self._queue_engagement_update,
            'update_transcript': self._queue_transcript_update,
            'update_countdown': self._queue_countdown_update
        }

        # Start update loop
        self.root.after(GRAPH_UPDATE_INTERVAL, self._process_updates)

        # Track recording state
        self.is_recording = False

    def _setup_ui(self):
        """Create UI components."""
        # Top frame for engagement graph
        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(-30, 0)
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Engagement')
        self.ax.set_title('Real-time Engagement Score')
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Middle frame for transcript
        transcript_frame = ttk.Frame(self.root)
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Label(transcript_frame, text="Conversation:").pack(anchor=tk.W)

        self.transcript = scrolledtext.ScrolledText(
            transcript_frame,
            wrap=tk.WORD,
            height=10,
            font=('Arial', 10)
        )
        self.transcript.pack(fill=tk.BOTH, expand=True)

        # Bottom frame for controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        self.summary_button = ttk.Button(
            control_frame,
            text="Show Summary",
            command=self._show_session_summary
        )
        self.summary_button.pack(side=tk.RIGHT, padx=10)
        # Hold-to-speak button
        self.speak_button = tk.Button(
            control_frame,
            text="Hold to Speak",
            font=('Arial', 14),
            bg='lightblue',
            width=20,
            height=3
        )
        self.speak_button.pack(side=tk.LEFT, padx=20)

        # Bind button events
        self.speak_button.bind('<ButtonPress-1>', self._on_button_press)
        self.speak_button.bind('<ButtonRelease-1>', self._on_button_release)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Countdown label
        self.countdown_label = ttk.Label(control_frame, text="", foreground="orange")
        self.countdown_label.pack(side=tk.LEFT, padx=10)

        # Stop button
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Session",
            command=self._stop_session
        )
        self.stop_button.pack(side=tk.RIGHT, padx=20)

    def _queue_engagement_update(self, engagement: float):
        """Queue engagement update (thread-safe)."""
        self.update_queue.put(('engagement', engagement))

    def _queue_transcript_update(self, text: str):
        """Queue transcript update (thread-safe)."""
        self.update_queue.put(('transcript', text))

    def _queue_countdown_update(self, seconds_left: int):
        """Queue countdown update (thread-safe)."""
        self.update_queue.put(('countdown', seconds_left))

    def _process_updates(self):
        """Process queued updates in main thread."""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()

                if update_type == 'engagement':
                    self._update_engagement_plot(data)
                elif update_type == 'transcript':
                    self._update_transcript(data)
                elif update_type == 'countdown':
                    self._update_countdown(data)

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(GRAPH_UPDATE_INTERVAL, self._process_updates)

    def _update_engagement_plot(self, engagement: float):
        """Update engagement plot."""
        self.current_time += 0.1  # 10Hz updates
        self.engagement_history.append(engagement)
        self.time_history.append(self.current_time)

        if len(self.time_history) > 1:
            # Update plot data
            times = np.array(self.time_history) - self.current_time
            values = np.array(self.engagement_history)

            self.line.set_data(times, values)

            # Redraw canvas
            self.ax.draw_artist(self.ax.patch)
            self.ax.draw_artist(self.line)
            self.canvas.draw_idle()

    def _update_transcript(self, text: str):
        """Update conversation transcript."""
        self.transcript.insert(tk.END, text + '\n\n')
        self.transcript.see(tk.END)

    def _update_countdown(self, seconds_left: int):
        """Update countdown display."""
        if seconds_left > 0:
            self.countdown_label.config(text=f"Auto-advance in {seconds_left}s")
        else:
            self.countdown_label.config(text="")

    def _on_button_press(self, event):
        """Handle button press - start recording."""
        if not self.is_recording:
            self.is_recording = True
            self.speak_button.config(bg='red', text='Recording...')
            self.status_label.config(text="Recording...")
            self.countdown_label.config(text="")

            # Cancel auto-advance timer
            self.orchestrator.cancel_auto_advance_timer()

            # Start recording
            self.orchestrator.stt.start_recording()

    def _on_button_release(self, event):
        """Handle button release - stop recording and process."""
        if self.is_recording:
            self.is_recording = False
            self.speak_button.config(bg='lightblue', text='Hold to Speak')
            self.status_label.config(text="Processing...")

            # Stop recording and process turn
            audio_data = self.orchestrator.stt.stop_recording()

            # Submit to the async event loop running in the background thread
            if hasattr(self.orchestrator, 'event_loop') and self.orchestrator.event_loop:
                future = asyncio.run_coroutine_threadsafe(
                    self._process_turn_async(audio_data),
                    self.orchestrator.event_loop
                )

                # Optional: Handle the result
                def handle_result():
                    try:
                        future.result(timeout=0.1)  # Non-blocking check
                        self.status_label.config(text="Ready")
                    except asyncio.TimeoutError:
                        # Still processing, check again later
                        self.root.after(100, handle_result)
                    except Exception as e:
                        print(f"Turn processing error: {e}")
                        self.status_label.config(text="Error - Ready")

                self.root.after(100, handle_result)
            else:
                print("No event loop available")
                self.status_label.config(text="Error - Ready")

    async def _process_turn_async(self, audio_data):
        """Process turn in async context."""
        try:
            await self.orchestrator.process_turn(audio_data)
            self.status_label.config(text="Ready")
        except Exception as e:
            print(f"Turn processing error: {e}")
            self.status_label.config(text="Error - Ready")

    def _stop_session(self):
        """Stop the conversation session."""
        if self.orchestrator.turn_count > 0:
            # Show summary before closing
            self._show_session_summary()

        self.orchestrator.stop()
        self.root.after(2000, self.root.quit)  # Give time for summary window

    def run(self):
        """Run the UI."""
        self.root.mainloop()

    # Add this method to the ConversationUI class
    def _show_session_summary(self):
        """Show session summary in a new window."""
        summary = self.orchestrator.get_session_summary()

        # Create summary window
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Session Summary")
        summary_window.geometry("600x500")

        # Create scrolled text widget
        summary_text = scrolledtext.ScrolledText(
            summary_window,
            wrap=tk.WORD,
            font=('Courier', 10)
        )
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Format summary text
        summary_content = self._format_summary_text(summary)
        summary_text.insert(tk.END, summary_content)
        summary_text.config(state=tk.DISABLED)

        # Add close button
        close_button = ttk.Button(
            summary_window,
            text="Close",
            command=summary_window.destroy
        )
        close_button.pack(pady=10)

    def _format_summary_text(self, summary: dict) -> str:
        """Format summary dictionary as readable text."""
        text = "SESSION SUMMARY\n" + "=" * 50 + "\n\n"

        # Session info
        session_info = summary["session_info"]
        text += f"Total Turns: {session_info['total_turns']}\n"
        text += f"Session Duration: {session_info['session_duration']:.1f} seconds\n"
        text += f"Final Engagement: {session_info['final_engagement']:.3f}\n\n"

        # RL Performance
        rl_perf = summary["rl_performance"]
        if "error" not in rl_perf:
            text += f"TOTAL REWARD: {rl_perf['total_reward']:.2f}\n"
            text += f"AVERAGE REWARD: {rl_perf['average_reward']:.3f}\n\n"

            # Best strategy - Fixed to use dot notation
            best = rl_perf["best_strategy"]
            text += "🏆 WINNING STRATEGY:\n"
            text += f"   Tone: {best['strategy'].tone}\n"
            text += f"   Topic: {best['strategy'].topic}\n"
            text += f"   Emotion: {best['strategy'].emotion}\n"
            text += f"   Hook: {best['strategy'].hook}\n"
            text += f"   Average Reward: {best['average_reward']:.3f}\n"
            text += f"   Used {best['usage_count']} times\n\n"

            # Top strategies - Fixed to use dot notation
            text += "🏅 TOP 5 STRATEGIES:\n"
            for i, strategy_info in enumerate(rl_perf["top_strategies"][:5], 1):
                s = strategy_info["strategy"]
                text += f"   {i}. {s.tone}/{s.topic}/{s.emotion}/{s.hook}\n"
                text += f"      Avg Reward: {strategy_info['average_reward']:.3f} "
                text += f"(used {strategy_info['usage_count']} times)\n"

            text += "\n"

            # Learning progress
            learning = rl_perf["learning_progress"]
            text += "📈 LEARNING PROGRESS:\n"
            text += f"   Early Average Reward: {learning['early_average_reward']:.3f}\n"
            text += f"   Recent Average Reward: {learning['recent_average_reward']:.3f}\n"
            text += f"   Improvement: {learning['improvement']:.3f}\n\n"

            # Exploration stats
            exploration = rl_perf["exploration_stats"]
            text += "🔍 EXPLORATION:\n"
            text += f"   Strategies Tried: {exploration['strategies_tried']}/{exploration['total_strategies']}\n"
            text += f"   Final Epsilon: {exploration['final_epsilon']:.3f}\n"

        return text