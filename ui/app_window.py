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
            'update_transcript': self._queue_transcript_update
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

    def _process_updates(self):
        """Process queued updates in main thread."""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()

                if update_type == 'engagement':
                    self._update_engagement_plot(data)
                elif update_type == 'transcript':
                    self._update_transcript(data)

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

    def _on_button_press(self, event):
        """Handle button press - start recording."""
        if not self.is_recording:
            self.is_recording = True
            self.speak_button.config(bg='red', text='Recording...')
            self.status_label.config(text="Recording...")

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

            # Process in background
            asyncio.create_task(self._process_turn_async(audio_data))

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
        self.orchestrator.stop()
        self.root.quit()

    def run(self):
        """Run the UI."""
        self.root.mainloop()