"""
Non-blocking Text-to-Speech (TTS) Engine for Audio Feedback.
Uses macOS native `say` or pyttsx3 in a background thread to prevent UI stutter.
"""

import threading
import queue
import platform
import subprocess


class TextToSpeechEngine:
    """
    Asynchronous audio synthesizer for real-time sign language feedback.
    """

    def __init__(self, rate: int = 160, enabled: bool = True):
        self.enabled = enabled
        self.rate = rate
        self.queue = queue.Queue()
        self.is_macos = platform.system() == "Darwin"
        self.stopped = False

        if self.enabled:
            self.worker = threading.Thread(target=self._speech_loop, daemon=True)
            self.worker.start()

    def speak(self, text: str):
        """Pushes phrase to background speech queue."""
        if not self.enabled or not text:
            return
        self.queue.put(text)

    def _speech_loop(self):
        while not self.stopped:
            try:
                text = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if self.is_macos:
                    # Native macOS say command is ultra-fast and never locks Python GIL
                    subprocess.run(["say", "-r", str(self.rate), text], check=False)
                else:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.rate)
                    engine.say(text)
                    engine.runAndWait()
            except Exception:
                pass
            finally:
                self.queue.task_done()

    def stop(self):
        self.stopped = True
