"""
SurgIQ — Voice Feedback
========================
Converts LLM coaching text to speech so surgeons can hear
feedback without looking away from the surgical field.

Uses macOS built-in 'say' command (no API key needed) or
gTTS (Google Text-to-Speech) as a fallback.

Usage:
    from pipeline.voice_feedback import VoiceFeedback

    vf = VoiceFeedback()
    vf.speak("Maintain gentle traction on the grasper.")
"""

import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class VoiceFeedback:
    """
    Speaks coaching feedback aloud using text-to-speech.

    Runs in a background thread so it never blocks the pipeline.
    Only one utterance plays at a time — new feedback cancels old.
    """

    def __init__(self, rate: int = 180, voice: str = "Samantha"):
        """
        Parameters
        ----------
        rate  : int  Words per minute (default 180 — natural speech pace)
        voice : str  macOS voice name (default Samantha — clear, professional)
                     Other options: Alex, Victoria, Karen, Daniel
        """
        self.rate      = rate
        self.voice     = voice
        self._process  = None
        self._lock     = threading.Lock()

        # Test if 'say' command is available (macOS)
        result = subprocess.run(["which", "say"], capture_output=True)
        self._has_say = result.returncode == 0

        if self._has_say:
            print(f"[VoiceFeedback] Using macOS 'say' — voice={voice}  rate={rate}")
        else:
            print("[VoiceFeedback] macOS 'say' not found. Voice feedback disabled.")

    def speak(self, text: str) -> None:
        """
        Speak text aloud in a background thread.
        Cancels any currently playing speech first.

        Parameters
        ----------
        text : str  The coaching feedback to speak.
        """
        if not self._has_say or not text:
            return

        # Run in background thread — never blocks the pipeline
        thread = threading.Thread(
            target=self._speak_sync,
            args=(text,),
            daemon=True,
        )
        thread.start()

    def _speak_sync(self, text: str) -> None:
        """Internal: cancel current speech and start new."""
        with self._lock:
            # Kill previous speech if still playing
            if self._process and self._process.poll() is None:
                self._process.terminate()

            # Keep feedback short — take first 2 sentences only
            sentences = text.replace("  ", " ").split(". ")
            short_text = ". ".join(sentences[:2])
            if not short_text.endswith("."):
                short_text += "."

            self._process = subprocess.Popen(
                ["say", "-v", self.voice, "-r", str(self.rate), short_text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._process.wait()

    def stop(self) -> None:
        """Stop any currently playing speech."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
