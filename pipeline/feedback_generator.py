"""
SurgIQ — Feedback Generator
============================
Calls the Groq LLM (Llama3-70B) to generate real-time surgical coaching
feedback based on the current instrument activity and tracking state.

Rate-limited to at most one LLM call every FEEDBACK_INTERVAL_SECONDS
to avoid hammering the API during live video.

Usage:
    from pipeline.feedback_generator import FeedbackGenerator

    gen = FeedbackGenerator()
    feedback = gen.generate(
        classifier_label="hook_only",
        tracks=[{"track_id": 1, "class_name": "Hook", ...}],
        frame_idx=120,
    )
    print(feedback)   # "Good hook placement. Consider ..."
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    from groq import Groq
except ImportError:
    raise ImportError("groq not installed. Run: pip install groq")


# Coaching context per instrument activity label
COACHING_CONTEXT = {
    "no_instrument": (
        "No surgical instruments are currently visible. "
        "The surgeon may be repositioning or the view is obstructed."
    ),
    "grasper_only": (
        "Only the Grasper instrument is active. "
        "The surgeon is likely retracting or holding tissue."
    ),
    "hook_only": (
        "Only the Hook (L-hook electrocautery) is active. "
        "The surgeon is performing dissection or coagulation."
    ),
    "both_instruments": (
        "Both the Grasper and Hook are simultaneously active. "
        "The surgeon is performing bimanual dissection."
    ),
}


class FeedbackGenerator:
    """
    Groq LLM-based surgical coaching feedback generator.

    Generates short, actionable feedback for the surgeon trainee
    based on current instrument activity.  Rate-limited to avoid
    excessive API calls during live video.
    """

    def __init__(
        self,
        api_key: str            = cfg.GROQ_API_KEY,
        model: str              = cfg.GROQ_MODEL,
        interval_s: float       = cfg.FEEDBACK_INTERVAL_SECONDS,
        max_tokens: int         = cfg.GROQ_MAX_TOKENS,
    ):
        if not api_key:
            print("[FeedbackGenerator] WARNING: GROQ_API_KEY not set. "
                  "Feedback will use placeholder text.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

        self.model       = model
        self.interval_s  = interval_s
        self.max_tokens  = max_tokens
        self._last_call  = 0.0
        self._last_label = None

        print(f"[FeedbackGenerator] model={model}  interval={interval_s}s")

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        classifier_label: str,
        tracks: list[dict],
        frame_idx: int,
    ) -> str | None:
        """
        Generate coaching feedback for the current frame state.

        Returns None if rate-limited or if the label hasn't changed.
        Returns a feedback string otherwise.

        Parameters
        ----------
        classifier_label : str
            Current instrument activity class (from TechniqueClassifier).
        tracks : list[dict]
            Active tracks from InstrumentTracker.
        frame_idx : int
            Current frame index (used for logging).
        """
        now = time.time()

        # Rate limit: don't call more than once per interval
        if now - self._last_call < self.interval_s:
            return None

        # Skip if label unchanged (no new coaching needed)
        if classifier_label == self._last_label:
            return None

        self._last_call  = now
        self._last_label = classifier_label

        # Build context string
        context    = COACHING_CONTEXT.get(classifier_label, "Unknown instrument state.")
        track_info = self._format_tracks(tracks)

        prompt = (
            f"You are SurgIQ, an AI surgical training coach for laparoscopic cholecystectomy.\n\n"
            f"Current situation: {context}\n"
            f"Active tracks: {track_info}\n\n"
            f"Give one short, specific coaching tip (2-3 sentences max) for the surgical trainee. "
            f"Focus on technique, safety, or efficiency. Be direct and practical."
        )

        if self.client is None:
            # No API key — return placeholder
            return f"[No API key] Instrument state: {classifier_label}"

        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = self.max_tokens,
                temperature = 0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[FeedbackGenerator] LLM call failed: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_tracks(self, tracks: list[dict]) -> str:
        if not tracks:
            return "none"
        parts = [f"{t['class_name']} (ID={t['track_id']}, conf={t['confidence']:.2f})"
                 for t in tracks]
        return ", ".join(parts)
