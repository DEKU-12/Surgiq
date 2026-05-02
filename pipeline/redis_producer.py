"""
SurgIQ — Redis Stream Producer
================================
Publishes per-frame pipeline results to a Redis Stream so the
Streamlit dashboard (consumer) can display them in real time.

Stream key format: surgiq:session:<session_id>

Each message contains:
    frame_idx        : int
    timestamp        : float  (unix time)
    classifier_label : str
    classifier_conf  : float
    num_tracks       : int
    tracks_json      : str    (JSON-encoded list of track dicts)
    feedback         : str    (LLM coaching text, or "")

Usage:
    from pipeline.redis_producer import RedisProducer

    producer = RedisProducer(session_id="demo")
    producer.publish(frame_idx=1, classifier_result={...}, tracks=[...], feedback="...")
    producer.close()
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    import redis
except ImportError:
    raise ImportError("redis not installed. Run: pip install redis")


class RedisProducer:
    """
    Publishes pipeline results to a Redis Stream.

    If Redis is unavailable, falls back to silent no-op so the
    pipeline can still run without a Redis server (e.g. during testing).
    """

    def __init__(self, session_id: str = "default"):
        self.stream_key = f"{cfg.REDIS_STREAM_KEY_PREFIX}:{session_id}"
        self._client    = None

        try:
            self._client = redis.Redis(
                host     = cfg.REDIS_HOST,
                port     = cfg.REDIS_PORT,
                decode_responses = True,
            )
            self._client.ping()
            print(f"[RedisProducer] Connected → {self.stream_key}")
        except Exception as e:
            print(f"[RedisProducer] WARNING: Redis unavailable ({e}). "
                  "Running in no-op mode.")
            self._client = None

    # ── Publish ───────────────────────────────────────────────────────────────

    def publish(
        self,
        frame_idx: int,
        classifier_result: dict,
        tracks: list[dict],
        feedback: str | None = None,
    ) -> str | None:
        """
        Publish one frame's results to the Redis stream.

        Parameters
        ----------
        frame_idx          : int    Current frame index.
        classifier_result  : dict   Output of TechniqueClassifier.classify().
        tracks             : list   Output of InstrumentTracker.update().
        feedback           : str    LLM coaching text (or None).

        Returns
        -------
        str | None  Redis message ID, or None if Redis is unavailable.
        """
        if self._client is None:
            return None

        # Simplify tracks for serialisation (remove heavy fields if any)
        tracks_clean = [
            {
                "track_id"  : t["track_id"],
                "class_name": t["class_name"],
                "confidence": round(t["confidence"], 3),
                "bbox_xyxy" : [round(v, 1) for v in t["bbox_xyxy"]],
            }
            for t in tracks
        ]

        message = {
            "frame_idx"        : frame_idx,
            "timestamp"        : round(time.time(), 3),
            "classifier_label" : classifier_result.get("label", "unknown"),
            "classifier_conf"  : round(classifier_result.get("confidence", 0.0), 3),
            "num_tracks"       : len(tracks),
            "tracks_json"      : json.dumps(tracks_clean),
            "feedback"         : feedback or "",
        }

        try:
            msg_id = self._client.xadd(
                self.stream_key,
                message,
                maxlen = cfg.REDIS_STREAM_MAXLEN,
                approximate = True,
            )
            return msg_id
        except Exception as e:
            print(f"[RedisProducer] Publish error: {e}")
            return None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._client:
            self._client.close()
            print("[RedisProducer] Connection closed.")

    def clear_stream(self) -> None:
        """Delete all messages in the stream (useful between sessions)."""
        if self._client:
            self._client.delete(self.stream_key)
            print(f"[RedisProducer] Stream cleared: {self.stream_key}")
