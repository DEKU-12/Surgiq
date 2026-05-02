"""
SurgIQ — Redis Stream Consumer
================================
Reads pipeline results from a Redis Stream for display in the
Streamlit dashboard.

Supports both:
  - Blocking reads (for real-time dashboard updates)
  - Batch reads (for replaying a recorded session)

Usage:
    from pipeline.redis_consumer import RedisConsumer

    consumer = RedisConsumer(session_id="demo")

    # Real-time: blocks until a new message arrives (up to timeout_ms)
    for message in consumer.listen(timeout_ms=1000):
        print(message["classifier_label"], message["feedback"])

    # Batch: read all messages since the beginning
    messages = consumer.read_all()
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    import redis
except ImportError:
    raise ImportError("redis not installed. Run: pip install redis")


class RedisConsumer:
    """
    Reads per-frame pipeline results from a Redis Stream.

    Messages are decoded from the raw Redis format into clean dicts.
    """

    def __init__(self, session_id: str = "default"):
        self.stream_key  = f"{cfg.REDIS_STREAM_KEY_PREFIX}:{session_id}"
        self._last_id    = "$"   # start from the latest message by default
        self._client     = None

        try:
            self._client = redis.Redis(
                host             = cfg.REDIS_HOST,
                port             = cfg.REDIS_PORT,
                decode_responses = True,
            )
            self._client.ping()
            print(f"[RedisConsumer] Connected → {self.stream_key}")
        except Exception as e:
            print(f"[RedisConsumer] WARNING: Redis unavailable ({e}). "
                  "Returning empty messages.")

    # ── Listen (real-time) ────────────────────────────────────────────────────

    def listen(self, timeout_ms: int = 500):
        """
        Generator that yields new messages as they arrive.

        Parameters
        ----------
        timeout_ms : int
            How long to block waiting for new messages.
            Set to 0 to block indefinitely.

        Yields
        ------
        dict  Decoded message (see _decode).
        """
        if self._client is None:
            return

        while True:
            try:
                results = self._client.xread(
                    {self.stream_key: self._last_id},
                    count = 10,
                    block = timeout_ms,
                )
                if not results:
                    continue

                for _, messages in results:
                    for msg_id, data in messages:
                        self._last_id = msg_id
                        yield self._decode(data)

            except Exception as e:
                print(f"[RedisConsumer] Read error: {e}")
                break

    # ── Read all (batch / replay) ─────────────────────────────────────────────

    def read_all(self, count: int = 1000) -> list[dict]:
        """
        Read up to `count` messages from the beginning of the stream.
        Useful for replaying a session in the dashboard.
        """
        if self._client is None:
            return []

        try:
            results = self._client.xrange(self.stream_key, count=count)
            return [self._decode(data) for _, data in results]
        except Exception as e:
            print(f"[RedisConsumer] read_all error: {e}")
            return []

    # ── Latest ────────────────────────────────────────────────────────────────

    def latest(self) -> dict | None:
        """Return the most recent message, or None if stream is empty."""
        if self._client is None:
            return None

        try:
            results = self._client.xrevrange(self.stream_key, count=1)
            if not results:
                return None
            _, data = results[0]
            return self._decode(data)
        except Exception as e:
            print(f"[RedisConsumer] latest error: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _decode(self, data: dict) -> dict:
        """Convert raw Redis strings to typed Python values."""
        return {
            "frame_idx"        : int(data.get("frame_idx", 0)),
            "timestamp"        : float(data.get("timestamp", 0.0)),
            "classifier_label" : data.get("classifier_label", "unknown"),
            "classifier_conf"  : float(data.get("classifier_conf", 0.0)),
            "num_tracks"       : int(data.get("num_tracks", 0)),
            "tracks"           : json.loads(data.get("tracks_json", "[]")),
            "feedback"         : data.get("feedback", ""),
        }

    def reset(self) -> None:
        """Reset the consumer to read from the beginning of the stream."""
        self._last_id = "0"

    def close(self) -> None:
        if self._client:
            self._client.close()
