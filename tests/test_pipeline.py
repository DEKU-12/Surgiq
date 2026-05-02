"""
SurgIQ — Unit Tests
=====================
Tests for the core pipeline components.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # shorter tracebacks
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_frame():
    """512x512 random BGR frame."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Two sample detections — one Grasper, one Hook."""
    return [
        {
            "class_idx"  : 0,
            "class_name" : "Grasper",
            "confidence" : 0.91,
            "bbox_xyxy"  : [100.0, 50.0, 300.0, 400.0],
            "bbox_xywhn" : [0.39, 0.44, 0.39, 0.69],
        },
        {
            "class_idx"  : 1,
            "class_name" : "Hook",
            "confidence" : 0.85,
            "bbox_xyxy"  : [320.0, 100.0, 480.0, 380.0],
            "bbox_xywhn" : [0.78, 0.47, 0.31, 0.55],
        },
    ]


@pytest.fixture
def sample_tracks():
    """Two confirmed tracks."""
    return [
        {"track_id": 1, "class_idx": 0, "class_name": "Grasper",
         "confidence": 0.91, "bbox_xyxy": [100.0, 50.0, 300.0, 400.0]},
        {"track_id": 2, "class_idx": 1, "class_name": "Hook",
         "confidence": 0.85, "bbox_xyxy": [320.0, 100.0, 480.0, 380.0]},
    ]


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_yolo_class_names(self):
        import config as cfg
        assert cfg.YOLO_CLASS_NAMES == ["Grasper", "Hook"]

    def test_classifier_labels(self):
        from training.prepare_dataset import CLASSIFIER_LABELS
        assert "no_instrument"    in CLASSIFIER_LABELS
        assert "grasper_only"     in CLASSIFIER_LABELS
        assert "hook_only"        in CLASSIFIER_LABELS
        assert "both_instruments" in CLASSIFIER_LABELS

    def test_test_videos_held_out(self):
        from training.prepare_dataset import TEST_VIDEOS
        assert "video48" in TEST_VIDEOS
        assert "video52" in TEST_VIDEOS
        assert "video55" in TEST_VIDEOS

    def test_mask_pixel_values(self):
        from training.prepare_dataset import MASK_GRASPER, MASK_HOOK
        assert MASK_GRASPER == 13
        assert MASK_HOOK    == 12

    def test_device_returns_valid(self):
        import config as cfg
        import torch
        device = cfg.get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("mps", "cuda", "cpu")


# ── mask_to_bboxes Tests ──────────────────────────────────────────────────────

class TestMaskToBboxes:
    """Tests for the mask → YOLO bbox conversion."""

    def _make_mask(self, h=480, w=640, grasper_rect=None, hook_rect=None):
        """Create a synthetic grayscale mask with instrument pixels."""
        mask = np.zeros((h, w), dtype=np.uint8)
        if grasper_rect:
            y1, y2, x1, x2 = grasper_rect
            mask[y1:y2, x1:x2] = 13   # MASK_GRASPER
        if hook_rect:
            y1, y2, x1, x2 = hook_rect
            mask[y1:y2, x1:x2] = 12   # MASK_HOOK
        return mask

    def test_empty_mask_returns_empty(self, tmp_path):
        import cv2
        from training.prepare_dataset import mask_to_bboxes

        mask = np.zeros((480, 640), dtype=np.uint8)
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_to_bboxes(path)
        assert result == []

    def test_grasper_detected(self, tmp_path):
        import cv2
        from training.prepare_dataset import mask_to_bboxes

        mask = self._make_mask(grasper_rect=(100, 300, 200, 400))
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_to_bboxes(path, min_area=100)
        assert len(result) == 1
        assert result[0][0] == 0   # Grasper class_idx

    def test_hook_detected(self, tmp_path):
        import cv2
        from training.prepare_dataset import mask_to_bboxes

        mask = self._make_mask(hook_rect=(50, 200, 300, 500))
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_to_bboxes(path, min_area=100)
        assert len(result) == 1
        assert result[0][0] == 1   # Hook class_idx

    def test_both_instruments_detected(self, tmp_path):
        import cv2
        from training.prepare_dataset import mask_to_bboxes

        mask = self._make_mask(
            grasper_rect=(10, 200, 10, 200),
            hook_rect=(250, 400, 300, 500),
        )
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_to_bboxes(path, min_area=100)
        class_ids = {r[0] for r in result}
        assert 0 in class_ids   # Grasper
        assert 1 in class_ids   # Hook

    def test_bbox_normalised(self, tmp_path):
        """All bbox values must be in [0, 1]."""
        import cv2
        from training.prepare_dataset import mask_to_bboxes

        mask = self._make_mask(grasper_rect=(100, 300, 200, 400))
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_to_bboxes(path, min_area=100)
        for _, cx, cy, bw, bh in result:
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 <  bw <= 1.0
            assert 0.0 <  bh <= 1.0

    def test_min_area_filters_noise(self, tmp_path):
        """Tiny pixel blobs below min_area should be ignored."""
        import cv2
        from training.prepare_dataset import mask_to_bboxes

        # Only 4 pixels — below default min_area=150
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:102, 100:102] = 13
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_to_bboxes(path, min_area=150)
        assert result == []

    def test_one_bbox_per_class(self, tmp_path):
        """Each class should produce exactly ONE merged bbox."""
        import cv2
        from training.prepare_dataset import mask_to_bboxes

        # Two separate Grasper blobs — should merge into one bbox
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[50:100,  50:150]  = 13
        mask[200:300, 300:450] = 13
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_to_bboxes(path, min_area=100)
        grasper_boxes = [r for r in result if r[0] == 0]
        assert len(grasper_boxes) == 1   # merged into ONE box


# ── Classifier Label Tests ────────────────────────────────────────────────────

class TestGetClassifierLabel:
    def test_no_instrument(self):
        from training.prepare_dataset import get_classifier_label
        assert get_classifier_label([]) == "no_instrument"

    def test_grasper_only(self):
        from training.prepare_dataset import get_classifier_label
        anns = [(0, 0.5, 0.5, 0.3, 0.3)]   # class_idx=0 (Grasper)
        assert get_classifier_label(anns) == "grasper_only"

    def test_hook_only(self):
        from training.prepare_dataset import get_classifier_label
        anns = [(1, 0.5, 0.5, 0.3, 0.3)]   # class_idx=1 (Hook)
        assert get_classifier_label(anns) == "hook_only"

    def test_both_instruments(self):
        from training.prepare_dataset import get_classifier_label
        anns = [(0, 0.3, 0.5, 0.2, 0.3), (1, 0.7, 0.5, 0.2, 0.3)]
        assert get_classifier_label(anns) == "both_instruments"


# ── InstrumentDetector Tests ──────────────────────────────────────────────────

class TestInstrumentDetector:
    """Tests InstrumentDetector with a mocked YOLO model."""

    def _mock_detector(self):
        with patch("pipeline.instrument_detector.YOLO") as mock_yolo, \
             patch("pathlib.Path.exists", return_value=True):
            from pipeline.instrument_detector import InstrumentDetector
            detector = InstrumentDetector.__new__(InstrumentDetector)
            detector.conf_threshold = 0.25
            detector.iou_threshold  = 0.45
            detector.device         = "cpu"
            detector.model          = MagicMock()
            return detector

    def test_annotate_returns_same_shape(self, dummy_frame, sample_detections):
        detector = self._mock_detector()
        annotated = detector.annotate(dummy_frame, sample_detections)
        assert annotated.shape == dummy_frame.shape

    def test_annotate_does_not_mutate_original(self, dummy_frame, sample_detections):
        original = dummy_frame.copy()
        detector = self._mock_detector()
        detector.annotate(dummy_frame, sample_detections)
        assert np.array_equal(dummy_frame, original)

    def test_empty_detections_returns_unchanged(self, dummy_frame):
        detector = self._mock_detector()
        annotated = detector.annotate(dummy_frame, [])
        assert np.array_equal(annotated, dummy_frame)


# ── RedisProducer Tests ───────────────────────────────────────────────────────

class TestRedisProducer:
    """Tests RedisProducer with a mocked Redis client."""

    def test_publish_formats_message_correctly(self, sample_tracks):
        with patch("pipeline.redis_producer.redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.xadd.return_value = "1234-0"
            mock_redis.Redis.return_value = mock_client

            from pipeline.redis_producer import RedisProducer
            producer = RedisProducer(session_id="test")

            clf_result = {"label": "both_instruments", "confidence": 0.94}
            msg_id = producer.publish(
                frame_idx         = 42,
                classifier_result = clf_result,
                tracks            = sample_tracks,
                feedback          = "Keep steady tension on the grasper.",
            )

            assert msg_id == "1234-0"
            call_args = mock_client.xadd.call_args
            message   = call_args[0][1]

            assert message["frame_idx"] == 42
            assert message["classifier_label"] == "both_instruments"
            assert message["feedback"] == "Keep steady tension on the grasper."
            tracks_decoded = json.loads(message["tracks_json"])
            assert len(tracks_decoded) == 2


# ── RedisConsumer Tests ───────────────────────────────────────────────────────

class TestRedisConsumer:
    def test_decode_produces_correct_types(self):
        with patch("pipeline.redis_consumer.redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.Redis.return_value = mock_client

            from pipeline.redis_consumer import RedisConsumer
            consumer = RedisConsumer.__new__(RedisConsumer)
            consumer._client   = mock_client
            consumer.stream_key = "surgiq:session:test"
            consumer._last_id  = "$"

            raw = {
                "frame_idx"        : "10",
                "timestamp"        : "1700000000.5",
                "classifier_label" : "hook_only",
                "classifier_conf"  : "0.88",
                "num_tracks"       : "1",
                "tracks_json"      : '[{"track_id": 3}]',
                "feedback"         : "Good dissection plane.",
            }
            decoded = consumer._decode(raw)

            assert decoded["frame_idx"]        == 10
            assert decoded["timestamp"]        == 1700000000.5
            assert decoded["classifier_label"] == "hook_only"
            assert decoded["classifier_conf"]  == 0.88
            assert decoded["num_tracks"]       == 1
            assert decoded["tracks"][0]["track_id"] == 3
            assert decoded["feedback"]         == "Good dissection plane."


# ── FeedbackGenerator Tests ───────────────────────────────────────────────────

class TestFeedbackGenerator:
    def test_rate_limiting(self):
        """Should return None if called twice within interval."""
        with patch("pipeline.feedback_generator.Groq"):
            from pipeline.feedback_generator import FeedbackGenerator
            gen = FeedbackGenerator(api_key="fake-key", interval_s=60)
            gen._last_call  = 0
            gen._last_label = None

            # First call — should trigger LLM
            gen.client = MagicMock()
            gen.client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Good technique."))]
            )
            result1 = gen.generate("hook_only", [], frame_idx=1)

            # Second call — same label, should be None (rate-limited + same label)
            result2 = gen.generate("hook_only", [], frame_idx=2)
            assert result2 is None

    def test_label_change_triggers_new_feedback(self):
        """Label change should bypass label-same check."""
        with patch("pipeline.feedback_generator.Groq"):
            from pipeline.feedback_generator import FeedbackGenerator
            import time
            gen = FeedbackGenerator(api_key="fake-key", interval_s=0)
            gen.client = MagicMock()
            gen.client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Switch detected."))]
            )
            gen._last_call  = 0
            gen._last_label = "hook_only"

            # Different label — should fire
            result = gen.generate("both_instruments", [], frame_idx=10)
            assert result == "Switch detected."


# ── VideoProcessor Tests ──────────────────────────────────────────────────────

class TestVideoProcessor:
    def test_invalid_source_raises(self):
        from pipeline.video_processor import VideoProcessor
        with pytest.raises(RuntimeError):
            with VideoProcessor("/nonexistent/path.mp4") as vp:
                pass

    def test_frame_split_logic(self):
        from training.prepare_dataset import frame_split
        # First 80% → train
        assert frame_split(0,   100) == "train"
        assert frame_split(79,  100) == "train"
        # Last 20% → val
        assert frame_split(80,  100) == "val"
        assert frame_split(99,  100) == "val"
