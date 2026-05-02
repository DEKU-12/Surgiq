"""
SurgIQ — Real-Time Surgical Training Pipeline
==============================================
Orchestrates the full pipeline:
  VideoProcessor → InstrumentDetector → InstrumentTracker
      → TechniqueClassifier → FeedbackGenerator → RedisProducer

Usage:
    # Run on a video file
    python main.py --source path/to/surgery.mp4

    # Run on webcam
    python main.py --source 0

    # Run without Redis (standalone mode)
    python main.py --source video.mp4 --no-redis

    # Show annotated video window
    python main.py --source video.mp4 --display
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from pipeline.video_processor      import VideoProcessor
from pipeline.instrument_detector  import InstrumentDetector
from pipeline.instrument_tracker   import InstrumentTracker
from pipeline.technique_classifier import TechniqueClassifier
from pipeline.feedback_generator   import FeedbackGenerator
from pipeline.redis_producer       import RedisProducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SurgIQ Real-Time Pipeline")
    parser.add_argument(
        "--source", default="0",
        help="Video file path or webcam index (default: 0)"
    )
    parser.add_argument(
        "--session-id", default="demo",
        help="Redis stream session ID (default: demo)"
    )
    parser.add_argument(
        "--no-redis", action="store_true",
        help="Disable Redis publishing (standalone mode)"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show annotated video in a window (requires display)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = run until end)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse source — convert to int if webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("  SurgIQ — Real-Time Surgical Training Pipeline")
    print(f"  Source     : {source}")
    print(f"  Session    : {args.session_id}")
    print(f"  Redis      : {'disabled' if args.no_redis else 'enabled'}")
    print(f"  Display    : {args.display}")
    print("=" * 60)

    # ── Initialise components ─────────────────────────────────────────────────
    detector   = InstrumentDetector()
    tracker    = InstrumentTracker()
    classifier = TechniqueClassifier()
    feedback   = FeedbackGenerator()
    producer   = None if args.no_redis else RedisProducer(session_id=args.session_id)

    # ── Stats ─────────────────────────────────────────────────────────────────
    frame_times  = []
    label_counts = {label: 0 for label in ["no_instrument", "grasper_only",
                                             "hook_only", "both_instruments"]}

    print("\nPipeline running. Press Ctrl+C to stop.\n")

    try:
        with VideoProcessor(source, target_fps=cfg.PIPELINE_FPS) as vp:
            for frame_idx, frame in vp:
                t0 = time.perf_counter()

                # 1. Detect instruments
                detections = detector.detect(frame)

                # 2. Track instruments
                tracks = tracker.update(detections, frame)

                # 3. Classify instrument activity
                clf_result = classifier.classify(frame)
                label      = clf_result["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

                # 4. Generate LLM feedback (rate-limited)
                tip = feedback.generate(
                    classifier_label = label,
                    tracks           = tracks,
                    frame_idx        = frame_idx,
                )

                # 5. Publish to Redis
                if producer:
                    producer.publish(
                        frame_idx         = frame_idx,
                        classifier_result = clf_result,
                        tracks            = tracks,
                        feedback          = tip,
                    )

                # 6. Console log
                elapsed_ms = (time.perf_counter() - t0) * 1000
                frame_times.append(elapsed_ms)

                det_str   = ", ".join(f"{d['class_name']}({d['confidence']:.2f})"
                                       for d in detections) or "none"
                track_str = f"{len(tracks)} tracks"
                tip_str   = f" | {tip[:60]}..." if tip else ""

                print(f"  Frame {frame_idx:>5} | {elapsed_ms:>5.1f}ms | "
                      f"{label:<20} | det: {det_str} | {track_str}{tip_str}")

                # 7. Display (optional)
                if args.display:
                    annotated = detector.annotate(frame, detections)
                    cv2.putText(annotated, f"Label: {label}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    cv2.imshow("SurgIQ", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # 8. Max frames limit
                if args.max_frames and frame_idx >= args.max_frames:
                    print(f"\n  Reached --max-frames={args.max_frames}. Stopping.")
                    break

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        if args.display:
            cv2.destroyAllWindows()
        if producer:
            producer.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    if frame_times:
        avg_ms = sum(frame_times) / len(frame_times)
        print(f"\n{'=' * 60}")
        print(f"  Processed : {len(frame_times)} frames")
        print(f"  Avg latency: {avg_ms:.1f}ms  ({1000/avg_ms:.1f} FPS effective)")
        print(f"\n  Instrument activity breakdown:")
        total = sum(label_counts.values())
        for lbl, cnt in label_counts.items():
            pct = cnt / total * 100 if total else 0
            print(f"    {lbl:<22}: {cnt:>5} frames ({pct:>5.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
