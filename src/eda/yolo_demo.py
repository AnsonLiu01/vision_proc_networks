"""
YOLO object detector over the POV walk video.

Uses the same video the EEG/CORnet analysis uses (AC3.mp4, see
src/eda/cornet_test.ipynb). Annotated output is written under runs/detect/.

Run it:
    uv run python src/eda/yolo_demo.py  # quick demo (first 300 frames)
    uv run python src/eda/yolo_demo.py --mode full  # whole video, every frame
    uv run python src/eda/yolo_demo.py --mode full --stride 5  # whole video, sampled
    uv run python src/eda/yolo_demo.py --mode chunked --chunks 4  # split, detect, stitch
    uv run python src/eda/yolo_demo.py --conf 0.5  # only label detections >= 50% confident
    uv run python src/eda/yolo_demo.py --classes 0  # people only (COCO class 0)
    uv run python src/eda/yolo_demo.py --track  # quick demo with persistent object IDs
    uv run python src/eda/yolo_demo.py --mode full --track  # track IDs across whole video
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imageio_ffmpeg
from loguru import logger
from ultralytics import YOLO

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()   # bundled static ffmpeg binary
PROJECT_DIR = Path("runs/detect")          # ultralytics output root
LOG_EVERY = 50                             # log progress every N frames


def _count_frames(path):
    """Total frames in a video, for progress/ETA reporting."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _stream_predict(model, source, name, label, track=False,
                    tracker="bytetrack.yaml", **predict_kwargs):
    """
    Run YOLO in streaming mode (stream=True) and log progress as it goes.

    stream=True yields one Results object at a time instead of accumulating them
    all in RAM (avoids the ultralytics OOM warning on long videos), and gives us
    a per-frame hook to report throughput (fps) and ETA via loguru. Returns the
    ultralytics save_dir of the annotated video.

    When track=True, dispatch to model.track() instead of model.predict() so
    objects keep a persistent ID across frames (ByteTrack/BoT-SORT). Tracking is
    stateful and needs sequential frames, so it must not be parallel-chunked.
    """
    total = _count_frames(source)
    stride = predict_kwargs.get("vid_stride", 1)
    expected = max(1, total // stride)        # results we'll actually iterate over
    mode = "track" if track else "detect"
    logger.info(f"[{label}] start ({mode}): {expected} frames to annotate (runs/detect/{name})")

    common = dict(source=str(source), stream=True, save=True,
                  name=name, exist_ok=True, verbose=False, **predict_kwargs)
    stream = (model.track(tracker=tracker, **common) if track
              else model.predict(**common))

    t0 = time.perf_counter()
    save_dir, done = None, 0
    for done, r in enumerate(stream, start=1):
        save_dir = r.save_dir
        if done % LOG_EVERY == 0:
            elapsed = time.perf_counter() - t0
            fps = done / elapsed if elapsed else 0.0
            eta = (expected - done) / fps if fps else 0.0
            logger.info(f"[{label}] {done}/{expected} | {fps:.1f} fps | ETA {eta:.0f}s")

    elapsed = time.perf_counter() - t0
    fps = done / elapsed if elapsed else 0.0
    logger.success(f"[{label}] done: {done} frames in {elapsed:.1f}s ({fps:.1f} fps) -> {save_dir}")
    return save_dir

# Same video as the notebook: eeg_data_path / "AC3.mp4"
EEG_DATA_PATH = Path("/Users/ansonliu/Downloads/Example EEG Data")
VIDEO_FILE = EEG_DATA_PATH / "AC3.mp4"


def _process_chunk(task):
    """
    Worker: load a fresh YOLO model and annotate one chunk.

    Module-level (not a method) so it is picklable for ProcessPoolExecutor.
    Returns the path to the annotated chunk video.
    """
    model_name, chunk_path, name, threads, predict_kwargs = task
    if threads:
        import torch
        torch.set_num_threads(threads)   # avoid CPU oversubscription across workers
    model = YOLO(model_name)
    save_dir = _stream_predict(model, chunk_path, name=name, label=name, **predict_kwargs)
    return str(Path(save_dir) / Path(chunk_path).name)


class YoloVideoDemo:
    """Wraps a YOLO model and runs it over a video in quick / full / chunked mode."""

    def __init__(self, video_file, model_name="yolo26s.pt"):
        self.video_file = Path(video_file)
        if not self.video_file.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_file}\n"
                "Update EEG_DATA_PATH to wherever 'AC3.mp4' lives on this machine."
            )
        self.model_name = model_name
        # Weights download from Ultralytics on first use.
        self.model = YOLO(model_name)

    def _predict(self, source, name, **kwargs):
        """
        Stream inference and let ultralytics save the annotated video.

        Pins name + exist_ok so reruns land in runs/detect/<name>/ instead of the
        auto-incrementing predict, predict2, predict3 folders. Returns the save_dir.
        """
        return _stream_predict(self.model, source, name=name, label=name, **kwargs)

    def _make_clip(self, n_frames):
        """Write the first n_frames to a temp mp4 so the quick demo stays fast."""
        cap = cv2.VideoCapture(str(self.video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        clip_path = self.video_file.with_name(f"{self.video_file.stem}_clip{n_frames}.mp4")

        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        written = 0
        while written < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written += 1
        cap.release()
        writer.release()
        print(f"Quick clip: {written} frames -> {clip_path}")
        return clip_path

    def _duration(self):
        """Video duration in seconds (frame count / fps, via OpenCV)."""
        cap = cv2.VideoCapture(str(self.video_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n_frames / fps if fps else 0.0

    def _split(self, n_chunks, workdir):
        """Split the video into n_chunks equal time slices with ffmpeg (stream copy)."""
        seg = self._duration() / n_chunks
        paths = []
        for i in range(n_chunks):
            out = workdir / f"chunk_{i:03d}.mp4"
            cmd = [FFMPEG, "-y", "-ss", f"{i * seg:.3f}", "-i", str(self.video_file),
                   "-t", f"{seg:.3f}", "-c", "copy", str(out)]
            subprocess.run(cmd, check=True, capture_output=True)
            paths.append(out)
        return paths

    def _concat(self, paths, out_path):
        """Stitch annotated chunks back into one mp4 with the ffmpeg concat demuxer."""
        listfile = out_path.with_suffix(".txt")
        listfile.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
        cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
               "-c", "copy", str(out_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        listfile.unlink(missing_ok=True)
        return out_path

    def run_quick(self, n_frames=300, **kwargs):
        """Fast sanity check: detect on a short clip from the start of the video."""
        clip = self._make_clip(n_frames)
        return self._predict(clip, name="quick", **kwargs)

    def run_full(self, vid_stride=1, **kwargs):
        """Detect across the whole video (vid_stride > 1 samples frames to go faster)."""
        return self._predict(self.video_file, name="full", vid_stride=vid_stride, **kwargs)

    def run_chunked(self, n_chunks=4, parallel=True, **kwargs):
        """
        Split the video into n_chunks with ffmpeg, detect on each chunk (in
        parallel by default so it finishes quicker), then stitch the annotated
        chunks back into a single video with ffmpeg.
        """
        # raw split chunks live in a temp dir that auto-cleans on exit; only the
        # stitched, annotated video is kept (under runs/detect/chunked/).
        with tempfile.TemporaryDirectory(prefix=f"{self.video_file.stem}_chunks_") as tmp:
            chunks = self._split(n_chunks, Path(tmp))

            print(f"Split into {len(chunks)} chunks -> {tmp}")

            # divide CPU threads across workers so parallel chunks don't fight for cores
            threads = max(1, (os.cpu_count() or 1) // n_chunks) if parallel else None
            tasks = [(self.model_name, str(c), f"chunk_{i:03d}", threads, kwargs)
                     for i, c in enumerate(chunks)]

            if parallel and n_chunks > 1:
                with ProcessPoolExecutor(max_workers=n_chunks) as ex:
                    annotated = list(ex.map(_process_chunk, tasks))
            else:
                annotated = [_process_chunk(t) for t in tasks]

            out_path = PROJECT_DIR / "chunked" / f"{self.video_file.stem}_annotated.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._concat(annotated, out_path)
            print(f"Stitched {len(annotated)} chunks -> {out_path}")

        return out_path

    def run(self, mode="quick", **kwargs):
        """Dispatch to the requested runner."""
        if mode == "quick":
            return self.run_quick(**kwargs)
        if mode == "full":
            return self.run_full(**kwargs)
        if mode == "chunked":
            return self.run_chunked(**kwargs)
        raise ValueError(f"unknown mode: {mode!r} (expected quick/full/chunked)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO over the POV walk video.")
    parser.add_argument("--mode", choices=["quick", "full", "chunked"], default="quick")
    parser.add_argument("--video", default=str(VIDEO_FILE), help="source video path")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO weights to load")
    parser.add_argument("--frames", type=int, default=300,
                        help="quick mode: number of frames from the start")
    parser.add_argument("--stride", type=int, default=1,
                        help="full mode: process every Nth frame (vid_stride)")
    parser.add_argument("--chunks", type=int, default=4,
                        help="chunked mode: number of ffmpeg splits")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True,
                        help="chunked mode: process chunks in parallel")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold: only draw detections at or above this (0-1)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold: lower => more aggressive dedup of overlapping boxes")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="restrict to these COCO class ids (e.g. --classes 0 2 means person, car)")
    parser.add_argument("--track", action="store_true",
                        help="track objects with persistent IDs across frames (model.track)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="tracker config when --track is set (bytetrack.yaml or botsort.yaml)")
    args = parser.parse_args()

    # Detection knobs forwarded to ultralytics' predictor for every mode.
    predict_kwargs = {"conf": args.conf, "iou": args.iou}
    if args.classes is not None:
        predict_kwargs["classes"] = args.classes
    if args.track:
        predict_kwargs["track"] = True
        predict_kwargs["tracker"] = args.tracker
        if args.mode == "chunked":
            logger.warning(
                "tracking needs sequential frames, but chunked mode splits the "
                "video into independent (parallel) slices: object IDs will reset "
                "at every chunk boundary. Use --mode full for continuous IDs."
            )

    demo = YoloVideoDemo(args.video, model_name=args.model)
    if args.mode == "quick":
        demo.run(mode="quick", n_frames=args.frames, **predict_kwargs)
    elif args.mode == "full":
        demo.run(mode="full", vid_stride=args.stride, **predict_kwargs)
    else:
        demo.run(mode="chunked", n_chunks=args.chunks, parallel=args.parallel, **predict_kwargs)
