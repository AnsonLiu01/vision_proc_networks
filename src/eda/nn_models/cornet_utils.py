"""Reusable CORnet-S inference plumbing for the face-inversion encoding GLM.

Extracted from ``cornet_analysis.ipynb`` so the ``"dense"`` sampling path of
``cornet_full_run.ipynb`` (decode a subject's head-cam video frame by frame and
run CORnet on each frame) can reuse the model, hooks, preprocessing, video, and
face-crop helpers without duplicating them in the notebook. Mirrors the
``eeg_utils.py`` precedent.

The ``"event"`` sampling path needs none of this — it only reads the cached
per-event activations in ``data/cornet_analysis_outputs/sub{N}_cornet*.npz`` —
so the heavy imports (``torch``, ``torchvision``, ``cornet``, ``cv2``) are
deferred into the functions that need them. Importing this module therefore
stays cheap and does not require a GPU-less torch install to be present.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (lifted from cornet_analysis.ipynb Configuration cell)
# ---------------------------------------------------------------------------

LAYERS = ["V1", "V2", "V4", "IT"]          # early -> late ventral stream

# ImageNet preprocessing (CORnet-S was trained with these stats).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Use the original corrected video by default; flip for the Shotcut-stabilised
# ``_shotcut.mp4`` version (see the note in cornet_analysis.ipynb).
USE_STABILISED_VIDEO = False


def _find_data_root() -> Path:
    "Locate 'data/SUBJECT eeg data' walking up from this file, then the cwd."
    starts = [Path(__file__).resolve(), Path.cwd().resolve()]
    for start in starts:
        for base in [start, *start.parents]:
            cand = base / "data" / "SUBJECT eeg data"
            if cand.exists():
                return cand
    raise FileNotFoundError("Could not locate 'data/SUBJECT eeg data'")


# ---------------------------------------------------------------------------
# CORnet-S model + forward hooks
# ---------------------------------------------------------------------------

def _patch_torch_load_cpu() -> None:
    """Force ``torch.load`` onto CPU (the pretrained weights were authored for GPU).

    Recursion-guarded so calling it twice does not wrap the patch around itself
    (which would blow the stack on the next load) — matches the notebook cell.
    """
    import torch

    if getattr(torch.load, "_cpu_patched", False):
        return
    _original_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        kwargs["map_location"] = torch.device("cpu")
        return _original_torch_load(*args, **kwargs)

    _cpu_torch_load._cpu_patched = True
    torch.load = _cpu_torch_load


def load_model_and_hooks(layers: list[str] = LAYERS):
    """Load pretrained CORnet-S on CPU with forward hooks on the cortical areas.

    Returns ``(model, activations, handles)`` where ``activations`` is a dict the
    hooks write each layer's raw output tensor into on every forward pass, and
    ``handles`` are the hook handles (call ``.remove()`` to detach). CORnet is
    wrapped in ``nn.DataParallel``; the areas live under ``model.module``.
    """
    import cornet

    _patch_torch_load_cpu()
    model = cornet.cornet_s(pretrained=True)
    model.eval()

    activations: dict[str, "object"] = {}

    def _get_activation(name):
        def hook(_module, _inp, output):
            activations[name] = output.detach()
        return hook

    handles = [
        getattr(model.module, layer).register_forward_hook(_get_activation(layer))
        for layer in layers
    ]
    return model, activations, handles


@lru_cache(maxsize=1)
def _get_transform():
    "torchvision preprocessing pipeline (cached; built lazily to defer imports)."
    import torchvision.transforms as transforms

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def frame_to_features(rgb_frame, model, activations, layers: list[str] = LAYERS):
    """RGB uint8 ndarray (H, W, 3) -> {layer: flattened float32 activation vector}.

    ``model`` and ``activations`` are the pair returned by ``load_model_and_hooks``;
    passing them in (rather than using notebook globals) keeps this importable.
    """
    import torch
    from PIL import Image

    img = Image.fromarray(rgb_frame)
    x = _get_transform()(img).unsqueeze(0)
    with torch.no_grad():
        _ = model(x)
    return {layer: activations[layer].flatten().cpu().numpy() for layer in layers}


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def video_path(sub: int, data_root: Path | None = None,
               use_stabilised: bool = USE_STABILISED_VIDEO) -> Path:
    root = _find_data_root() if data_root is None else data_root
    suffix = "_shotcut" if use_stabilised else ""
    return root / f"sub{sub}" / f"s{sub}_corrected{suffix}.mp4"


def open_video(sub: int, data_root: Path | None = None,
               use_stabilised: bool = USE_STABILISED_VIDEO):
    "Return (cap, fps, n_frames) for a subject's head-cam video."
    import cv2

    path = video_path(sub, data_root, use_stabilised)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for sub{sub}: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, n_frames


def grab_frame(cap, t_sec: float, fps: float):
    "Return the RGB frame at video time t_sec, or None if out of range."
    import cv2

    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Face cropping (used when input_mode == "facecrop" in the dense path)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _face_cascade():
    import cv2

    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face_bbox(rgb, scale_factor: float = 1.05, min_neighbors: int = 5,
                     min_size: int = 50, min_confidence: float | None = None,
                     return_confidence: bool = True):
    """Largest face box (x, y, w, h) in ORIGINAL-frame coords, or None.

    Runs the detector on the frame AND its 180-rotated copy so inverted faces are
    found as reliably as upright ones; the rotation only LOCATES the face — the
    crop keeps the original orientation.

    Confidence: uses OpenCV's ``detectMultiScale3`` so each candidate box carries
    a ``levelWeight`` — an UNCALIBRATED cascade score (higher = the box survived
    deeper into the cascade; it is NOT a 0-1 probability, and its scale drifts
    with ``min_neighbors``). If ``min_confidence`` is given, boxes scoring below
    it are discarded, so a weak/spurious detection becomes a "no face" and
    ``crop_to_face`` falls back to the central square. With
    ``return_confidence=True`` returns ``(bbox, confidence)``; confidence is the
    kept box's score, or ``None`` when no box survived.
    """
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    best, best_area, best_conf = None, 0, None
    for rot in (0, 180):
        g = gray if rot == 0 else cv2.rotate(gray, cv2.ROTATE_180)
        objs, _reject, weights = _face_cascade().detectMultiScale3(
            g, scaleFactor=scale_factor, minNeighbors=min_neighbors,
            minSize=(min_size, min_size), outputRejectLevels=True)
        weights = np.ravel(weights)
        for i, (x, y, w, h) in enumerate(objs):
            conf = float(weights[i]) if i < weights.size else 0.0
            if min_confidence is not None and conf < min_confidence:
                continue
            if rot == 180:
                x, y = W - x - w, H - y - h
            if w * h > best_area:
                best_area, best, best_conf = w * h, (int(x), int(y), int(w), int(h)), conf
    if return_confidence:
        return best, best_conf
    return best


def crop_to_face(rgb, bbox, margin: float = 0.6):
    """Square crop around the face (+margin), orientation preserved.

    No face -> central square (drops the black letterbox bars; faces are ~centred).
    """
    H, W = rgb.shape[:2]
    if bbox is None:
        s = min(H, W)
        return rgb[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    s = min(int(round(max(w, h) * (1.0 + margin))), H, W)
    x0 = int(round(min(max(cx - s / 2.0, 0), W - s)))
    y0 = int(round(min(max(cy - s / 2.0, 0), H - s)))
    return rgb[y0:y0 + s, x0:x0 + s]
