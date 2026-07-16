"""Corrected EEG↔video alignment + YuNet face detection.

Sibling of ``cornet_utils.py`` / ``resnet_utils.py``, used by the
``*_full_run_yunet_detection.ipynb`` notebooks. Both the CORnet and ResNet runs
import this same module so their alignment and detection are guaranteed
identical and the two can be compared like for like.

It exists because ``face_alignment.ipynb`` established two independent bugs in
the original pipeline:

**1. Alignment — ``t_video = onset - t1`` mixes two clocks.** The
``RmIaf_mobileface.set`` annotation onsets live in a *front-trimmed* clock, while
``t1`` from ``Triggers.xlsx`` lives in the *original* recording clock. Regressing
one on the other gives slope exactly 1.000000 with 0.0 ms residual and a
non-zero intercept — the trim, in whole seconds:

    sub    1    2   3   4    5   6     7    8
    trim  90   51   0   0   33   0   155   39

Subs 3/4/6 only ever "worked" because their trim is 0. The fix is
``Triggers.xlsx`` **column K** ("Time in video frame rate"), which is built from
columns C and T1 *both in the original clock*, so the trim cancels and no ``t1``
arithmetic is needed. K also carries a frame-rate factor of exactly 0.96 (=24/25)
for the 25 fps subjects (1, 2) and 1.0 for the 30 fps subjects.

**2. Detection — Haar's ``min_size=50`` cuts into the stimuli.** The stimuli are
small *printed* faces on walls: six of eight subjects have faces below 50 px
(smallest 26 px), and subs 1-2 are 288x480 portrait video with a median face of
only ~58-63 px. On top of that ~66% of each frame is letterbox padding. YuNet on
de-letterboxed, 2x-upscaled frames takes 7/8 subjects to 93.5-100% (sub5 is
capped by 14.5% black video, a data-quality limit).

``detect_face_bbox`` here is a **drop-in replacement** for
``cornet_utils.detect_face_bbox`` — same signature and return contract — so the
notebooks swap ``cu.detect_face_bbox`` for ``yu.detect_face_bbox`` and change
nothing else. The one semantic difference: ``min_confidence`` is now a real
0-1 probability, not Haar's uncalibrated ``levelWeight``.
"""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

import eeg_utils as eu
from cornet_utils import crop_to_face  # noqa: F401  (re-exported; model-agnostic)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# YuNet score threshold. Unlike Haar's levelWeight this IS a probability, so the
# value means something across subjects and videos. 0.3 was chosen in
# face_alignment.ipynb §10 (it clears 80% on 7/8 subjects).
YUNET_THRESH = 0.3

YUNET_UPSCALE = 2.0        # the printed faces are small; give the detector more pixels
YUNET_DELETTERBOX = True   # ~66% of each frame is black padding on subs 3-8

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = eu.DATA_ROOT.parent / "models" / "face_detection_yunet_2023mar.onnx"

# A frame whose mean pixel is below this carries no usable image. Only sub5 has
# any (14.5% of its video is black dropout), and those events are missing data
# rather than detection failures.
DARK_MEAN = 10.0

# Triggers come in pairs: a press for the LEFT face, then ~2 s later a press for
# the RIGHT face of the same A4 sheet; pairs are ~10 s apart. Two events closer
# than this (seconds) are the two members of one pair. See face_alignment.ipynb §13.
PAIR_GAP_S = 5.0


# ---------------------------------------------------------------------------
# Alignment: Triggers.xlsx column K
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _sheet_rows(sub: int) -> tuple:
    return tuple(eu._read_xlsx_sheet(eu.DATA_ROOT / f"sub{sub}" / "Triggers.xlsx"))


@lru_cache(maxsize=None)
def xlsx_face_events(sub: int) -> tuple:
    """``((k_video, 'UP'|'IN'), ...)`` for one subject, sorted by time.

    ``k_video`` is column K — the video seek time in seconds, already corrected
    for both the .set front-trim and the frame-rate factor. Use it directly as
    ``t_video``; do **not** subtract ``t1`` from it.
    """
    out = []
    for r in _sheet_rows(sub):
        lab = str(r.get("N", "")).strip().lower()
        if lab in ("up", "in") and "K" in r and "C" in r:
            out.append((float(r["K"]), int(r["C"]), lab.upper()))
    out.sort(key=lambda x: x[1])                      # order by original-clock sample
    return tuple((k, lab) for k, _c, lab in out)


def pair_roles(times, cut: float = PAIR_GAP_S) -> list:
    """Per-event press role ('left' | 'right' | None) from the pair gap structure.

    The paradigm's triggers come in pairs — a press for the LEFT face, then ~2 s
    later a press for the RIGHT face of the same sheet, with ~10 s between pairs.
    Given event times in seconds **already sorted in time** (as
    ``xlsx_face_events`` returns them), this returns a list the same length: the
    two members of each pair are labelled ``'left'`` / ``'right'`` and any
    unpaired press is ``None``.

    Pass the role to ``detect_face_bbox`` so a left/right pair is read off the two
    different side-by-side faces instead of both collapsing onto the single
    top-scoring one. Grouping is by gap, not by even/odd index, because some
    subjects' pairing is phase-shifted (sub2). The left/right *assignment* is by
    on-screen x-position; even if a subject's press order were mirrored, the pair
    still receives the two distinct faces (and both share the UP/IN condition), so
    the RDM input is correct either way.
    """
    roles = [None] * len(times)
    i = 0
    while i < len(times):
        if i + 1 < len(times) and (times[i + 1] - times[i]) < cut:
            roles[i], roles[i + 1] = "left", "right"
            i += 2
        else:
            i += 1
    return roles


@lru_cache(maxsize=None)
def fps_factor(sub: int) -> float:
    """The K/H frame-rate factor, derived from the sheet (0.96 for 25 fps subs, else 1.0)."""
    ratios = []
    for r in _sheet_rows(sub):
        if str(r.get("N", "")).strip().lower() in ("up", "in") and "H" in r and "K" in r:
            h = float(r["H"])
            if h:
                ratios.append(float(r["K"]) / h)
    if not ratios:
        raise ValueError(f"sub{sub}: no annotated rows with H and K")
    uniq = np.unique(np.round(ratios, 5))
    if uniq.size != 1:
        raise ValueError(f"sub{sub}: K/H is not constant: {uniq}")
    return float(uniq[0])


@lru_cache(maxsize=None)
def video_trim_s(sub: int) -> int:
    """Whole seconds trimmed off the FRONT of the .set, vs the original recording.

    Measured, not assumed: regress the .set annotation onsets on the xlsx column-C
    times. Slope must come out at 1.0 (a pure shift, not clock drift) and the
    intercept is the trim. Events are paired **by sorted order** — nearest-neighbour
    matching mispairs under a large offset (sub7's is 155 s) and hides the shift.
    """
    raw = eu.load_eeg(sub)
    sf = raw.info["sfreq"]
    onsets = np.array([o for o, _ in eu.get_face_events(raw)])
    x_eeg = np.array([c for c in sorted(
        int(r["C"]) for r in _sheet_rows(sub)
        if str(r.get("N", "")).strip().lower() in ("up", "in") and "C" in r)]) / sf
    if len(onsets) != len(x_eeg):
        raise ValueError(f"sub{sub}: {len(onsets)} .set events vs {len(x_eeg)} xlsx events")
    A = np.vstack([x_eeg, np.ones_like(x_eeg)]).T
    slope, inter = np.linalg.lstsq(A, onsets, rcond=None)[0]
    if abs(slope - 1.0) > 1e-4:
        raise ValueError(f"sub{sub}: slope {slope:.6f} != 1 -- clock DRIFT, not a shift; "
                         "the column-K fix assumes a pure shift")
    return int(round(-inter))


def video_to_eeg(t_video: float, sub: int) -> float:
    """Video time -> EEG time in the TRIMMED .set clock (the one annotations live in).

    The inverse of the column-K mapping::

        t_video = K                                (original clock, trim-invariant)
        t_eeg   = K / factor + t1 - trim           (trimmed .set clock)

    The ``t1 - trim`` term is the whole fix: ``t1`` is an original-clock quantity
    while the .set signal is a trimmed-clock one, and the trim reconciles them.
    """
    raw = eu.load_eeg(sub)
    t1 = eu.get_t1(sub, raw)
    return t_video / fps_factor(sub) + t1 - video_trim_s(sub)


# ---------------------------------------------------------------------------
# Detection: YuNet
# ---------------------------------------------------------------------------

def _yunet_model():
    "YuNet detector, downloading the 232 KB ONNX on first use."
    import cv2

    if not YUNET_PATH.exists():
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading YuNet -> {YUNET_PATH}")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    if getattr(_yunet_model, "_d", None) is None:
        _yunet_model._d = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320), 0.05, 0.3, 5000)
    return _yunet_model._d


def content_bbox(rgb, thresh: int = 12):
    "Bounding box (x, y, w, h) of the non-black content, i.e. the frame minus letterbox bars."
    import cv2

    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = g > thresh
    cols = np.flatnonzero(m.any(axis=0))
    rows = np.flatnonzero(m.any(axis=1))
    if cols.size == 0 or rows.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def _detect_all_faces(rgb, upscale: float, deletterbox: bool) -> list:
    """Every face in the frame, highest score first, deduped across the 0/180 passes.

    Each entry is ``{"bbox": (x, y, w, h), "cx", "cy", "w", "score"}`` with the box
    in ORIGINAL-frame coords. Shared core of ``detect_face_bbox``: it runs the
    detector on the frame **and its 180° rotation** (essential — half the stimuli
    are INVERTED faces and YuNet is trained on upright ones) and un-rotates each
    box; the rotation only LOCATES faces, orientation is not changed. A face found
    in both passes is kept once (highest score), so left/right selection sees
    distinct physical faces rather than a 0°/180° duplicate.
    """
    import cv2

    ox = oy = 0
    if deletterbox:
        bb = content_bbox(rgb)
        if bb is not None and bb[2] > 40 and bb[3] > 40:
            ox, oy, w, h = bb
            rgb = rgb[oy:oy + h, ox:ox + w]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if upscale != 1.0:
        bgr = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    H, W = bgr.shape[:2]
    if W < 20 or H < 20:
        return []

    det = _yunet_model()
    det.setInputSize((W, H))
    cands = []
    for rot in (0, 180):
        img = bgr if rot == 0 else cv2.rotate(bgr, cv2.ROTATE_180)
        _n, faces = det.detect(img)
        if faces is None:
            continue
        for f in faces:
            x, y, w, h, s = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            if rot == 180:
                x, y = W - x - w, H - y - h
            bx = int(round(x / upscale + ox)); by = int(round(y / upscale + oy))
            bw = int(round(w / upscale)); bh = int(round(h / upscale))
            cands.append({"bbox": (bx, by, bw, bh), "cx": bx + bw / 2.0,
                          "cy": by + bh / 2.0, "w": bw, "score": s})

    cands.sort(key=lambda d: -d["score"])
    kept = []
    for c in cands:
        if all((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2
               > (0.5 * max(c["w"], k["w"])) ** 2 for k in kept):
            kept.append(c)
    return kept


def detect_face_bbox(rgb, min_confidence: float | None = YUNET_THRESH,
                     return_confidence: bool = True,
                     upscale: float = YUNET_UPSCALE,
                     deletterbox: bool = YUNET_DELETTERBOX,
                     role: str | None = None):
    """Face box (x, y, w, h) in ORIGINAL-frame coords, or None.

    Drop-in replacement for ``cornet_utils.detect_face_bbox``: same signature and
    return contract (so ``crop_to_face`` still falls back to the central square on
    a None box), with one added keyword. Differences from the Haar version:

    - ``min_confidence`` is a real 0-1 probability (Haar's ``levelWeight`` was
      uncalibrated and drifted with ``min_neighbors``).
    - The frame is de-letterboxed and upscaled first, because the stimuli are
      small printed faces and ~66% of each frame is padding.

    ``role`` selects *which* face when a frame holds more than one — the sheets
    carry two faces side by side, and a left/right trigger pair should read off the
    two different faces, not both collapse onto the single top-scoring box (see
    face_alignment.ipynb §13):

    - ``None`` (default): the highest-scoring face — original, drop-in behaviour.
    - ``'left'`` / ``'right'``: among faces clearing ``min_confidence``, the
      leftmost / rightmost by on-screen x. Falls back to the top-scoring face when
      fewer than two faces qualify (nothing to disambiguate). Pair the role from
      ``pair_roles`` with events from ``xlsx_face_events``.

    The threshold semantics are unchanged for ``role=None``: the top face is
    returned only if it clears ``min_confidence``, else None (with its score).
    """
    faces = _detect_all_faces(rgb, upscale, deletterbox)
    if not faces:
        return (None, None) if return_confidence else None

    chosen = None
    if role in ("left", "right"):
        qual = (faces if min_confidence is None
                else [f for f in faces if f["score"] >= min_confidence])
        if len(qual) >= 2:
            chosen = (min(qual, key=lambda f: f["cx"]) if role == "left"
                      else max(qual, key=lambda f: f["cx"]))

    if chosen is None:                               # role None, or <2 qualifying faces
        best = faces[0]                              # highest score (faces is score-sorted)
        if min_confidence is not None and best["score"] < min_confidence:
            return (None, best["score"]) if return_confidence else None
        chosen = best

    return (chosen["bbox"], chosen["score"]) if return_confidence else chosen["bbox"]


def is_dark(rgb) -> bool:
    "True if the frame is effectively black (a camera dropout; only sub5 has these)."
    return float(np.mean(rgb)) < DARK_MEAN
