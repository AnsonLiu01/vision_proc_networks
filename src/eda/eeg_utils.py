"""Shared EEG loading, video-start alignment, and face-regressor helpers.

These are ported from ``cornet_analysis.ipynb`` so the face-inversion regressor
analysis (``face_regressor_analysis.ipynb``) can reuse the exact same data
loading and trigger logic without re-running the CORnet pipeline.

The dataset is the mobile-EEG face-inversion paradigm (Krugliak & Clarke 2022).
Each subject's ``RmIaf_mobileface.set`` holds 64 scalp channels, three
accelerometer channels (``x_dir``/``y_dir``/``z_dir``), a pre-convolved
``FaceInversion`` regressor channel, and ``UP``/``IN`` face-event annotations.
EEG↔video alignment is grounded in ``Triggers.xlsx`` (the ``T 1`` video-start
trigger), which is authoritative and present for every subject.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mne
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS = range(1, 9)

# 64 scalp channels, then x/y/z accelerometers + the FaceInversion regressor.
N_EEG_CHANNELS = 64
ACCEL_CHANNELS = ["x_dir", "y_dir", "z_dir"]
EMBEDDED_REGRESSOR = "FaceInversion"

FACE_LABELS = {"UP": "upright", "IN": "inverted"}

# Posterior ROI (~17 channels): occipital + parieto-occipital + parietal, matching
# the "all occipital, parietal and parietal-occipital electrodes" ROI of the paper.
POSTERIOR_ROI = [
    "O1", "Oz", "O2", "Iz",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
]


def _find_data_root() -> Path:
    "Locate 'data/SUBJECT eeg data' walking up from this file, then the cwd."
    starts = [Path(__file__).resolve(), Path.cwd().resolve()]
    for start in starts:
        for base in [start, *start.parents]:
            cand = base / "data" / "SUBJECT eeg data"
            if cand.exists():
                return cand
    raise FileNotFoundError("Could not locate 'data/SUBJECT eeg data'")


DATA_ROOT = _find_data_root()


# ---------------------------------------------------------------------------
# Triggers.xlsx (dependency-free reader) + video-start sample
# ---------------------------------------------------------------------------

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_xlsx_sheet(path: Path) -> list[dict]:
    "Dependency-free .xlsx reader: list of {column_letter: value} per row."
    z = zipfile.ZipFile(path)
    strings: list[str] = []
    try:
        tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in tree.findall(_XLSX_NS + "si"):
            strings.append("".join(n.text or "" for n in si.iter(_XLSX_NS + "t")))
    except KeyError:
        pass
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.find(_XLSX_NS + "sheetData").findall(_XLSX_NS + "row"):
        cells = {}
        for c in row.findall(_XLSX_NS + "c"):
            v = c.find(_XLSX_NS + "v")
            if v is None:
                continue
            val = strings[int(v.text)] if c.get("t") == "s" else v.text
            col = "".join(ch for ch in c.get("r") if ch.isalpha())
            cells[col] = val
        rows.append(cells)
    return rows


def video_start_sample(sub: int) -> int:
    # Column B = trigger label, column C = EEG samples from EEG start to that
    # trigger. Each sheet has two 'T 1' rows: video start (earliest) and video
    # end (latest); we take the earliest.
    rows = _read_xlsx_sheet(DATA_ROOT / f"sub{sub}" / "Triggers.xlsx")
    samples = [
        int(r["C"]) for r in rows
        if str(r.get("B", "")).strip().upper().replace(" ", "") == "T1" and "C" in r
    ]
    if not samples:
        raise ValueError(f"No 'T 1' trigger found in sub{sub} Triggers.xlsx")
    return min(samples)


# ---------------------------------------------------------------------------
# EEG loading + events + alignment
# ---------------------------------------------------------------------------

def eeg_path(sub: int) -> Path:
    return DATA_ROOT / f"sub{sub}" / "RmIaf_mobileface.set"


def load_eeg(sub: int) -> mne.io.BaseRaw:
    "Preloaded Raw (500 Hz, 68 channels)."
    return mne.io.read_raw_eeglab(str(eeg_path(sub)), preload=True, verbose="ERROR")


def get_t1(sub: int, raw: mne.io.BaseRaw) -> float:
    # Video-start time (s from EEG start), read from Triggers.xlsx (authoritative,
    # present for all subjects). Where the .set also kept a 'T 1' annotation,
    # cross-check and warn on mismatch.
    sf = raw.info["sfreq"]
    t1 = video_start_sample(sub) / sf
    mne_t1 = [
        float(o) for o, d in zip(raw.annotations.onset, raw.annotations.description)
        if d.strip().upper().replace(" ", "") == "T1"
    ]
    if mne_t1 and abs(min(mne_t1) - t1) > 0.1:
        print(f"  sub{sub}: WARNING xlsx T1={t1:.3f}s vs .set T1={min(mne_t1):.3f}s")
    return t1


def get_face_events(raw: mne.io.BaseRaw) -> list[tuple[float, str]]:
    "Sorted list of (onset_s, 'UP'|'IN') for the face events."
    out = [
        (float(onset), desc.strip().upper())
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if desc.strip().upper() in FACE_LABELS
    ]
    return sorted(out, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Face regressors
# ---------------------------------------------------------------------------

def build_face_regressors(raw: mne.io.BaseRaw, hamming_s: float = 2.0) -> dict:
    """Continuous, full-length face regressors at ``raw``'s current sample rate.

    Rate-agnostic: builds everything from ``raw``'s annotations, ``sfreq`` and
    ``n_times``, so calling it after ``raw.resample(fps)`` yields regressors on
    the downsampled grid automatically.

    Returns a dict with:
      - ``face_inversion``: stick function (-0.5 upright / +0.5 inverted) convolved
        with a ``hamming_s`` Hamming window (the paper's construction).
      - ``face_presence``: stick function (1 at any UP/IN event) convolved with the
        same window — "when *any* face appears", orientation-agnostic.
      - ``accel``: (3, n_times) array of the x/y/z accelerometer channels.
      - ``t``: time axis in seconds (EEG time).
    """
    sf = raw.info["sfreq"]
    n = raw.n_times
    events = get_face_events(raw)

    inv_stick = np.zeros(n, dtype=float)
    pres_stick = np.zeros(n, dtype=float)
    for onset_s, label in events:
        idx = int(round(onset_s * sf))
        if 0 <= idx < n:
            inv_stick[idx] += 0.5 if label == "IN" else -0.5
            pres_stick[idx] += 1.0

    win_len = max(int(round(hamming_s * sf)), 1)
    window = np.hamming(win_len)
    face_inversion = np.convolve(inv_stick, window, mode="same")
    face_presence = np.convolve(pres_stick, window, mode="same")

    present = [c for c in ACCEL_CHANNELS if c in raw.ch_names]
    accel = raw.get_data(picks=present) if present else np.empty((0, n))

    return {
        "face_inversion": face_inversion,
        "face_presence": face_presence,
        "accel": accel,
        "t": np.arange(n) / sf,
        "sfreq": sf,
        "events": events,
    }


def regressor_embedded_corr(raw: mne.io.BaseRaw) -> float:
    """Correlation between the rebuilt face_inversion regressor and the embedded
    ``FaceInversion`` channel, at the raw's native rate. Sanity check that our
    reconstruction matches the regressor the original authors stored."""
    if EMBEDDED_REGRESSOR not in raw.ch_names:
        return float("nan")
    reg = build_face_regressors(raw)["face_inversion"]
    embedded = raw.get_data(picks=[EMBEDDED_REGRESSOR])[0]
    if np.std(reg) == 0 or np.std(embedded) == 0:
        return float("nan")
    return float(np.corrcoef(reg, embedded)[0, 1])
