"""Shared GLM + group-statistics helpers for the face-inversion analyses.

Two groups of functions live here.

**Ported verbatim** from cell 15 of ``{cornet,resnet}_full_run_yunet_detection.ipynb``:
``fit_betas``, ``rfx_ttest`` and ``bh_fdr``. They are reproduced byte-for-byte
(only the module docstring and the ``outlier_sd`` default being made explicit
differ) so that any notebook importing them gets numerically identical betas to
the inline copies. The full-run notebooks are deliberately left untouched — this
module is imported by ``model_comparison.ipynb`` only, and exists so that the
model-comparison statistics are computed by demonstrably the same code path that
produced the per-layer tables.

**New**, for the CORnet-S vs ResNet-50 comparison: ``paired_compare``,
``cohens_dz``, ``bf10_ttest``, ``tost`` and ``power_dz`` / ``min_detectable_dz``.

Only ``numpy`` and ``scipy`` are needed; nothing here imports ``torch``, ``mne``
or ``cv2``, so the module stays cheap to import.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import quad

# The authors' 4-SD outlier cut (posterior_betas / elec_glm), matching OUTLIER_SD
# in the full-run notebooks.
OUTLIER_SD = 4.0


# ---------------------------------------------------------------------------
# Ported verbatim from the full-run notebooks (cell 15)
# ---------------------------------------------------------------------------

def bh_fdr(p, q=0.05):
    "Benjamini-Hochberg FDR mask (verbatim from face_regressor_analysis.ipynb)."
    p = np.asarray(p); n = len(p)
    ranked = np.sort(p)
    below = ranked <= q * (np.arange(1, n + 1) / n)
    if not below.any():
        return np.zeros(n, bool)
    return p <= ranked[int(np.max(np.where(below)))]


def fit_betas(signal, regressor, outlier_sd=OUTLIER_SD):
    """GLM slope of (z-scored signal) ~ regressor, vectorized over columns.

    signal: (n_frames,) scalar ROI, or (n_frames, n_units) per-unit matrix.
    Z-scores each column over frames, drops frames >outlier_sd from that column's
    mean (per-column mask, matching posterior_betas / elec_glm), then OLS slope and
    its two-sided p-value. Returns (slope, p): scalars if signal is 1-D, else arrays.
    Zero-variance columns yield NaN (handled by nan-aware aggregation downstream).

    Note on naming: the notebooks call this the "encoding GLM". Nothing is fitted
    from model features to the EEG signal, so this is not an encoding model in the
    Naselaris (2011) sense — it is one contrast statistic applied independently to
    two systems (a parallel-GLM / shared-statistic design). The computation is
    unchanged; only the description is corrected.
    """
    x = np.asarray(regressor, float)
    Y = np.asarray(signal, float)
    one_d = Y.ndim == 1
    if one_d:
        Y = Y[:, None]
    Z = stats.zscore(Y, axis=0)                          # per-column z-score over frames
    M = (np.abs(Z) <= outlier_sd).astype(float)          # per-column keep mask
    xb = x[:, None]
    n = M.sum(0)
    xbar = (M * xb).sum(0) / n
    ybar = (M * Z).sum(0) / n
    Sxx = (M * xb**2).sum(0) - n * xbar**2
    Sxy = (M * xb * Z).sum(0) - n * xbar * ybar
    Syy = (M * Z**2).sum(0) - n * ybar**2
    slope = Sxy / Sxx
    df = n - 2
    resid = np.clip(Syy - slope * Sxy, 0, None)
    se = np.sqrt((resid / df) / Sxx)
    t = slope / se
    p = 2 * stats.t.sf(np.abs(t), df)
    if one_d:
        return float(slope[0]), float(p[0])
    return slope, p


def rfx_ttest(betas):
    "Group random-effects: one-sample t-test of per-subject betas vs 0 -> (t, p)."
    b = np.asarray(betas, float)
    b = b[np.isfinite(b)]
    t, p = stats.ttest_1samp(b, 0.0)
    return float(t), float(p)


# ---------------------------------------------------------------------------
# Model comparison (new)
# ---------------------------------------------------------------------------

def cohens_dz(diff):
    """Cohen's dz for a paired design: mean(diff) / sd(diff), sd with ddof=1.

    dz is the effect size the paired one-sample t-test is powered on, so it is
    also the natural unit in which to express a TOST equivalence bound.
    """
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    return float(d.mean() / d.std(ddof=1))


def bf10_ttest(t, n, r=0.707):
    """JZS Bayes factor (BF10) for a one-sample / paired t-test (Rouder et al. 2009).

    ``r`` is the Cauchy prior scale; 0.707 is the standard "medium" default.
    BF10 > 1 favours a difference, BF10 < 1 favours the null. Report BF01 = 1/BF10
    when the claim is absence of a difference. Keysers et al. (2020) give the
    conventional reading: BF01 < 3 is only *anecdotal* evidence for the null,
    which is what n=8 delivers here.
    """
    df = n - 1

    def _num(g):
        return ((1 + n * g) ** -0.5
                * (1 + t**2 / ((1 + n * g) * df)) ** (-(df + 1) / 2)
                * (2 * np.pi) ** -0.5 * r * g**-1.5 * np.exp(-r**2 / (2 * g)))

    den = (1 + t**2 / df) ** (-(df + 1) / 2)
    integral, _ = quad(_num, 0, np.inf, limit=200)
    return float(integral / den)


def tost(diff, sesoi_dz):
    """Two one-sided tests for equivalence on paired differences.

    ``sesoi_dz`` is the smallest effect size of interest, in Cohen's dz, and is
    converted to raw units via the observed sd of the differences. Returns
    ``(p_tost, equivalent)`` where ``p_tost`` is the larger of the two one-sided
    p-values; ``equivalent`` is ``p_tost < 0.05``.

    Pre-specify ``sesoi_dz`` before reporting: at n=8 only very large bounds are
    testable (see ``min_detectable_dz``), so a bound chosen after seeing the data
    is not a meaningful equivalence claim.
    """
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = d.size
    df = n - 1
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    lo, hi = -sesoi_dz * sd, sesoi_dz * sd
    p_lower = stats.t.sf((d.mean() - lo) / se, df)      # H0: diff <= lo
    p_upper = stats.t.cdf((d.mean() - hi) / se, df)     # H0: diff >= hi
    p_tost = float(max(p_lower, p_upper))
    return p_tost, bool(p_tost < 0.05)


def power_dz(dz, n, alpha=0.05):
    "Two-tailed power of a one-sample/paired t-test at effect size dz and size n."
    df = n - 1
    crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = dz * np.sqrt(n)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def min_detectable_dz(n, target=0.80, alpha=0.05):
    "Smallest dz detectable at `target` power — the honest floor on any null claim."
    for dz in np.arange(0.05, 5.0, 0.01):
        if power_dz(dz, n, alpha) >= target:
            return float(round(dz, 2))
    return float("nan")


def paired_compare(a, b, sesoi_dz=1.0, labels=("A", "B")):
    """Full paired comparison of two models' per-subject betas.

    ``a`` and ``b`` are length-n arrays of per-subject betas for the same subjects
    in the same order. Returns a dict with the paired t-test, Cohen's dz, JZS Bayes
    factors, TOST at ``sesoi_dz``, and the across-subject Pearson correlation
    between the two models.

    The correlation is the load-bearing statistic when the two models do not
    differ: a non-significant paired t only fails to reject, whereas a high r shows
    the two models are driven by the same across-stimulus variance.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    t, p = stats.ttest_1samp(d, 0.0)
    bf10 = bf10_ttest(float(t), n)
    p_tost, equivalent = tost(d, sesoi_dz)
    r, p_r = stats.pearsonr(a, b)
    return {
        "labels": labels,
        "n": int(n),
        "mean_diff": float(d.mean()),
        "sd_diff": float(d.std(ddof=1)),
        "t": float(t),
        "p": float(p),
        "dz": cohens_dz(d),
        "bf10": bf10,
        "bf01": float(1.0 / bf10),
        "sesoi_dz": float(sesoi_dz),
        "p_tost": p_tost,
        "equivalent": equivalent,
        "r": float(r),
        "p_r": float(p_r),
    }


# ---------------------------------------------------------------------------
# Event-mode signal loading (mirrors _event_signals in the full-run notebooks)
# ---------------------------------------------------------------------------

def event_roi_betas(npz_path, layers):
    """Per-layer scalar-ROI beta for one subject's cached event activations.

    Reproduces the full-run notebooks' event path exactly: the ROI signal is the
    mean activation across units (``A.mean(axis=1)``) and the regressor is +0.5
    for ``IN`` / -0.5 for ``UP``. Returns ``(betas_by_layer, labels)``.
    """
    d = np.load(npz_path, allow_pickle=True)
    labels = np.array([str(s).strip().upper() for s in d["event_labels"]])
    reg = np.where(labels == "IN", 0.5, -0.5).astype(float)
    out = {}
    for L in layers:
        A = np.asarray(d[f"event_{L}"], dtype=float)      # (n_events, n_units)
        out[L], _ = fit_betas(A.mean(axis=1), reg)
    return out, labels
