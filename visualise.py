"""
visualise.py — Plotting utilities for the BSS/ICA pipeline
===========================================================
Produces the figures described in Script 3:
  - Waveforms of sources, mixed signals, recovered signals
  - Covariance matrix heatmap
  - Eigenspectrum bar chart
  - Whitened data scatter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

COLORS = {
    "source":    "#1D9E75",   # teal  — original sources
    "mixed":     "#BA7517",   # amber — mixed observations
    "recovered": "#185FA5",   # blue  — ICA output
    "whitened":  "#9B59B6",   # purple
    "neutral":   "#888780",
}

def _label_rows(ax, labels, fontsize=9):
    for i, label in enumerate(labels):
        ax.text(-0.01, (len(labels) - i - 0.5) / len(labels),
                label, transform=ax.transAxes,
                ha="right", va="center", fontsize=fontsize, color="#444")


# ---------------------------------------------------------------------------
# Figure 1 — Full pipeline waveforms
# ---------------------------------------------------------------------------

def plot_pipeline(
    S: np.ndarray,
    X: np.ndarray,
    S_hat: np.ndarray,
    sr: int = 1,
    title: str = "BSS / ICA Pipeline",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel figure: Original sources | Mixed signals | Recovered sources.

    Parameters
    ----------
    S     : (n_src, n_samples)   original sources
    X     : (n_ch,  n_samples)   mixed observations
    S_hat : (n_src, n_samples)   ICA-recovered sources
    sr    : sample rate (for time axis)
    """
    n_src = S.shape[0]
    n_mix = X.shape[0]
    n_rec = S_hat.shape[0]

    t = np.arange(S.shape[1]) / sr

    fig = plt.figure(figsize=(14, 2.2 * max(n_src, n_mix, n_rec) + 1.2))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)

    n_rows = max(n_src, n_mix, n_rec)
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.05, wspace=0.35)

    def _plot_col(signals, col, color, col_title):
        for i in range(signals.shape[0]):
            ax = fig.add_subplot(gs[i, col])
            ax.plot(t, signals[i], lw=0.6, color=color, alpha=0.9)
            ax.axhline(0, lw=0.4, color="#ccc")
            ax.set_ylabel(f"ch {i+1}", fontsize=8, rotation=0, labelpad=22, va="center")
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(col_title, fontsize=10, fontweight="500", pad=6)
            if i < signals.shape[0] - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time (s)" if sr > 1 else "sample", fontsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)

    _plot_col(S,     0, COLORS["source"],    "Original sources  S")
    _plot_col(X,     1, COLORS["mixed"],     "Mixed signals  X = AS")
    _plot_col(S_hat, 2, COLORS["recovered"], "Recovered sources  Ŝ = WX")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Covariance matrix heatmap
# ---------------------------------------------------------------------------

def plot_covariance(
    C: np.ndarray,
    title: str = "Covariance matrix  C = E[X Xᵀ]",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of the covariance matrix."""
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(C, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(C).max(), vmax=np.abs(C).max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = C.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"x{i+1}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"x{i+1}" for i in range(n)], fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="k" if abs(C[i,j]) < 0.5 * np.abs(C).max() else "w")
    ax.set_title(title, fontsize=10, fontweight="500", pad=8)
    ax.spines[:].set_linewidth(0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Eigenspectrum
# ---------------------------------------------------------------------------

def plot_eigenspectrum(
    D: np.ndarray,
    title: str = "Eigenspectrum  (diagonal of D)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of eigenvalues — shows variance explained by each PC."""
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(D))
    bars = ax.bar(x, D, color=COLORS["source"], alpha=0.85, width=0.6, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ{i+1}" for i in x], fontsize=9)
    ax.set_ylabel("eigenvalue  (variance)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="500", pad=8)
    for bar, val in zip(bars, D):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + D.max() * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — 2D scatter: before / after whitening
# ---------------------------------------------------------------------------

def plot_whitening_scatter(
    X: np.ndarray,
    Z: np.ndarray,
    max_pts: int = 3000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2D scatter comparing the first two channels before and after whitening.
    Only works when n_channels >= 2.
    """
    if X.shape[0] < 2:
        return None

    idx = np.random.choice(X.shape[1], size=min(max_pts, X.shape[1]), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, data, color, label in zip(
        axes,
        [X[:, idx], Z[:, idx]],
        [COLORS["mixed"], COLORS["whitened"]],
        ["Mixed  X (first 2 channels)", "Whitened  Z"],
    ):
        ax.scatter(data[0], data[1], s=1.5, alpha=0.35, color=color, linewidths=0)
        ax.set_xlabel("channel 1", fontsize=9)
        ax.set_ylabel("channel 2", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="500")
        ax.set_aspect("equal", "box")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(labelsize=8)

    fig.suptitle("Effect of whitening on data distribution", fontsize=11, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Single-file waveform (for the audio demo)
# ---------------------------------------------------------------------------

def plot_waveform(
    signal: np.ndarray,
    sr: int = 1,
    title: str = "Waveform",
    color: str = COLORS["source"],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Simple single-channel waveform plot."""
    t = np.arange(len(signal)) / sr
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(t, signal, lw=0.5, color=color, alpha=0.9)
    ax.axhline(0, lw=0.4, color="#ccc")
    ax.set_title(title, fontsize=10, fontweight="500")
    ax.set_xlabel("time (s)" if sr > 1 else "sample", fontsize=8)
    ax.set_ylabel("amplitude", fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
