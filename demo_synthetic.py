"""
demo_synthetic.py — BSS/ICA on synthetic signals
=================================================
Generates three independent source signals, mixes them with a random
matrix A, runs the full ICA pipeline, and plots the results.

Usage
-----
    python demo_synthetic.py

No audio file is required — all signals are generated mathematically.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ica import blind_source_separation
from metrics import evaluate, print_report
from visualise import (
    plot_pipeline,
    plot_covariance,
    plot_eigenspectrum,
    plot_whitening_scatter,
)
from audio_io import simulate_mixing

OUTPUT_DIR = Path("outputs/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Generate independent source signals
# ---------------------------------------------------------------------------

def make_sources(n_samples: int = 20_000, sr: int = 8_000) -> np.ndarray:
    """
    Three statistically independent signals:
      s1 — sine wave (musical tone)
      s2 — sawtooth wave (different harmonic content)
      s3 — band-limited noise (speech-like)
    """
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)

    s1 = np.sin(2 * np.pi * 440 * t)                             # 440 Hz sine

    # Sawtooth: sum of harmonics
    s2 = sum(
        np.sin(2 * np.pi * k * 330 * t) / k
        for k in range(1, 9)
    )
    s2 /= np.max(np.abs(s2))

    # Band-limited noise (low-pass filtered white noise via simple moving average)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n_samples)
    window = 20
    s3 = np.convolve(noise, np.ones(window) / window, mode="same")
    s3 /= np.max(np.abs(s3))

    S = np.vstack([s1, s2, s3])                                   # (3, n_samples)
    return S, sr


# ---------------------------------------------------------------------------
# 2. Mix
# ---------------------------------------------------------------------------

def main():
    print("=" * 52)
    print("  BSS / ICA — Synthetic Signal Demo")
    print("=" * 52)

    S, sr = make_sources()
    print(f"  Sources shape : {S.shape}  (n_sources × n_samples)")

    X, A = simulate_mixing(S, n_mics=3, random_state=7)
    print(f"  Mixed shape   : {X.shape}")
    print(f"\n  Mixing matrix A:\n{A}\n")

    # ---------------------------------------------------------------------------
    # 3. Run ICA pipeline
    # ---------------------------------------------------------------------------

    print("  Running ICA pipeline…")
    result = blind_source_separation(X, max_iter=1000, tol=1e-7, random_state=42)

    print(f"\n  Eigenvalues  D  = {np.round(result['D'], 4)}")
    print(f"  Whitened data cov ≈ I:\n{np.round(result['Z'] @ result['Z'].T / result['Z'].shape[1], 3)}")
    print(f"\n  W_total (≈ A⁻¹):\n{np.round(result['W_total'], 4)}")

    # ---------------------------------------------------------------------------
    # 4. Evaluate
    # ---------------------------------------------------------------------------

    eval_results = evaluate(S, result["S"])
    print_report(eval_results, result["converged"])

    S_hat_aligned = eval_results["S_aligned"]

    # ---------------------------------------------------------------------------
    # 5. Plots
    # ---------------------------------------------------------------------------

    print("  Saving figures…")

    fig1 = plot_pipeline(
        S, X, S_hat_aligned, sr=sr,
        title="ICA Pipeline — Synthetic Signals",
        save_path=str(OUTPUT_DIR / "pipeline_waveforms.png"),
    )
    plt.close(fig1)

    fig2 = plot_covariance(
        result["C"],
        save_path=str(OUTPUT_DIR / "covariance_matrix.png"),
    )
    plt.close(fig2)

    fig3 = plot_eigenspectrum(
        result["D"],
        save_path=str(OUTPUT_DIR / "eigenspectrum.png"),
    )
    plt.close(fig3)

    fig4 = plot_whitening_scatter(
        result["X_centered"], result["Z"],
        save_path=str(OUTPUT_DIR / "whitening_scatter.png"),
    )
    if fig4:
        plt.close(fig4)

    print(f"\n  Figures saved to:  {OUTPUT_DIR}/")
    print("  pipeline_waveforms.png")
    print("  covariance_matrix.png")
    print("  eigenspectrum.png")
    print("  whitening_scatter.png")
    print()


if __name__ == "__main__":
    main()
