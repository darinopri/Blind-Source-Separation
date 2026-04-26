"""
demo_synthetic.py — BSS/ICA on synthetic signals
=================================================

Generates three independent source signals, mixes them with a random
matrix A, runs the full ICA pipeline, saves plots AND audio files.

Usage
-----
python demo_synthetic.py              # basic run
python demo_synthetic.py --compare   # also compares against scikit-learn FastICA

No audio file is required — all signals are generated mathematically.
"""

import argparse
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
from audio_io import simulate_mixing, save_audio

OUTPUT_DIR = Path("outputs/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="BSS/ICA demo on synthetic signals."
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Also run scikit-learn FastICA and print a side-by-side comparison table.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------

def make_sources(n_samples: int = 20_000, sr: int = 8_000):
    """
    Three statistically independent signals:
      s1 — sine wave (musical tone)
      s2 — sawtooth wave (different harmonic content)
      s3 — band-limited noise (speech-like)
    """
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)

    s1 = np.sin(2 * np.pi * 440 * t)

    s2 = sum(np.sin(2 * np.pi * k * 330 * t) / k for k in range(1, 9))
    s2 /= np.max(np.abs(s2))

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n_samples)
    s3 = np.convolve(noise, np.ones(20) / 20, mode="same")
    s3 /= np.max(np.abs(s3))

    return np.vstack([s1, s2, s3]), sr   # (3, n_samples)


# ---------------------------------------------------------------------------
# sklearn comparison
# ---------------------------------------------------------------------------

def _compare_with_sklearn(S_true: np.ndarray, X: np.ndarray, our_eval: dict) -> None:
    """Run scikit-learn FastICA on the same data and print a comparison table."""
    try:
        from sklearn.decomposition import FastICA as SklearnFastICA
    except ImportError:
        print("  [!] scikit-learn not installed. Run: pip install scikit-learn\n")
        return

    print(" Running scikit-learn FastICA…")
    sk_ica = SklearnFastICA(
        n_components=S_true.shape[0],
        fun="logcosh",      # logcosh' = tanh — same non-linearity as our g(u)
        max_iter=1000,
        tol=1e-7,
        random_state=42,
    )
    S_sk = sk_ica.fit_transform(X.T).T   # sklearn expects (n_samples, n_features)
    sk_eval = evaluate(S_true, S_sk)

    n   = S_true.shape[0]
    C   = 9    # column width per source
    LW  = 20   # label column width
    W   = 2 + LW + n * C + 10

    def _v(x: float) -> str:
        return f"{x:+{C-1}.1f}" if np.isfinite(x) else f"{'∞':>{C-1}}"

    def _mean(vals: list) -> str:
        finite = [v for v in vals if np.isfinite(v)]
        m = float(np.mean(finite)) if finite else float("inf")
        return f"{m:+7.2f}" if np.isfinite(m) else "      ∞"

    hdr = "".join(f"{'Src '+str(i+1):>{C}}" for i in range(n))
    print("\n" + "=" * W)
    print("  Custom FastICA vs. scikit-learn FastICA")
    print("=" * W)
    print(f"  {'Method':<{LW}}{hdr}   {'Mean':>7}")
    print("-" * W)

    pairs = [
        ("SNR (dB)", "snr_db"),
        ("SIR (dB)", "sir_db"),
        ("SAR (dB)", "sar_db"),
    ]
    for idx, (metric, key) in enumerate(pairs):
        if idx:
            print("-" * W)
        o = our_eval[key]
        s = sk_eval[key]
        print(f"  {metric+' custom':<{LW}}" + "".join(f"{_v(v):>{C}}" for v in o) + f"   {_mean(o):>7}")
        print(f"  {metric+' sklearn':<{LW}}" + "".join(f"{_v(v):>{C}}" for v in s) + f"   {_mean(s):>7}")

    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 52)
    print(" BSS / ICA — Synthetic Signal Demo")
    print("=" * 52)

    S, sr = make_sources()
    print(f" Sources shape : {S.shape} (n_sources × n_samples)")

    X, A = simulate_mixing(S, n_mics=3, random_state=7)
    print(f" Mixed shape   : {X.shape}")
    print(f"\n Mixing matrix A:\n{A}\n")

    # --- Run ICA pipeline ----------------------------------------------------
    print(" Running ICA pipeline…")
    result = blind_source_separation(X, max_iter=1000, tol=1e-7, random_state=42)

    print(f"\n Eigenvalues D = {np.round(result['D'], 4)}")
    print(f" Whitened data cov ≈ I:\n"
          f"{np.round(result['Z'] @ result['Z'].T / result['Z'].shape[1], 3)}")
    print(f"\n W_total (≈ A⁻¹):\n{np.round(result['W_total'], 4)}")

    # --- Evaluate ------------------------------------------------------------
    eval_results = evaluate(S, result["S"])
    print_report(eval_results, result["converged"])

    # --- Optional sklearn comparison -----------------------------------------
    if args.compare:
        _compare_with_sklearn(S, X, eval_results)

    S_hat_aligned = eval_results["S_aligned"]

    # --- Save audio ----------------------------------------------------------
    print(" Saving audio files…")
    save_audio(str(OUTPUT_DIR / "source_1_sine.wav"),     S[0], sr)
    save_audio(str(OUTPUT_DIR / "source_2_sawtooth.wav"), S[1], sr)
    save_audio(str(OUTPUT_DIR / "source_3_noise.wav"),    S[2], sr)
    save_audio(str(OUTPUT_DIR / "mixed_mic1.wav"), X[0], sr)
    save_audio(str(OUTPUT_DIR / "mixed_mic2.wav"), X[1], sr)
    save_audio(str(OUTPUT_DIR / "mixed_mic3.wav"), X[2], sr)
    save_audio(str(OUTPUT_DIR / "separated_ic1.wav"), S_hat_aligned[0], sr)
    save_audio(str(OUTPUT_DIR / "separated_ic2.wav"), S_hat_aligned[1], sr)
    save_audio(str(OUTPUT_DIR / "separated_ic3.wav"), S_hat_aligned[2], sr)
    print(f" Audio saved to: {OUTPUT_DIR}/")

    # --- Save plots ----------------------------------------------------------
    print("\n Saving figures…")
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

    print(f" Figures saved to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
