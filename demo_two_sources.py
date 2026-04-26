"""
demo_two_sources.py — the "cocktail party" problem with two real sources
========================================================================

Takes two separate WAV files (e.g. a voice and music track), artificially
mixes them as two microphone recordings, runs ICA, and saves the separated
signals.

Usage
-----
# With your own two files:
python demo_two_sources.py voice.wav music.wav

# No files? Generates two synthetic sources and demonstrates on those:
python demo_two_sources.py

Optional flags:
  --seed         Random seed for the mixing matrix (default 42)
  --max-seconds  How many seconds to use from each file (default 5)
  --output-dir   Where to save results (default outputs/two_sources/)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd

from ica import blind_source_separation
from audio_io import load_audio, save_audio
from metrics import evaluate, print_report
from visualise import plot_pipeline, plot_covariance, plot_eigenspectrum

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ICA on two real audio files — the cocktail party problem."
    )
    p.add_argument("file1", nargs="?", default=None,
                   help="First WAV file (e.g. voice.wav)")
    p.add_argument("file2", nargs="?", default=None,
                   help="Second WAV file (e.g. music.wav)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max-seconds", type=float, default=5.0)
    p.add_argument("--output-dir",  type=str,   default="outputs/two_sources")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_to(signal: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return signal
    g = gcd(sr_from, sr_to)
    return resample_poly(signal, sr_to // g, sr_from // g).astype(np.float32)


def load_mono(path: str, target_sr: int, max_samples: int) -> np.ndarray:
    """Load WAV, average to mono, resample to target_sr, clip to max_samples."""
    data, sr = load_audio(path)
    mono = data.mean(axis=0).astype(np.float32)
    mono = resample_to(mono, sr, target_sr)
    peak = np.max(np.abs(mono))
    if peak > 1e-6:
        mono /= peak
    return mono[:max_samples]


def make_sine(n: int, sr: int, freq=440.0) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def make_sawtooth(n: int, sr: int, freq=330.0) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False)
    s = sum(np.sin(2 * np.pi * k * freq * t) / k for k in range(1, 9))
    s = s / np.max(np.abs(s))
    return s.astype(np.float32)


def random_mixing_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    SR = 8_000
    max_samples = int(args.max_seconds * SR)

    print("=" * 56)
    print(" BSS / ICA — Two Sources (Cocktail Party)")
    print("=" * 56)

    # --- Load or generate sources --------------------------------------------
    if args.file1 and args.file2:
        path1, path2 = Path(args.file1), Path(args.file2)

        missing = [p for p in [path1, path2] if not p.exists()]
        if missing:
            for p in missing:
                print(f"\n  ERROR: file not found: {p}")
            print("\n  Run without arguments for a synthetic demo:")
            print("      python demo_two_sources.py\n")
            return

        print(f"\n  File 1 : {path1}")
        print(f"  File 2 : {path2}")
        print(f"  Sample rate : {SR} Hz (resampled if needed)")
        print(f"  Duration    : up to {args.max_seconds} s\n")

        s1 = load_mono(str(path1), SR, max_samples)
        s2 = load_mono(str(path2), SR, max_samples)

        n = min(len(s1), len(s2))
        s1, s2 = s1[:n], s2[:n]
        label1, label2 = path1.stem, path2.stem

    else:
        print("\n  No files provided — using synthetic signals.")
        print(f"  (440 Hz sine  +  330 Hz sawtooth)\n")
        n = max_samples
        s1 = make_sine(n, SR, freq=440.0)
        s2 = make_sawtooth(n, SR, freq=330.0)
        label1, label2 = "sine_440Hz", "sawtooth_330Hz"

    S = np.vstack([s1, s2])   # (2, n_samples) — source matrix
    print(f"  Sources S : {S.shape}  ({n} samples = {n/SR:.2f} s)")

    # --- Mix: simulate 2 microphones -----------------------------------------
    A = random_mixing_matrix(2, args.seed)
    X = A @ S                  # (2, n_samples) — what the microphones "hear"

    print(f"  Mixture X : {X.shape}")
    print(f"\n  Mixing matrix A:")
    print(f"    mic 1 = {A[0,0]:+.3f} × {label1}  {A[0,1]:+.3f} × {label2}")
    print(f"    mic 2 = {A[1,0]:+.3f} × {label1}  {A[1,1]:+.3f} × {label2}")

    save_audio(str(out / f"original_1_{label1}.wav"), s1, SR)
    save_audio(str(out / f"original_2_{label2}.wav"), s2, SR)
    save_audio(str(out / "mixed_mic1.wav"), X[0], SR)
    save_audio(str(out / "mixed_mic2.wav"), X[1], SR)
    print(f"\n  Saved originals and mixtures → {out}/")

    # --- ICA -----------------------------------------------------------------
    print("\n  Running ICA…")
    result = blind_source_separation(X, max_iter=2000, tol=1e-8,
                                     random_state=args.seed)

    print(f"  Converged   : {'yes' if result['converged'] else 'NO (increase max_iter)'}")
    print(f"  Eigenvalues : {np.round(result['D'], 4)}")

    # --- Evaluate ------------------------------------------------------------
    eval_res = evaluate(S, result["S"])
    print_report(eval_res, result["converged"])
    S_hat = eval_res["S_aligned"]

    # --- Save separated signals ----------------------------------------------
    save_audio(str(out / "separated_ic1.wav"), S_hat[0], SR)
    save_audio(str(out / "separated_ic2.wav"), S_hat[1], SR)

    # --- Plots ---------------------------------------------------------------
    print("  Saving figures…")

    fig1 = plot_pipeline(
        S, X, S_hat, sr=SR,
        title=f"ICA: {label1}  +  {label2}",
        save_path=str(out / "pipeline_waveforms.png"),
    )
    plt.close(fig1)

    fig2 = plot_covariance(
        result["C"],
        save_path=str(out / "covariance_matrix.png"),
    )
    plt.close(fig2)

    fig3 = plot_eigenspectrum(
        result["D"],
        save_path=str(out / "eigenspectrum.png"),
    )
    plt.close(fig3)

    # --- Summary -------------------------------------------------------------
    snr = eval_res["snr_db"]
    print(f"\n{'='*56}")
    print(f"  Results saved to: {out}/")
    print(f"{'='*56}")
    print(f"  original_1_{label1}.wav  — source 1 (reference)")
    print(f"  original_2_{label2}.wav  — source 2 (reference)")
    print(f"  mixed_mic1.wav            — mixture (mic 1)")
    print(f"  mixed_mic2.wav            — mixture (mic 2)")
    print(f"  separated_ic1.wav         — ICA: recovered signal 1")
    print(f"  separated_ic2.wav         — ICA: recovered signal 2")
    print()
    print("  Listen to:")
    print("    1. mixed_mic1.wav     — both sources mixed together")
    print("    2. separated_ic1.wav  — does it sound like source 1?")
    print("    3. separated_ic2.wav  — does it sound like source 2?")
    print(f"\n  SNR source 1: {snr[0]:+.1f} dB")
    print(f"  SNR source 2: {snr[1]:+.1f} dB")
    print(f"  (higher is better)\n")


if __name__ == "__main__":
    main()
