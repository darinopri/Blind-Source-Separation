"""
demo_audio.py — BSS/ICA on a real audio file or fully synthetic sources
=======================================================================

If an audio file is provided, its first channel is used as source 1 and
additional synthetic sources are added to build a meaningful multi-source
scenario.  If no audio file is given, all sources are generated internally
(identical approach to demo_synthetic.py but with more variety).

Usage
-----
# No audio file needed — runs out of the box:
python demo_audio.py

# With your own WAV file:
python demo_audio.py path/to/your_audio.wav

# With extra options:
python demo_audio.py path/to/your_audio.wav --n-mics 3 --seed 7
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ica import blind_source_separation
from audio_io import load_audio, save_audio, simulate_mixing
from metrics import evaluate, print_report
from visualise import plot_pipeline, plot_covariance, plot_eigenspectrum, plot_waveform

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Blind Source Separation demo on an audio file (or fully synthetic)."
    )
    p.add_argument(
        "audio_file",
        nargs="?",          # makes the argument optional
        default=None,
        help="Path to a .wav file (optional). If omitted, all sources are generated internally.",
    )
    p.add_argument("--n-mics",     type=int,   default=2,
                   help="Number of virtual microphones (default 2)")
    p.add_argument("--n-sources",  type=int,   default=None,
                   help="Number of sources to separate (default: same as n_mics)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed for mixing matrix and synthetic sources")
    p.add_argument("--max-seconds",type=float, default=5.0,
                   help="Truncate audio to this many seconds (default 5)")
    p.add_argument("--output-dir", type=str,   default="outputs/audio",
                   help="Directory for output files (default outputs/audio/)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

SR_DEFAULT = 8_000   # sample rate used for purely synthetic runs

def make_sine(n_samples: int, sr: int, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

def make_sawtooth(n_samples: int, sr: int, freq: float = 330.0) -> np.ndarray:
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    s = sum(np.sin(2 * np.pi * k * freq * t) / k for k in range(1, 9))
    s /= np.max(np.abs(s) + 1e-10)
    return s.astype(np.float32)

def make_noise(n_samples: int, seed: int = 0, window: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n_samples)
    s = np.convolve(raw, np.ones(window) / window, mode="same")
    s /= np.max(np.abs(s) + 1e-10)
    return s.astype(np.float32)

def make_chirp(n_samples: int, sr: int,
               f0: float = 200.0, f1: float = 1200.0) -> np.ndarray:
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    s = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / (n_samples / sr)) * t)
    return s.astype(np.float32)

SYNTH_BUILDERS = [make_sawtooth, make_chirp, make_noise]

def build_synthetic_source(idx: int, n_samples: int, sr: int, seed: int) -> np.ndarray:
    if idx == 0:
        return make_sine(n_samples, sr)
    fn = SYNTH_BUILDERS[(idx - 1) % len(SYNTH_BUILDERS)]
    try:
        return fn(n_samples, sr)
    except TypeError:
        return fn(n_samples, seed=seed + idx)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 54)
    print(" BSS / ICA — Audio Demo")
    print("=" * 54)

    n_mics    = args.n_mics
    n_sources = args.n_sources or n_mics

    # --- Decide source matrix ------------------------------------------------
    if args.audio_file is not None:
        # --- Load WAV ---------------------------------------------------------
        wav_path = Path(args.audio_file)
        if not wav_path.exists():
            print(f"\n  ERROR: file not found: {wav_path}")
            print("  Run without an argument to use fully synthetic sources:\n"
                  "      python demo_audio.py\n")
            sys.exit(1)

        data, sr = load_audio(str(wav_path))
        print(f"\n  File        : {wav_path}")
        print(f"  Channels    : {data.shape[0]}")
        print(f"  Sample rate : {sr} Hz")
        print(f"  Duration    : {data.shape[1]/sr:.2f} s")

        max_samples = int(args.max_seconds * sr)
        audio_mono  = data[0, :max_samples]
        n_samples   = len(audio_mono)
        print(f"  Using first {n_samples} samples ({n_samples/sr:.2f} s)")

        # source 0 = loaded audio; remaining = synthetic
        sources_list = [audio_mono]
        for i in range(1, n_sources):
            sources_list.append(build_synthetic_source(i, n_samples, sr, args.seed))
        label = wav_path.name

    else:
        # --- Fully synthetic --------------------------------------------------
        sr        = SR_DEFAULT
        n_samples = int(args.max_seconds * sr)
        print(f"\n  No audio file provided — generating {n_sources} synthetic sources.")
        print(f"  Sample rate : {sr} Hz  |  Length : {n_samples} samples ({args.max_seconds:.1f} s)")

        sources_list = [
            build_synthetic_source(i, n_samples, sr, args.seed)
            for i in range(n_sources)
        ]
        label = "synthetic"

    S = np.vstack(sources_list[:n_sources])   # (n_sources, n_samples)
    print(f"\n  Source matrix S : {S.shape}")

    # --- Mix -----------------------------------------------------------------
    X, A = simulate_mixing(S, n_mics=n_mics, random_state=args.seed)
    print(f"  Mixed matrix X  : {X.shape}")
    print(f"\n  Mixing matrix A:\n{np.round(A, 3)}\n")

    save_audio(str(output_dir / "source_original.wav"), S[0], sr)
    for i in range(n_mics):
        save_audio(str(output_dir / f"mixed_mic{i+1}.wav"), X[i], sr)
    print(f"  Saved mixed signals  → {output_dir}/mixed_mic*.wav")

    # --- ICA pipeline --------------------------------------------------------
    print("\n  Running whitening + FastICA …")
    result = blind_source_separation(
        X, n_components=n_sources,
        max_iter=1000, tol=1e-7, random_state=args.seed,
    )
    print(f"  Converged   : {result['converged']}")
    print(f"  Eigenvalues : {np.round(result['D'], 4)}")
    cov_z = result["Z"] @ result["Z"].T / result["Z"].shape[1]
    print(f"  Whitened cov ≈ I:\n{np.round(cov_z, 3)}")

    # --- Evaluate ------------------------------------------------------------
    eval_results = evaluate(S, result["S"])
    print_report(eval_results, result["converged"])
    S_aligned = eval_results["S_aligned"]

    for i in range(S_aligned.shape[0]):
        save_audio(str(output_dir / f"separated_ic{i+1}.wav"), S_aligned[i], sr)
    print(f"  Saved separated components → {output_dir}/separated_ic*.wav")

    # --- Plots ---------------------------------------------------------------
    print("\n  Saving figures …")

    fig1 = plot_pipeline(
        S, X, S_aligned, sr=sr,
        title=f"ICA Pipeline — {label}",
        save_path=str(output_dir / "pipeline_waveforms.png"),
    )
    plt.close(fig1)

    fig2 = plot_covariance(
        result["C"],
        save_path=str(output_dir / "covariance_matrix.png"),
    )
    plt.close(fig2)

    fig3 = plot_eigenspectrum(
        result["D"],
        save_path=str(output_dir / "eigenspectrum.png"),
    )
    plt.close(fig3)

    fig4 = plot_waveform(
        S_aligned[0], sr=sr,
        title=f"Recovered component 1  (SNR = {eval_results['snr_db'][0]:.1f} dB)",
        save_path=str(output_dir / "recovered_ic1_waveform.png"),
    )
    plt.close(fig4)

    print(f"\n  All outputs in: {output_dir}/")
    for name in [
        "pipeline_waveforms.png", "covariance_matrix.png",
        "eigenspectrum.png",       "recovered_ic1_waveform.png",
        "source_original.wav",     "mixed_mic*.wav",
        "separated_ic*.wav",
    ]:
        print(f"    {name}")
    print()


if __name__ == "__main__":
    main()
    p = argparse.ArgumentParser(
        description="Blind Source Separation demo on an audio file."
    )
    p.add_argument("audio_file", help="Path to a .wav file (e.g. trumpet.wav)")
    p.add_argument("--n-mics", type=int, default=2,
                   help="Number of virtual microphones / mixed channels (default 2)")
    p.add_argument("--n-sources", type=int, default=None,
                   help="Number of sources to separate (default: same as n_mics)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-seconds", type=float, default=5.0,
                   help="Truncate audio to this many seconds (default 5)")
    p.add_argument("--output-dir", type=str, default="outputs/audio",
                   help="Directory for output files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_source(n_samples: int, sr: int, kind: str, seed: int = 0) -> np.ndarray:
    """Generate a synthetic independent source for mixing."""
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    if kind == "sawtooth":
        s = sum(np.sin(2 * np.pi * k * 330 * t) / k for k in range(1, 9))
        s /= np.max(np.abs(s) + 1e-10)
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(n_samples)
        w = 30
        s = np.convolve(raw, np.ones(w) / w, mode="same")
        s /= np.max(np.abs(s) + 1e-10)
    elif kind == "chirp":
        f0, f1 = 200, 1200
        s = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / (n_samples / sr)) * t)
    else:
        rng = np.random.default_rng(seed)
        s = rng.standard_normal(n_samples)
        s /= np.max(np.abs(s) + 1e-10)
    return s.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load audio ----------------------------------------------------------
    print("=" * 54)
    print("  BSS / ICA — Audio File Demo")
    print("=" * 54)

    data, sr = load_audio(args.audio_file)
    print(f"\n  File      : {args.audio_file}")
    print(f"  Channels  : {data.shape[0]}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration  : {data.shape[1]/sr:.2f} s")

    # Truncate
    max_samples = int(args.max_seconds * sr)
    data = data[:, :max_samples]

    # Use first channel as the "main" audio source
    audio_mono = data[0]
    n_samples = len(audio_mono)
    print(f"  Using first {n_samples} samples ({n_samples/sr:.2f} s)")

    # --- Build source matrix S -----------------------------------------------
    # We treat the loaded audio as source 1, add synthetic sources to fill
    # the required number for a meaningful separation demonstration.
    synth_kinds = ["sawtooth", "noise", "chirp", "random"]
    n_mics = args.n_mics
    n_sources = args.n_sources or n_mics

    sources_list = [audio_mono]
    for i in range(n_sources - 1):
        kind = synth_kinds[i % len(synth_kinds)]
        sources_list.append(make_synthetic_source(n_samples, sr, kind, seed=args.seed + i))

    S = np.vstack(sources_list[:n_sources])          # (n_sources, n_samples)
    print(f"\n  Source matrix S : {S.shape}")
    print(f"  (source 0 = audio, sources 1..{n_sources-1} = synthetic)")

    # --- Mix -----------------------------------------------------------------
    X, A = simulate_mixing(S, n_mics=n_mics, random_state=args.seed)
    print(f"  Mixed matrix  X : {X.shape}")
    print(f"\n  Mixing matrix A:\n{np.round(A, 3)}\n")

    # Save original audio for reference
    save_audio(str(output_dir / "source_original.wav"), audio_mono, sr)

    # Save mixed signals
    for i in range(n_mics):
        save_audio(str(output_dir / f"mixed_mic{i+1}.wav"), X[i], sr)
    print(f"  Saved mixed signals → {output_dir}/mixed_mic*.wav")

    # --- ICA pipeline --------------------------------------------------------
    print("\n  Running whitening + FastICA…")
    result = blind_source_separation(
        X,
        n_components=n_sources,
        max_iter=1000,
        tol=1e-7,
        random_state=args.seed,
    )

    print(f"  Converged  : {result['converged']}")
    print(f"  Eigenvalues: {np.round(result['D'], 4)}")
    print(f"  Whitened cov ≈ I:\n{np.round(result['Z'] @ result['Z'].T / result['Z'].shape[1], 3)}")

    S_hat = result["S"]

    # --- Evaluate ------------------------------------------------------------
    eval_results = evaluate(S, S_hat)
    print_report(eval_results, result["converged"])

    S_aligned = eval_results["S_aligned"]

    # --- Save separated audio ------------------------------------------------
    for i in range(S_aligned.shape[0]):
        out_path = str(output_dir / f"separated_ic{i+1}.wav")
        save_audio(out_path, S_aligned[i], sr)
    print(f"  Saved separated components → {output_dir}/separated_ic*.wav")

    # --- Plots ---------------------------------------------------------------
    print("\n  Saving figures…")

    fig1 = plot_pipeline(
        S, X, S_aligned, sr=sr,
        title=f"ICA Pipeline — {Path(args.audio_file).name}",
        save_path=str(output_dir / "pipeline_waveforms.png"),
    )
    plt.close(fig1)

    fig2 = plot_covariance(
        result["C"],
        save_path=str(output_dir / "covariance_matrix.png"),
    )
    plt.close(fig2)

    fig3 = plot_eigenspectrum(
        result["D"],
        save_path=str(output_dir / "eigenspectrum.png"),
    )
    plt.close(fig3)

    # Plot the recovered component that best matches the original audio
    best_ic = eval_results["permutation"][0]
    fig4 = plot_waveform(
        S_aligned[0], sr=sr,
        title=f"Recovered component 1 (SNR = {eval_results['snr_db'][0]:.1f} dB)",
        save_path=str(output_dir / "recovered_ic1_waveform.png"),
    )
    plt.close(fig4)

    print(f"\n  All outputs in: {output_dir}/")
    print("    pipeline_waveforms.png")
    print("    covariance_matrix.png")
    print("    eigenspectrum.png")
    print("    recovered_ic1_waveform.png")
    print("    source_original.wav")
    print("    mixed_mic*.wav")
    print("    separated_ic*.wav")
    print()


if __name__ == "__main__":
    main()
