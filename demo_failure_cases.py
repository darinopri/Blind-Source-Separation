"""
demo_failure_cases.py — ICA failure case analysis
==================================================

Three experiments documenting when FastICA degrades:

  1. Near-Gaussian sources  — ICA cannot separate Gaussian signals;
                              SNR drops as more sources approach Gaussian.
  2. Ill-conditioned mixing — high cond(A) amplifies inversion errors.
  3. Noisy observations     — additive noise (X = AS + ε) limits separation.

Each experiment averages results over multiple random mixing matrices.
Figures are saved to outputs/failure_cases/.

Usage
-----
python demo_failure_cases.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ica import blind_source_separation
from metrics import evaluate

OUTPUT_DIR = Path("outputs/failure_cases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 20_000
N_TRIALS  = 5      # number of random mixing matrices averaged per condition

COLORS = {"SNR": "#1D9E75", "SIR": "#185FA5", "SAR": "#BA7517"}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_base_sources(n_samples: int, seed: int = 0) -> np.ndarray:
    """Three independent, highly non-Gaussian signals."""
    rng = np.random.default_rng(seed)
    t  = np.linspace(0, 1, n_samples, endpoint=False)
    s1 = np.sin(2 * np.pi * 440 * t)                  # sine — sub-Gaussian
    s2 = np.sign(np.sin(2 * np.pi * 220 * t))          # square wave — super-Gaussian
    s3 = rng.uniform(-1, 1, n_samples)                 # uniform — sub-Gaussian
    return np.vstack([s1, s2, s3])


def random_mix(S: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    A = rng.standard_normal((n, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A @ S, A


def run_ica(X: np.ndarray) -> np.ndarray:
    return blind_source_separation(X, max_iter=1000, tol=1e-7, random_state=42)["S"]


def avg_metrics(S: np.ndarray, S_hat: np.ndarray) -> tuple[float, float, float]:
    """Mean SNR, SIR, SAR in dB (finite values only)."""
    r = evaluate(S, S_hat)

    def _m(vals):
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else 0.0

    return _m(r["snr_db"]), _m(r["sir_db"]), _m(r["sar_db"])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_chart(labels, snrs, sirs, sars, xlabel, title, save_path):
    x, w = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, snrs, w, label="SNR", color=COLORS["SNR"], alpha=0.85)
    ax.bar(x,     sirs, w, label="SIR", color=COLORS["SIR"], alpha=0.85)
    ax.bar(x + w, sars, w, label="SAR", color=COLORS["SAR"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("dB", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="500")
    ax.legend(fontsize=9); ax.axhline(0, lw=0.5, color="#ccc")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _line_chart(labels, snrs, sirs, sars, xlabel, title, save_path):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    for vals, lbl, col in [(snrs, "SNR", COLORS["SNR"]),
                            (sirs, "SIR", COLORS["SIR"]),
                            (sars, "SAR", COLORS["SAR"])]:
        ax.plot(x, vals, marker="o", label=lbl, color=col, lw=1.8, ms=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("dB", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="500")
    ax.legend(fontsize=9); ax.axhline(0, lw=0.5, color="#ccc")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 1 — Near-Gaussian sources
# ---------------------------------------------------------------------------

def experiment_near_gaussian() -> None:
    """
    Replace 0, 1, 2, or 3 sources with standard Gaussian noise.

    ICA assumes non-Gaussian sources. As sources become more Gaussian,
    negentropy approaches zero and FastICA loses its optimisation signal.
    """
    W = 64
    print("\n" + "=" * W)
    print("  Experiment 1 — Near-Gaussian Sources")
    print("  Replacing sources with N(0,1) noise one at a time")
    print("=" * W)
    print(f"  {'# Gaussian':>12}   {'Mean SNR':>9}   {'Mean SIR':>9}   {'Mean SAR':>9}")
    print("-" * W)

    snrs, sirs, sars, labels = [], [], [], []

    for n_gauss in range(4):
        snr_list, sir_list, sar_list = [], [], []

        for trial in range(N_TRIALS):
            S = make_base_sources(N_SAMPLES, seed=trial)
            for k in range(n_gauss):
                rng = np.random.default_rng(trial * 10 + k)
                S[-(k + 1)] = rng.standard_normal(N_SAMPLES)

            X, _ = random_mix(S, seed=7 + trial)
            S_hat = run_ica(X)
            snr, sir, sar = avg_metrics(S, S_hat)
            snr_list.append(snr); sir_list.append(sir); sar_list.append(sar)

        m_snr = float(np.mean(snr_list))
        m_sir = float(np.mean(sir_list))
        m_sar = float(np.mean(sar_list))
        note  = "  ← baseline" if n_gauss == 0 else \
                "  ← ICA fails" if n_gauss == 3 else ""
        print(f"  {n_gauss:>12}   {m_snr:>+8.1f}   {m_sir:>+8.1f}   {m_sar:>+8.1f}  dB{note}")
        snrs.append(m_snr); sirs.append(m_sir); sars.append(m_sar)
        labels.append(str(n_gauss))

    print("=" * W)
    _bar_chart(labels, snrs, sirs, sars,
               xlabel="Number of Gaussian sources (out of 3)",
               title="Experiment 1 — Near-Gaussian Sources",
               save_path=str(OUTPUT_DIR / "exp1_near_gaussian.png"))


# ---------------------------------------------------------------------------
# Experiment 2 — Ill-conditioned mixing matrix
# ---------------------------------------------------------------------------

def _matrix_with_cond(n: int, target_cond: float, seed: int) -> np.ndarray:
    """n×n matrix with the given condition number, constructed via SVD."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv   = np.linspace(1.0, float(target_cond), n)
    return U @ np.diag(sv) @ V.T


def experiment_ill_conditioned() -> None:
    """
    Construct mixing matrices with increasing condition numbers under fixed noise.

    The whitening step normalises conditioning in noiseless data, so the effect
    only appears when noise is present: a high cond(A) stretches the observation
    space unevenly, amplifying noise more in poorly-mixed directions than in
    well-mixed ones. We fix the observation noise at 20 dB input SNR to reveal
    this interaction.
    """
    NOISE_SNR_DB = 20.0   # fixed noise floor that exposes conditioning effects
    W = 68
    print("\n" + "=" * W)
    print("  Experiment 2 — Ill-Conditioned Mixing Matrix")
    print(f"  Fixed observation noise at {NOISE_SNR_DB:.0f} dB input SNR")
    print("=" * W)
    print(f"  {'cond(A)':>10}   {'Mean SNR':>9}   {'Mean SIR':>9}   {'Mean SAR':>9}")
    print("-" * W)

    cond_numbers = [2, 10, 50, 200, 500]
    snrs, sirs, sars, labels = [], [], [], []
    S_base = make_base_sources(N_SAMPLES, seed=0)

    for cond in cond_numbers:
        snr_list, sir_list, sar_list = [], [], []
        for trial in range(N_TRIALS):
            A       = _matrix_with_cond(3, float(cond), seed=trial)
            X_clean = A @ S_base
            sigma   = np.sqrt(float(np.mean(X_clean ** 2)) / (10 ** (NOISE_SNR_DB / 10)))
            rng     = np.random.default_rng(trial + 100)
            X       = X_clean + rng.standard_normal(X_clean.shape) * sigma
            S_hat   = run_ica(X)
            snr, sir, sar = avg_metrics(S_base, S_hat)
            snr_list.append(snr); sir_list.append(sir); sar_list.append(sar)

        m_snr = float(np.mean(snr_list))
        m_sir = float(np.mean(sir_list))
        m_sar = float(np.mean(sar_list))
        note  = "  ← easy"      if cond ==   2 else \
                "  ← very hard" if cond == 500 else ""
        print(f"  {cond:>10}   {m_snr:>+8.1f}   {m_sir:>+8.1f}   {m_sar:>+8.1f}  dB{note}")
        snrs.append(m_snr); sirs.append(m_sir); sars.append(m_sar)
        labels.append(str(cond))

    print("=" * W)
    _line_chart(labels, snrs, sirs, sars,
                xlabel="Condition number of A",
                title=f"Experiment 2 — Ill-Conditioned Mixing (input SNR = {NOISE_SNR_DB:.0f} dB)",
                save_path=str(OUTPUT_DIR / "exp2_ill_conditioned.png"))


# ---------------------------------------------------------------------------
# Experiment 3 — Noisy observations
# ---------------------------------------------------------------------------

def experiment_noisy_observations() -> None:
    """
    Add white Gaussian noise at controlled input-SNR levels.

    The noise floor in X = AS + ε sets an upper bound on output quality:
    ICA cannot recover signal energy that is buried in noise.
    """
    W = 70
    print("\n" + "=" * W)
    print("  Experiment 3 — Noisy Observations  (X = AS + ε)")
    print("  Sigma chosen to match target input SNR per level")
    print("=" * W)
    print(f"  {'Input SNR':>10}   {'Mean SNR':>9}   {'Mean SIR':>9}   {'Mean SAR':>9}")
    print("-" * W)

    input_snrs = [float("inf"), 30, 20, 10, 5]
    snrs, sirs, sars, labels = [], [], [], []

    S_base = make_base_sources(N_SAMPLES, seed=0)
    X_clean, _ = random_mix(S_base, seed=7)
    sig_power   = float(np.mean(X_clean ** 2))

    for in_snr in input_snrs:
        snr_list, sir_list, sar_list = [], [], []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(trial)
            if np.isinf(in_snr):
                X = X_clean.copy()
            else:
                sigma = np.sqrt(sig_power / (10 ** (in_snr / 10)))
                X = X_clean + rng.standard_normal(X_clean.shape) * sigma

            S_hat = run_ica(X)
            snr, sir, sar = avg_metrics(S_base, S_hat)
            snr_list.append(snr); sir_list.append(sir); sar_list.append(sar)

        m_snr = float(np.mean(snr_list))
        m_sir = float(np.mean(sir_list))
        m_sar = float(np.mean(sar_list))
        label = "∞" if np.isinf(in_snr) else f"{int(in_snr)} dB"
        note  = "  ← no noise" if np.isinf(in_snr) else ""
        print(f"  {label:>10}   {m_snr:>+8.1f}   {m_sir:>+8.1f}   {m_sar:>+8.1f}  dB{note}")
        snrs.append(m_snr); sirs.append(m_sir); sars.append(m_sar)
        labels.append(label)

    print("=" * W)
    _line_chart(labels, snrs, sirs, sars,
                xlabel="Input SNR (dB)",
                title="Experiment 3 — Noisy Observations (X = AS + ε)",
                save_path=str(OUTPUT_DIR / "exp3_noisy_observations.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  BSS / ICA — Failure Case Analysis")
    print(f"  Each result averaged over {N_TRIALS} random mixing matrices")
    print("=" * 64)

    experiment_near_gaussian()
    experiment_ill_conditioned()
    experiment_noisy_observations()

    print(f"\n Figures saved to: {OUTPUT_DIR}/")
    print("   exp1_near_gaussian.png")
    print("   exp2_ill_conditioned.png")
    print("   exp3_noisy_observations.png\n")


if __name__ == "__main__":
    main()
