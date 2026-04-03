"""
metrics.py — Evaluation metrics for ICA source separation
==========================================================
"""

import numpy as np
from itertools import permutations


def signal_to_noise_ratio(original: np.ndarray, recovered: np.ndarray) -> float:
    """
    SNR in dB between the original and recovered signal.

        SNR = 10 log10( ||s||^2 / ||s - s_hat||^2 )

    A higher SNR means better recovery.
    """
    power_signal = np.mean(original ** 2)
    power_noise  = np.mean((original - recovered) ** 2)
    if power_noise < 1e-12:
        return float("inf")
    return 10.0 * np.log10(power_signal / power_noise)


def normalise_sources(S: np.ndarray) -> np.ndarray:
    """
    Normalise each source row to unit variance.
    ICA can only recover sources up to a scaling factor.
    """
    std = S.std(axis=1, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    return S / std


def align_sources(
    S_true: np.ndarray,
    S_hat: np.ndarray,
) -> tuple[np.ndarray, list[int], list[float]]:
    """
    Find the permutation (and sign flip) of recovered sources that best
    matches the original sources. ICA is permutation-ambiguous.

    Uses absolute correlation to score each permutation.

    Parameters
    ----------
    S_true : (n, n_samples)  — original sources (normalised)
    S_hat  : (n, n_samples)  — recovered sources (normalised)

    Returns
    -------
    S_aligned  : S_hat reordered to match S_true
    perm       : permutation index list
    corrs      : absolute correlation for each matched pair
    """
    n = S_true.shape[0]
    S_true_n = normalise_sources(S_true)
    S_hat_n  = normalise_sources(S_hat)

    # Correlation matrix: C[i,j] = corr(s_true_i, s_hat_j)
    corr_mat = np.abs(
        (S_true_n @ S_hat_n.T) / S_true_n.shape[1]
    )

    best_score = -1.0
    best_perm  = list(range(n))

    for perm in permutations(range(n)):
        score = sum(corr_mat[i, perm[i]] for i in range(n))
        if score > best_score:
            best_score = score
            best_perm  = list(perm)

    S_aligned = S_hat[best_perm, :]

    # Correct sign flips
    for i in range(n):
        c = np.corrcoef(S_true_n[i], S_aligned[i])[0, 1]
        if c < 0:
            S_aligned[i] = -S_aligned[i]

    corrs = [corr_mat[i, best_perm[i]] for i in range(n)]
    return S_aligned, best_perm, corrs


def evaluate(
    S_true: np.ndarray,
    S_hat: np.ndarray,
) -> dict:
    """
    Full evaluation: align, then compute per-source SNR.

    Returns dict with keys:
        S_aligned  — aligned recovered sources
        permutation
        correlations
        snr_db      — list of SNR values (dB) per source
        mean_snr_db
    """
    S_true_n = normalise_sources(S_true)
    S_hat_n  = normalise_sources(S_hat)

    S_aligned, perm, corrs = align_sources(S_true_n, S_hat_n)

    snr_values = [
        signal_to_noise_ratio(S_true_n[i], S_aligned[i])
        for i in range(S_true_n.shape[0])
    ]

    return dict(
        S_aligned=S_aligned,
        permutation=perm,
        correlations=corrs,
        snr_db=snr_values,
        mean_snr_db=float(np.mean(snr_values)),
    )


def print_report(results: dict, converged: bool) -> None:
    """Pretty-print the evaluation report to stdout."""
    print("\n" + "=" * 52)
    print("  BSS / ICA — Evaluation Report")
    print("=" * 52)
    print(f"  FastICA converged : {'yes' if converged else 'NO (increase max_iter)'}")
    print(f"  Best permutation  : {results['permutation']}")
    print()
    for i, (corr, snr) in enumerate(zip(results["correlations"], results["snr_db"])):
        snr_str = f"{snr:+.1f} dB" if snr != float("inf") else "∞ dB (perfect)"
        print(f"  Source {i+1}:  |correlation| = {corr:.4f}   SNR = {snr_str}")
    print()
    print(f"  Mean SNR          : {results['mean_snr_db']:+.2f} dB")
    print("=" * 52 + "\n")
