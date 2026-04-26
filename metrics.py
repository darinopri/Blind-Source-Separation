"""
metrics.py — Evaluation metrics for ICA source separation
==========================================================
"""

import numpy as np
from itertools import permutations


def signal_to_noise_ratio(original: np.ndarray, recovered: np.ndarray) -> float:
    """
    SNR in dB between original and recovered signal.

        SNR = 10 log10( ||s||² / ||s − ŝ||² )
    """
    power_signal = np.mean(original ** 2)
    power_noise  = np.mean((original - recovered) ** 2)
    if power_noise < 1e-12:
        return float("inf")
    return 10.0 * np.log10(power_signal / power_noise)


def normalise_sources(S: np.ndarray) -> np.ndarray:
    """Normalise each row to unit variance."""
    std = S.std(axis=1, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    return S / std


def align_sources(
    S_true: np.ndarray,
    S_hat: np.ndarray,
) -> tuple[np.ndarray, list[int], list[float]]:
    """
    Find the permutation (and sign flip) that best matches S_hat to S_true.
    Uses absolute correlation as the matching criterion.
    """
    n = S_true.shape[0]
    S_true_n = normalise_sources(S_true)
    S_hat_n  = normalise_sources(S_hat)

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
    for i in range(n):
        c = np.corrcoef(S_true_n[i], S_aligned[i])[0, 1]
        if c < 0:
            S_aligned[i] = -S_aligned[i]

    corrs = [corr_mat[i, best_perm[i]] for i in range(n)]
    return S_aligned, best_perm, corrs


def sir_sar(
    S_true: np.ndarray,
    S_hat_aligned: np.ndarray,
) -> tuple[list[float], list[float]]:
    """
    SIR and SAR per source via the BSS_EVAL decomposition.

    For each aligned pair (s_j, ŝ_j), decompose ŝ_j as:

        ŝ_j = s_target + e_interf + e_artif

    where:
        s_target         — projection of ŝ_j onto the matched source s_j
        s_target+e_interf — projection of ŝ_j onto all true sources
        e_artif          — residual not explained by any true source

    SIR = 10 log10( ||s_target||²          / ||e_interf||² )
    SAR = 10 log10( ||s_target + e_interf||² / ||e_artif||²  )

    Parameters
    ----------
    S_true        : (n, T)  normalised original sources
    S_hat_aligned : (n, T)  recovered sources already aligned to S_true

    Returns
    -------
    sir_values, sar_values : lists of floats in dB (inf = perfect)
    """
    n = S_true.shape[0]
    G = S_true @ S_true.T   # Gram matrix (n × n)

    sir_values: list[float] = []
    sar_values: list[float] = []

    for j in range(n):
        s_j     = S_true[j]
        s_hat_j = S_hat_aligned[j]

        # Projection of ŝ_j onto the matched source s_j only
        alpha    = np.dot(s_hat_j, s_j) / (np.dot(s_j, s_j) + 1e-12)
        s_target = alpha * s_j

        # Projection of ŝ_j onto ALL true sources — solve G c = S_true ŝ_j
        rhs = S_true @ s_hat_j
        try:
            coeffs = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            coeffs, *_ = np.linalg.lstsq(G, rhs, rcond=None)
        s_proj = S_true.T @ coeffs   # best approximation using all sources

        e_interf = s_proj - s_target
        e_artif  = s_hat_j - s_proj

        p_target = np.mean(s_target ** 2)
        p_interf = np.mean(e_interf ** 2)
        p_proj   = np.mean(s_proj   ** 2)
        p_artif  = np.mean(e_artif  ** 2)

        sir_values.append(
            float("inf") if p_interf < 1e-12
            else 10.0 * np.log10(p_target / p_interf)
        )
        sar_values.append(
            float("inf") if p_artif < 1e-12
            else 10.0 * np.log10(p_proj / p_artif)
        )

    return sir_values, sar_values


def evaluate(
    S_true: np.ndarray,
    S_hat: np.ndarray,
) -> dict:
    """
    Full evaluation: align, then compute SNR, SIR, and SAR per source.

    Returns dict with keys:
        S_aligned, permutation, correlations,
        snr_db, mean_snr_db,
        sir_db, mean_sir_db,
        sar_db, mean_sar_db
    """
    S_true_n = normalise_sources(S_true)
    S_hat_n  = normalise_sources(S_hat)

    S_aligned, perm, corrs = align_sources(S_true_n, S_hat_n)

    n = S_true_n.shape[0]
    snr_values              = [signal_to_noise_ratio(S_true_n[i], S_aligned[i]) for i in range(n)]
    sir_values, sar_values  = sir_sar(S_true_n, S_aligned)

    def _mean(vals: list) -> float:
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("inf")

    return dict(
        S_aligned=S_aligned,
        permutation=perm,
        correlations=corrs,
        snr_db=snr_values,
        mean_snr_db=_mean(snr_values),
        sir_db=sir_values,
        mean_sir_db=_mean(sir_values),
        sar_db=sar_values,
        mean_sar_db=_mean(sar_values),
    )


def print_report(results: dict, converged: bool) -> None:
    """Pretty-print the full evaluation report."""
    W = 72

    def _fmt(v: float) -> str:
        return f"{v:+.1f} dB" if np.isfinite(v) else "    ∞ dB"

    print("\n" + "=" * W)
    print("  BSS / ICA — Evaluation Report")
    print("=" * W)
    print(f"  FastICA converged : {'yes' if converged else 'NO (increase max_iter)'}")
    print(f"  Best permutation  : {results['permutation']}")
    print()
    for i, corr in enumerate(results["correlations"]):
        print(
            f"  Source {i+1} :  |corr| = {corr:.4f}"
            f"   SNR = {_fmt(results['snr_db'][i])}"
            f"   SIR = {_fmt(results['sir_db'][i])}"
            f"   SAR = {_fmt(results['sar_db'][i])}"
        )
    print()
    print(
        f"  Mean  SNR : {results['mean_snr_db']:+.2f} dB"
        f"   SIR : {results['mean_sir_db']:+.2f} dB"
        f"   SAR : {results['mean_sar_db']:+.2f} dB"
    )
    print("=" * W + "\n")
