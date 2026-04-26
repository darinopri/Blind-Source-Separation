"""
ica.py — Blind Source Separation via Independent Component Analysis
===================================================================
Implements the full pipeline described in Script 3:

    X = A @ S          (mixing model)
    Z = D^{-1/2} E^T X  (whitening)
    S = W @ Z          (FastICA unmixing)

All mathematics follow the presentation exactly:
  - Covariance matrix  C = E[X X^T]
  - Eigendecomposition C = E D E^T
  - Whitening          Z = D^{-1/2} E^T X
  - FastICA            maximises non-Gaussianity (kurtosis / negentropy)
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Step 1 — Centering
# ---------------------------------------------------------------------------

def center(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the mean of each observed signal.

    Parameters
    ----------
    X : ndarray, shape (n_sources, n_samples)
        Observed mixed signals (rows = channels).

    Returns
    -------
    X_centered : ndarray  — zero-mean version of X
    mean_vec   : ndarray  — mean of each channel (needed to restore later)
    """
    mean_vec = X.mean(axis=1, keepdims=True)
    return X - mean_vec, mean_vec.squeeze()


# ---------------------------------------------------------------------------
# Steps 2–4 — Covariance, Eigendecomposition, and Whitening
# ---------------------------------------------------------------------------

def compute_covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix.

        C = E[X X^T]  (unbiased: divide by n-1)

    Parameters
    ----------
    X : ndarray, shape (n_sources, n_samples)  — zero-mean data

    Returns
    -------
    C : ndarray, shape (n_sources, n_sources)
    """
    n = X.shape[1]
    return (X @ X.T) / (n - 1)


def eigendecompose(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecompose the covariance matrix.

        C = E D E^T

    Uses np.linalg.eigh (symmetric matrix → real, sorted eigenvalues).

    Parameters
    ----------
    C : ndarray, shape (n, n)  — symmetric positive semi-definite

    Returns
    -------
    E : ndarray, shape (n, n)  — eigenvectors (columns)
    D : ndarray, shape (n,)    — eigenvalues (ascending)
    """
    D, E = np.linalg.eigh(C)
    # Sort descending so largest variance component comes first
    idx = np.argsort(D)[::-1]
    return E[:, idx], D[idx]


def whiten(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Whiten the data so the covariance becomes the identity matrix.

        Z = D^{-1/2} E^T X

    Parameters
    ----------
    X : ndarray, shape (n_sources, n_samples)  — zero-mean data

    Returns
    -------
    Z        : ndarray, shape (n_sources, n_samples)  — whitened data
    E        : eigenvectors of covariance
    D        : eigenvalues  of covariance
    W_white  : the whitening matrix  D^{-1/2} E^T
    """
    C = compute_covariance(X)
    E, D = eigendecompose(C)

    # Regularise to avoid division by near-zero eigenvalues
    eps = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, eps)))

    W_white = D_inv_sqrt @ E.T          # shape (n, n)
    Z = W_white @ X                     # shape (n, n_samples)
    return Z, E, D, W_white


# ---------------------------------------------------------------------------
# Step 7 — FastICA
# ---------------------------------------------------------------------------

def _g(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-linearity g and its derivative g' for the FastICA fixed-point rule.

    Uses g(u) = tanh(u), a proxy for negentropy maximisation.
    """
    t = np.tanh(u)
    return t, 1.0 - t ** 2


def _gram_schmidt(W: np.ndarray) -> np.ndarray:
    """
    Symmetric decorrelation (Gram–Schmidt orthogonalisation).

    Ensures the rows of W remain orthogonal across iterations.

        W ← (W W^T)^{-1/2} W
    """
    A = W @ W.T
    vals, vecs = np.linalg.eigh(A)
    eps = 1e-10
    A_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, eps))) @ vecs.T
    return A_inv_sqrt @ W


def fastica(
    Z: np.ndarray,
    n_components: Optional[int] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    FastICA fixed-point algorithm.

    Estimates the unmixing matrix W such that:

        S = W @ Z

    where the rows of S are maximally non-Gaussian (independent).

    Parameters
    ----------
    Z            : ndarray, shape (n, n_samples)  — whitened data
    n_components : int or None  — number of ICs to extract (default: all)
    max_iter     : int          — maximum iterations per component
    tol          : float        — convergence tolerance (change in w)
    random_state : int          — seed for reproducibility

    Returns
    -------
    W        : ndarray, shape (n_components, n)  — unmixing matrix
    S        : ndarray, shape (n_components, n_samples)  — separated signals
    converged: bool
    """
    rng = np.random.default_rng(random_state)
    n, n_samples = Z.shape

    if n_components is None:
        n_components = n

    # Initialise W randomly, then orthogonalise
    W = rng.standard_normal((n_components, n))
    W = _gram_schmidt(W)

    converged = True
    for _ in range(max_iter):
        # Fixed-point update (parallel FastICA)
        # For each row w_i: w_i ← E[Z g(w_i^T Z)] − E[g'(w_i^T Z)] w_i
        U = W @ Z                           # (n_components, n_samples)
        g_u, gp_u = _g(U)

        W_new = (g_u @ Z.T) / n_samples - gp_u.mean(axis=1, keepdims=True) * W
        W_new = _gram_schmidt(W_new)

        # Convergence: max change in absolute dot-product of rows
        delta = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
        W = W_new

        if delta < tol:
            break
    else:
        converged = False

    S = W @ Z
    return W, S, converged


# ---------------------------------------------------------------------------
# Full pipeline (X → S)
# ---------------------------------------------------------------------------

def blind_source_separation(
    X: np.ndarray,
    n_components: Optional[int] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    random_state: int = 42,
) -> dict:
    """
    Complete BSS pipeline: X → whitening → FastICA → S.

    Parameters
    ----------
    X            : ndarray, shape (n_channels, n_samples)
    n_components : int or None
    max_iter     : int
    tol          : float
    random_state : int

    Returns
    -------
    dict with keys:
        X_centered  — zero-mean observations
        C           — covariance matrix
        E           — eigenvectors
        D           — eigenvalues
        W_white     — whitening matrix
        Z           — whitened data
        W_ica       — FastICA unmixing matrix (in whitened space)
        W_total     — total unmixing matrix  W_ica @ W_white  (≈ A^{-1})
        S           — recovered sources
        converged   — FastICA convergence flag
    """
    X_c, mean_vec = center(X)
    C = compute_covariance(X_c)
    E, D = eigendecompose(C)
    Z, E, D, W_white = whiten(X_c)
    W_ica, S, converged = fastica(Z, n_components, max_iter, tol, random_state)

    W_total = W_ica @ W_white   # Full unmixing: S = W_total @ X_centered

    return dict(
        X_centered=X_c,
        C=C,
        E=E,
        D=D,
        W_white=W_white,
        Z=Z,
        W_ica=W_ica,
        W_total=W_total,
        S=S,
        converged=converged,
        mean_vec=mean_vec,
    )
