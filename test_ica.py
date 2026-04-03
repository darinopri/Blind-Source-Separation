"""
test_ica.py — Unit tests for the BSS/ICA pipeline
==================================================

Run with:
    python -m pytest test_ica.py -v
or:
    python test_ica.py
"""

import numpy as np
import sys
import traceback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_sources(n=3, n_samples=10_000, seed=0):
    """Three independent signals: sine, sawtooth, noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_samples, endpoint=False)
    s1 = np.sin(2 * np.pi * 440 * t)
    s2 = np.sign(np.sin(2 * np.pi * 220 * t))
    s3 = rng.uniform(-1, 1, n_samples)
    return np.vstack([s1, s2, s3][:n])


def random_mixing(S, seed=1):
    """Random full-rank mixing matrix."""
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    A = rng.standard_normal((n, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A @ S, A


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def test_centering():
    from ica import center
    X = np.array([[1., 2., 3.], [4., 5., 6.]])
    Xc, m = center(X)
    assert np.allclose(Xc.mean(axis=1), 0, atol=1e-10), "Mean not zero after centering"
    assert np.allclose(m, [2., 5.]), "Wrong mean returned"
    print(PASS, "center() — zero mean and correct mean vector")


def test_covariance_shape():
    from ica import center, compute_covariance
    X, _ = center(generate_sources())
    C = compute_covariance(X)
    assert C.shape == (3, 3), f"Expected (3,3), got {C.shape}"
    assert np.allclose(C, C.T, atol=1e-10), "Covariance matrix not symmetric"
    print(PASS, "compute_covariance() — correct shape and symmetry")


def test_eigendecompose_reconstruction():
    from ica import center, compute_covariance, eigendecompose
    X, _ = center(generate_sources())
    C = compute_covariance(X)
    E, D = eigendecompose(C)
    # Reconstruct: C ≈ E @ diag(D) @ E^T
    C_reconstructed = E @ np.diag(D) @ E.T
    assert np.allclose(C, C_reconstructed, atol=1e-8), "Eigendecomposition does not reconstruct C"
    assert np.all(D >= 0), "Negative eigenvalues found"
    print(PASS, "eigendecompose() — C = E D Eᵀ reconstruction and non-negative D")


def test_whitening_identity_covariance():
    from ica import center, whiten
    X, _ = center(generate_sources())
    Z, E, D, W_white = whiten(X)
    n = Z.shape[1]
    C_white = (Z @ Z.T) / (n - 1)
    assert np.allclose(C_white, np.eye(3), atol=1e-6), \
        f"Whitened covariance not identity:\n{C_white}"
    print(PASS, "whiten() — covariance(Z) ≈ I")


def test_whitening_shape():
    from ica import center, whiten
    X, _ = center(generate_sources(n_samples=5000))
    Z, E, D, W_white = whiten(X)
    assert Z.shape == X.shape, f"Shape mismatch: Z={Z.shape}, X={X.shape}"
    assert E.shape == (3, 3)
    assert D.shape == (3,)
    assert W_white.shape == (3, 3)
    print(PASS, "whiten() — output shapes correct")


def test_fastica_convergence():
    from ica import center, whiten, fastica
    S = generate_sources()
    X, _ = random_mixing(S)
    X, _ = center(X)
    Z, *_ = whiten(X)
    W, S_hat, converged = fastica(Z, max_iter=500, tol=1e-6, random_state=0)
    assert converged, "FastICA did not converge in 500 iterations"
    print(PASS, "fastica() — converges in 500 iterations")


def test_fastica_orthogonality():
    from ica import center, whiten, fastica
    S = generate_sources()
    X, _ = random_mixing(S)
    X, _ = center(X)
    Z, *_ = whiten(X)
    W, S_hat, _ = fastica(Z, max_iter=500, random_state=0)
    WWT = W @ W.T
    assert np.allclose(WWT, np.eye(W.shape[0]), atol=1e-5), \
        f"W not orthogonal:\n{WWT}"
    print(PASS, "fastica() — W is orthogonal (W Wᵀ ≈ I)")


def test_full_pipeline_snr():
    """SNR > 10 dB indicates successful separation."""
    from ica import blind_source_separation
    from metrics import evaluate
    S = generate_sources(n_samples=20_000)
    X, A = random_mixing(S, seed=5)
    result = blind_source_separation(X, max_iter=1000, tol=1e-8, random_state=0)
    eval_results = evaluate(S, result["S"])
    mean_snr = eval_results["mean_snr_db"]
    assert mean_snr > 10.0, f"Mean SNR too low: {mean_snr:.2f} dB (expected > 10)"
    print(PASS, f"full pipeline — mean SNR = {mean_snr:.1f} dB  (> 10 dB threshold)")


def test_w_total_approx_a_inverse():
    """W_total should approximate A^{-1} (up to permutation/scaling)."""
    from ica import blind_source_separation
    S = generate_sources(n_samples=20_000)
    X, A = random_mixing(S, seed=3)
    result = blind_source_separation(X, max_iter=1000, random_state=0)
    W = result["W_total"]
    # W @ A should be close to a scaled permutation matrix
    WA = W @ A
    # Each row should have exactly one dominant element
    for i, row in enumerate(WA):
        rel = np.abs(row) / (np.max(np.abs(row)) + 1e-10)
        n_dominant = np.sum(rel > 0.8)
        assert n_dominant == 1, \
            f"Row {i} of W@A has {n_dominant} dominant elements (expected 1):\n{row}"
    print(PASS, "W_total @ A — permutation-scaled identity (each row has one dominant entry)")


def test_audio_io_roundtrip(tmp_path=None):
    """Save and reload a WAV file, check data is preserved."""
    import tempfile, os
    from audio_io import save_audio, load_audio
    rng = np.random.default_rng(7)
    data = rng.uniform(-1, 1, (2, 8000)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.wav")
        save_audio(path, data, sr=8000)
        loaded, sr = load_audio(path)
    assert sr == 8000, f"Wrong SR: {sr}"
    assert loaded.shape == data.shape, f"Shape mismatch: {loaded.shape} vs {data.shape}"
    assert np.allclose(data, loaded, atol=1e-3), "Round-trip data mismatch (>1e-3)"
    print(PASS, "audio_io — WAV save/load roundtrip (2-channel, 16-bit PCM)")


def test_evaluate_permutation():
    """evaluate() should correctly identify a permuted, sign-flipped version."""
    from metrics import evaluate
    rng = np.random.default_rng(0)
    S = rng.standard_normal((3, 10_000))
    # Shuffle + negate
    S_hat = np.vstack([-S[2], S[0], S[1]])
    res = evaluate(S, S_hat)
    # Correlations should be near 1 after alignment
    for i, corr in enumerate(res["correlations"]):
        assert corr > 0.99, f"Source {i}: correlation {corr:.4f} < 0.99 after alignment"
    print(PASS, "metrics.evaluate() — correct alignment and sign correction")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_centering,
    test_covariance_shape,
    test_eigendecompose_reconstruction,
    test_whitening_identity_covariance,
    test_whitening_shape,
    test_fastica_convergence,
    test_fastica_orthogonality,
    test_full_pipeline_snr,
    test_w_total_approx_a_inverse,
    test_audio_io_roundtrip,
    test_evaluate_permutation,
]


def run_tests():
    print("\n" + "=" * 54)
    print("  BSS / ICA — Test Suite")
    print("=" * 54)
    passed = 0
    failed = 0
    for test in ALL_TESTS:
        try:
            test()
            passed += 1
        except Exception as e:
            print(FAIL, test.__name__)
            traceback.print_exc()
            failed += 1
    print("=" * 54)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 54 + "\n")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
