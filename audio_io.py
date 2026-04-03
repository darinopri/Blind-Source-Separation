"""
audio_io.py — Audio loading and saving utilities
=================================================
Uses only stdlib (wave) and scipy.io.wavfile so no extra dependencies
beyond what is already available.
"""

import wave
import struct
import numpy as np
from pathlib import Path
from scipy.io import wavfile


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return float32 data normalised to [-1, 1].

    Parameters
    ----------
    path : str  — path to .wav file

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
           Rows = channels. Mono files return shape (1, n_samples).
    sr   : int  — sample rate in Hz
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError("Only WAV files are supported (use scipy or convert first).")

    sr, data = wavfile.read(str(path))

    # Normalise integer types to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Ensure shape (n_channels, n_samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]   # mono → (1, n_samples)
    else:
        data = data.T                # (n_samples, n_ch) → (n_ch, n_samples)

    return data, int(sr)


def save_audio(path: str, data: np.ndarray, sr: int) -> None:
    """
    Save float32 audio data as a 16-bit PCM WAV file.

    Parameters
    ----------
    path : str
    data : ndarray, shape (n_channels, n_samples)  or (n_samples,)
    sr   : int  — sample rate
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Clip and convert to int16
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    if pcm.shape[0] == 1:
        out = pcm[0]                  # mono
    else:
        out = pcm.T                   # (n_samples, n_ch)

    wavfile.write(str(path), sr, out)


def simulate_mixing(
    sources: np.ndarray,
    n_mics: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a random mixing matrix A and return mixed signals X = A @ S.

    Parameters
    ----------
    sources      : ndarray, shape (n_sources, n_samples)
    n_mics       : int  — number of microphones (default: same as n_sources)
    random_state : int

    Returns
    -------
    X : ndarray, shape (n_mics, n_samples)  — mixed signals
    A : ndarray, shape (n_mics, n_sources)  — mixing matrix
    """
    n_sources, n_samples = sources.shape
    if n_mics is None:
        n_mics = n_sources

    rng = np.random.default_rng(random_state)
    A = rng.standard_normal((n_mics, n_sources))
    # Normalise columns so no single source dominates
    A /= np.linalg.norm(A, axis=0, keepdims=True)

    X = A @ sources
    return X, A
