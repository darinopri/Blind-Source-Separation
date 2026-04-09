# Blind Source Separation — ICA Project

Implementation of Blind Source Separation using Independent Component Analysis (ICA), built from scratch with NumPy/SciPy for a Linear Algebra course project.

------------------------------------------------------------------------

## Installation

Clone the repository and install dependencies:

``` bash
git clone https://github.com/darinopri/Blind-Source-Separation.git
cd Blind-Source-Separation
pip install -r requirements.txt
```

Dependencies (also listed in `requirements.txt`):

-   `numpy >= 1.24`
-   `scipy >= 1.10`
-   `matplotlib >= 3.7`
-   `scikit-learn >= 1.3`

------------------------------------------------------------------------

## Project structure

```         
Blind-Source-Separation/
├── ica.py              # Core ICA pipeline (centering, whitening, FastICA)
├── audio_io.py         # WAV loading / saving / mixing simulation
├── visualise.py        # Matplotlib figures (waveforms, covariance, etc.)
├── metrics.py          # SNR, alignment, evaluation report
├── demo_synthetic.py   # Demo on generated signals (no audio file needed)
├── demo_audio.py       # Demo on a WAV file (generates one if none is given)
├── test_ica.py         # Full test suite
├── requirements.txt    # Python dependencies
└── README.md
```

------------------------------------------------------------------------

## The mathematics (pipeline)

### 1. Mixing model

```         
X = A S
```

-   **X** — observed signals, shape `(n_mics, n_samples)`
-   **A** — unknown mixing matrix
-   **S** — unknown original sources

### 2. Goal

Find **W** such that:

```         
S = W X,    W ≈ A⁻¹
```

### 3. Whitening

Compute the covariance matrix:

```         
C = E[X Xᵀ]
```

Eigendecompose it:

```         
C = E D Eᵀ
```

Apply the whitening transform:

```         
Z = D^{-1/2} Eᵀ X
```

After whitening, `cov(Z) = I`.

### 4. FastICA

Find **W** that maximises statistical independence (non-Gaussianity) of `S = W Z`. Uses the fixed-point update with `g(u) = tanh(u)` as a negentropy proxy, followed by symmetric orthogonalisation.

------------------------------------------------------------------------

## Quick start

### Run the synthetic demo (no audio file needed)

``` bash
python demo_synthetic.py
```

Generates three independent signals (sine, sawtooth, noise), mixes them with a random matrix, runs ICA, and saves plots to `outputs/synthetic/`.

### Run the audio demo

Without a WAV file — the script generates all sources internally:

``` bash
python demo_audio.py
```

With your own WAV file:

``` bash
python demo_audio.py path/to/your_audio.wav
```

Optional flags:

```         
--n-mics       Number of virtual microphones (default 2)
--n-sources    Number of sources to separate (default = n_mics)
--seed         Random seed for the mixing matrix
--max-seconds  Truncate audio to this length (default 5 s)
--output-dir   Where to save results (default outputs/audio/)
```

Example:

``` bash
python demo_audio.py path/to/your_audio.wav --n-mics 3 --max-seconds 5 --seed 7
```

### Run the tests

``` bash
python test_ica.py
```

------------------------------------------------------------------------

## Output files (after running demo_audio.py)

| File                         | Description                              |
|------------------------------|------------------------------------------|
| `source_original.wav`        | Original audio (reference)               |
| `mixed_mic1.wav` …           | Simulated microphone recordings          |
| `separated_ic1.wav` …        | ICA-recovered independent components     |
| `pipeline_waveforms.png`     | Sources / mixed / recovered side-by-side |
| `covariance_matrix.png`      | Heatmap of C = E[X Xᵀ]                   |
| `eigenspectrum.png`          | Bar chart of eigenvalues                 |
| `recovered_ic1_waveform.png` | Best recovered component waveform        |

------------------------------------------------------------------------

## Evaluation metrics

-   **Absolute correlation** — `|corr(s_true, s_hat)|` after optimal alignment
-   **SNR (dB)** — `10 log10(||s||² / ||s − ŝ||²)` per source

ICA recovers sources up to **permutation** and **sign flip** — both are corrected automatically in `metrics.evaluate()`.
