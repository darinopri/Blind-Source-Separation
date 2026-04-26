# Blind Source Separation — ICA Project

Implementation of Blind Source Separation using Independent Component Analysis (ICA), built from scratch with NumPy/SciPy for a Linear Algebra course project.

------------------------------------------------------------------------

## Installation

```bash
git clone https://github.com/darinopri/Blind-Source-Separation.git
cd Blind-Source-Separation
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7`
- `scikit-learn >= 1.3`

------------------------------------------------------------------------

## Project structure

```
Blind-Source-Separation/
├── ica.py               # Core ICA pipeline (centering, whitening, FastICA)
├── audio_io.py          # WAV loading / saving / mixing simulation
├── visualise.py         # Matplotlib figures (waveforms, covariance, etc.)
├── metrics.py           # SNR, SIR, SAR, alignment, evaluation report
├── demo_synthetic.py    # Demo on generated signals; --compare for sklearn
├── demo_audio.py        # Demo on a WAV file (generates sources if none given)
├── demo_two_sources.py  # Cocktail-party demo with two real WAV files
├── demo_failure_cases.py# Three failure-case experiments (Gaussian / cond / noise)
├── test_ica.py          # Full test suite (11 tests, all passing)
├── requirements.txt     # Python dependencies
└── README.md
```

------------------------------------------------------------------------

## The mathematics (pipeline)

### 1. Mixing model

```
X = A S
```

- **X** — observed signals, shape `(n_mics, n_samples)`
- **A** — unknown mixing matrix
- **S** — unknown original sources

### 2. Goal

Find **W** such that:

```
S = W X,    W ≈ A⁻¹
```

### 3. Whitening

Compute the sample covariance matrix:

```
C = E[X Xᵀ]
```

Eigendecompose it (`C = E D Eᵀ`) and apply the whitening transform:

```
Z = D^{-1/2} Eᵀ X
```

After whitening, `cov(Z) = I`, constraining the FastICA search to orthogonal matrices.

### 4. FastICA fixed-point iteration

Find **W** that maximises negentropy (non-Gaussianity) of `S = W Z`.  
Uses the fixed-point update with `g(u) = tanh(u)`, followed by symmetric
orthogonalisation after each iteration.

------------------------------------------------------------------------

## Quick start

### Synthetic demo (no audio file needed)

```bash
python demo_synthetic.py
```

Generates three independent signals (sine, sawtooth, band-limited noise),
mixes them with a random matrix, runs ICA, and saves plots + audio to
`outputs/synthetic/`.

### Synthetic demo with scikit-learn comparison

```bash
python demo_synthetic.py --compare
```

Runs both the custom FastICA and scikit-learn's `FastICA` on the same data
and prints a side-by-side table of SNR, SIR, and SAR for each source.

### Audio demo (one WAV file)

Without a file — generates all sources synthetically:

```bash
python demo_audio.py
```

With your own WAV file:

```bash
python demo_audio.py path/to/your_audio.wav --n-mics 3 --max-seconds 5
```

Optional flags:

```
--n-mics       Number of virtual microphones (default 2)
--n-sources    Number of sources to separate (default = n_mics)
--seed         Random seed for the mixing matrix
--max-seconds  Truncate audio to this length (default 5 s)
--output-dir   Where to save results (default outputs/audio/)
```

### Cocktail-party demo (two real WAV files)

```bash
python demo_two_sources.py voice.wav music.wav
```

No files? Falls back to two synthetic signals:

```bash
python demo_two_sources.py
```

Optional flags:

```
--seed         Random seed (default 42)
--max-seconds  Seconds to use (default 5)
--output-dir   Output directory (default outputs/two_sources/)
```

### Failure case analysis

```bash
python demo_failure_cases.py
```

Runs three experiments that document ICA's limits:

1. **Near-Gaussian sources** — replace sources with `N(0,1)` one at a time; SNR drops from ~55 dB (all non-Gaussian) to ~6 dB (all Gaussian).
2. **Ill-conditioned mixing** — vary `cond(A)` from 2 to 500 under fixed 20 dB input noise; SNR drops from ~19 dB to ~7 dB.
3. **Noisy observations** — add white noise at input SNR levels of ∞, 30, 20, 10, 5 dB; output SNR tracks the noise floor.

Figures are saved to `outputs/failure_cases/`.

### Run the tests

```bash
python test_ica.py
```

------------------------------------------------------------------------

## Output files

| File | Description |
|---|---|
| `source_*.wav` | Original source signals (reference) |
| `mixed_mic*.wav` | Simulated microphone recordings |
| `separated_ic*.wav` | ICA-recovered independent components |
| `pipeline_waveforms.png` | Sources / mixed / recovered side-by-side |
| `covariance_matrix.png` | Heatmap of C = E[X Xᵀ] |
| `eigenspectrum.png` | Bar chart of eigenvalues |
| `whitening_scatter.png` | Data distribution before and after whitening |

------------------------------------------------------------------------

## Evaluation metrics

All metrics are computed after optimal **permutation alignment** and
**sign correction** (both ambiguities inherent to ICA).

### SNR — Signal-to-Noise Ratio

```
SNR = 10 log10( ||s||² / ||s − ŝ||² )
```

Measures overall reconstruction fidelity.

### SIR — Signal-to-Interference Ratio

```
SIR = 10 log10( ||s_target||² / ||e_interf||² )
```

Quantifies leakage from other sources into the recovered component.
`s_target` is the projection of `ŝ` onto the matched true source;
`e_interf` is the contribution from all remaining sources.

### SAR — Signal-to-Artifacts Ratio

```
SAR = 10 log10( ||s_target + e_interf||² / ||e_artif||² )
```

Measures algorithmic distortions — the part of `ŝ` that cannot be
explained by any linear combination of true sources.

Higher values are better for all three metrics. Typical results on the
synthetic demo (3 sources, random mixing):

| Metric | Mean (dB) |
|--------|-----------|
| SNR    | ~28       |
| SIR    | ~33       |
| SAR    | ~29       |
