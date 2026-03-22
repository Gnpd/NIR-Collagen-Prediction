# NIR Spectroscopy for Collagen Quantification in Archaeological Bone

Reproduction of the analytical workflow from:

> Ryder, C. et al. (2026). *Refining near-infrared spectroscopy for collagen quantification: A new predictive model for archaeological bone.* **Journal of Archaeological Science**, 185, 106448.

The notebook implements the full pipeline — from raw reflectance spectra to collagen yield prediction — using Python open-source libraries.

---

## Contents

| File | Description | Open in Colab |
|---|---|---|
| `NIR_Collagen_Prediction.ipynb` | Main analysis notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gnpd/NIR-Collagen-Prediction/blob/main/NIR_Collagen_Prediction.ipynb) |
| `1-s2.0-S0305440325002973-mmc2.csv` | Supplementary data from the paper (176 samples, 2151 wavelengths) | |
| `plsr_2045.json` | Serialized PLSR model — SG preprocessing, 2030–2060 nm range, 1 component | |
| `rf_2045.json` | Serialized RandomForest model — SG preprocessing, 2030–2060 nm range | |

---

## Dataset

The CSV file contains spectral data for 176 bone samples:

- **140 Reference samples** — used for model calibration and validation. Already K-means balanced from an original set of 319 samples (see paper Section 2.3).
- **36 Zafarraya samples** — independent external validation set from Zafarraya Cave, Spain.

Columns:
- Sample ID, Country of Origin, Extraction Technique (ORAU / MPI)
- Collagen Yield (%)
- Reflectance at 350–2500 nm (2151 variables)

---

## Notebook Overview

The notebook reproduces the complete workflow in 13 sections:

| Section | Content |
|---|---|
| 1–2 | Imports and data loading |
| 3 | Exploratory data analysis (yield distribution, raw spectra) |
| 4 | Spectral preprocessing: reflectance → pseudo-absorbance → Savitzky-Golay 2nd derivative |
| 5 | PCA for outlier detection (full range and NIR range) |
| 6 | Stratified calibration / validation split (100 / 40 samples) |
| 7 | Helper functions |
| 8 | PLSR across 11 wavelength ranges (reproduces Table 5) |
| 9 | Random Forest regression — full NIR and 2030–2060 nm (reproduces Sections 3.1–3.2) |
| 10 | Variable importance plots |
| 11 | Combined LOO-CV models on all 140 samples |
| 12 | Zafarraya external validation with PVA consolidant detection (reproduces Tables 6–8) |
| 13 | Model persistence to JSON using OpenModels |

The **preferred model** (2030–2060 nm, 1 PLSR factor) targets the 2045 nm absorption feature associated with the 2nd overtone of C=O stretching and N-H stretching in collagen.

---

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
chemotools
openmodels
```

Install with:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install numpy pandas matplotlib scikit-learn chemotools openmodels
```

---

## Usage

1. Clone the repository.
2. Install dependencies (see above).
3. Open `NIR_Collagen_Prediction.ipynb` in Jupyter Lab or Jupyter Notebook.
4. Run all cells in order (`Kernel → Restart & Run All`).

The notebook is self-contained: all preprocessing, modelling, and evaluation steps run sequentially without external configuration.

---

## Key Results Reproduced

- The 2030–2060 nm range delivers the best balance between parsimony and accuracy (1 PLSR factor, Val R² ≈ 0.88, Val RMSE ≈ 1.78%).
- Restricting the spectral range to 2030–2060 nm avoids PVA consolidant absorption wavelengths (2135, 2250, 2296 nm), making the model robust for archaeological collections.
- Random Forest (780–2500 nm) achieves the lowest LOO-CV RMSE (≈ 1.35%) but requires the full spectral range.

---

## Reference

```bibtex
@article{ryder2026nir,
  title   = {Refining near-infrared spectroscopy for collagen quantification:
             A new predictive model for archaeological bone},
  author  = {Ryder, C. and others},
  journal = {Journal of Archaeological Science},
  volume  = {185},
  pages   = {106448},
  year    = {2026},
  doi     = {10.1016/j.jas.2025.106448}
}
```
