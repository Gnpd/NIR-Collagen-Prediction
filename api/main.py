"""
Collagen yield prediction API.

Exposes four endpoints:
  POST /predict/plsr      — PLSR model (JSON: single spectrum or batch)
  POST /predict/rf        — Random Forest model (JSON: single spectrum or batch)
  POST /predict/plsr/csv  — PLSR model (CSV file upload, one spectrum per row)
  POST /predict/rf/csv    — Random Forest model (CSV file upload, one spectrum per row)

Both models were serialized with openmodels and expect raw pseudo-absorbance
[log(1/R)] for 2151 wavelengths (350–2500 nm, 1 nm step) as input.
"""

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from openmodels import SerializationManager, SklearnSerializer
from chemotools.utils.discovery import all_estimators
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WAVELENGTHS = 2151  # 350–2500 nm inclusive


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SpectrumRequest(BaseModel):
    """
    One or more NIR spectra supplied as a JSON body.

    ``reflectance`` accepts:
    - A **single spectrum**: a flat list of 2151 floats.
    - **Multiple spectra**: a list of lists, each inner list being 2151 floats.

    By default the values are interpreted as pseudo-absorbance [log(1/R)].
    Set ``raw_reflectance=true`` to supply raw reflectance (0–1); the API
    will apply the log(1/R) conversion automatically.
    """

    reflectance: list[float] | list[list[float]]
    raw_reflectance: bool = False

    @field_validator("reflectance")
    @classmethod
    def check_shape(cls, v: list[float] | list[list[float]]) -> list[float] | list[list[float]]:
        if v and isinstance(v[0], list):
            for i, row in enumerate(v):
                if len(row) != N_WAVELENGTHS:
                    raise ValueError(
                        f"Spectrum at index {i} has {len(row)} values; "
                        f"expected {N_WAVELENGTHS} (350–2500 nm)."
                    )
        else:
            if len(v) != N_WAVELENGTHS:
                raise ValueError(
                    f"Expected {N_WAVELENGTHS} wavelength values (350–2500 nm), "
                    f"got {len(v)}."
                )
        return v

    def to_array(self) -> np.ndarray:
        """Return a 2-D (n_samples, n_wavelengths) float64 array."""
        arr = np.array(self.reflectance, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr


class PredictionResponse(BaseModel):
    sample: str | None = None
    collagen_yield_pct: float
    model: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_samples: int
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Model loading at startup
# ---------------------------------------------------------------------------

MODELS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # sys._MEIPASS is set by PyInstaller when running as a frozen executable;
    # model JSON files are bundled there via --add-data.
    base_dir = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
    serializer = SklearnSerializer(custom_estimators=all_estimators)
    manager = SerializationManager(serializer)

    for key, filename in [("plsr", "plsr_2045.json"), ("rf", "rf_2045.json")]:
        path = base_dir / filename
        if not path.exists():
            raise RuntimeError(f"Model file not found: {path}")
        MODELS[key] = manager.load(str(path), format_name="json")

    yield

    MODELS.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NIR Collagen Prediction API",
    description="""
Predict collagen yield (%) from near-infrared reflectance spectra of archaeological bone.

Based on [Ryder et al. (2026) — *Journal of Archaeological Science*, 185, 106448](https://doi.org/10.1016/j.jas.2025.106448).
The full Python replication of the analytical workflow is documented in the
[NIR Collagen Prediction notebook](https://github.com/Gnpd/NIR-Collagen-Prediction/blob/main/NIR_Collagen_Prediction.ipynb).

## Install

Pre-built binaries for Windows, macOS and Linux are available on the
[GitHub Releases page](https://github.com/Gnpd/NIR-Collagen-Prediction/releases).

## Models

Two pipelines are available. Both apply **Savitzky-Golay 2nd-derivative smoothing** and
restrict the spectrum to the **2030–2060 nm** range (31 features), which targets the 2045 nm
collagen absorption band while avoiding PVA consolidant peaks at 2135, 2250 and 2296 nm.

| Endpoint prefix | Algorithm | Components / trees |
|---|---|---|
| `/predict/plsr` | PLS Regression | 1 component |
| `/predict/rf` | Random Forest | 500 trees |

## Input

Each spectrum must cover **350–2500 nm at 1 nm steps (2151 values)**.

Values can be supplied as:
- **Pseudo-absorbance** `log(1/R)` — default, expected by the model pipeline.
- **Raw reflectance** (0–1) — set `raw_reflectance=true` and the API converts automatically.

## Endpoints

| Method | Path | Input format |
|---|---|---|
| POST | `/predict/plsr` | JSON body — one spectrum or a list of spectra |
| POST | `/predict/rf` | JSON body — one spectrum or a list of spectra |
| POST | `/predict/plsr/csv` | CSV file upload — one spectrum per row |
| POST | `/predict/rf/csv` | CSV file upload — one spectrum per row |

All endpoints return a `BatchPredictionResponse` with one prediction per spectrum.

---

For feedback, questions, or collaborations → [let's connect](https://www.linkedin.com/in/alejandro-gutierrez-99404123/)
""",
    version="1.0.0",
    lifespan=lifespan,
)


def _to_absorbance(
    X: np.ndarray, sample_names: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Convert raw reflectance to pseudo-absorbance log(1/R).

    Values <= 0 are clipped to 1e-6 (instrument noise at spectrum edges).
    Affected samples are reported as warnings rather than raising an error.
    """
    warnings: list[str] = []
    mask = X <= 0
    if np.any(mask):
        bad_rows = np.where(mask.any(axis=1))[0]
        for i in bad_rows:
            name = sample_names[i] if sample_names else f"index {i}"
            n_bad = int(mask[i].sum())
            warnings.append(
                f"Sample '{name}': {n_bad} reflectance value(s) <= 0 clipped to 1e-6 "
                f"(likely instrument noise at spectrum edges)."
            )
        X = np.where(mask, 1e-6, X)
    return np.log(1.0 / X), warnings


def _parse_csv(content: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse uploaded CSV into a 2-D float array (samples × wavelengths) and sample names.

    Expected format:
    - Row 0: header with wavenumber labels (first cell is ignored).
    - Column 0: sample names.
    - Remaining cells: spectral intensities.
    """
    try:
        df = pd.read_csv(io.BytesIO(content), header=0, index_col=0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.shape[1] != N_WAVELENGTHS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {N_WAVELENGTHS} spectral columns (one per wavelength) after "
                f"removing the sample-name column, got {df.shape[1]}."
            ),
        )
    try:
        print(df.to_numpy(dtype=np.float64))
        return df.to_numpy(dtype=np.float64), list(df.index.astype(str))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Non-numeric value in CSV: {exc}") from exc


def _predict_batch(
    model_key: str,
    X: np.ndarray,
    sample_names: list[str] | None = None,
    warnings: list[str] | None = None,
) -> BatchPredictionResponse:
    model = MODELS.get(model_key)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_key}' not loaded.")

    try:
        preds = model.predict(X).ravel()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    names = sample_names or [None] * len(preds)
    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(sample=name, collagen_yield_pct=float(p), model=model_key)
            for name, p in zip(names, preds)
        ],
        n_samples=len(preds),
        warnings=warnings or [],
    )


@app.get("/", include_in_schema=False)
def health():
    return {
        "status": "ok",
        "message": "NIR Collagen Prediction API is running. Visit /docs for the interactive documentation.",
    }


@app.post(
    "/predict/plsr",
    response_model=BatchPredictionResponse,
    summary="Predict collagen yield — PLSR (JSON)",
    description="""
Predict collagen yield (%) from NIR spectra using the PLSR model.

**Pipeline:** Savitzky-Golay 2nd-derivative → RangeCut 2030–2060 nm → PLSRegression (1 component).

**`reflectance` field** — two formats accepted:
- A flat list of 2151 floats for a single spectrum: `[v1, v2, ..., v2151]`
- A list of lists for multiple spectra: `[[v1, ...], [v1, ...], ...]`

Each spectrum must cover 350–2500 nm at 1 nm steps (2151 values).

**`raw_reflectance` field:**
- `false` (default): values are already pseudo-absorbance `log(1/R)`.
- `true`: values are raw reflectance (0–1); `log(1/R)` is applied automatically.
  Values ≤ 0 are clipped to 1e-6 and reported in `warnings`.
""",
)
def predict_plsr(body: SpectrumRequest) -> BatchPredictionResponse:
    X = body.to_array()
    warnings = []
    if body.raw_reflectance:
        X, warnings = _to_absorbance(X)
    return _predict_batch("plsr", X, warnings=warnings)


@app.post(
    "/predict/rf",
    response_model=BatchPredictionResponse,
    summary="Predict collagen yield — Random Forest (JSON)",
    description="""
Predict collagen yield (%) from NIR spectra using the Random Forest model.

**Pipeline:** Savitzky-Golay 2nd-derivative → RangeCut 2030–2060 nm → RandomForestRegressor (500 trees).

**`reflectance` field** — two formats accepted:
- A flat list of 2151 floats for a single spectrum: `[v1, v2, ..., v2151]`
- A list of lists for multiple spectra: `[[v1, ...], [v1, ...], ...]`

Each spectrum must cover 350–2500 nm at 1 nm steps (2151 values).

**`raw_reflectance` field:**
- `false` (default): values are already pseudo-absorbance `log(1/R)`.
- `true`: values are raw reflectance (0–1); `log(1/R)` is applied automatically.
  Values ≤ 0 are clipped to 1e-6 and reported in `warnings`.
""",
)
def predict_rf(body: SpectrumRequest) -> BatchPredictionResponse:
    X = body.to_array()
    warnings = []
    if body.raw_reflectance:
        X, warnings = _to_absorbance(X)
    return _predict_batch("rf", X, warnings=warnings)


@app.post(
    "/predict/plsr/csv",
    response_model=BatchPredictionResponse,
    summary="Batch prediction from CSV — PLSR",
    description="""
Predict collagen yield (%) from NIR spectra supplied as a CSV file, using the PLSR model.

**Pipeline:** Savitzky-Golay 2nd-derivative → RangeCut 2030–2060 nm → PLSRegression (1 component).

**CSV format:**
- **Row 0 (header):** wavenumber labels (first cell is ignored).
- **Column 0:** sample names — included in the response as `sample`.
- **Remaining cells:** spectral intensities, exactly **2151 columns** (wavelengths 350–2500 nm, 1 nm step).

**`raw_reflectance` query parameter:**
- `false` (default): values are already pseudo-absorbance `log(1/R)`.
- `true`: values are raw reflectance (0–1); `log(1/R)` is applied automatically.
  Values ≤ 0 are clipped to 1e-6 and reported in `warnings`.

Returns one prediction per spectrum, the total sample count, and any clipping warnings.
""",
)
async def predict_plsr_csv(
    file: UploadFile = File(...),
    raw_reflectance: bool = False,
) -> BatchPredictionResponse:
    X, names = _parse_csv(await file.read())
    warnings = []
    if raw_reflectance:
        X, warnings = _to_absorbance(X, names)
    return _predict_batch("plsr", X, names, warnings)


@app.post(
    "/predict/rf/csv",
    response_model=BatchPredictionResponse,
    summary="Batch prediction from CSV — Random Forest",
    description="""
Predict collagen yield (%) from NIR spectra supplied as a CSV file, using the Random Forest model.

**Pipeline:** Savitzky-Golay 2nd-derivative → RangeCut 2030–2060 nm → RandomForestRegressor (500 trees).

**CSV format:**
- **Row 0 (header):** wavenumber labels (first cell is ignored).
- **Column 0:** sample names — included in the response as `sample`.
- **Remaining cells:** spectral intensities, exactly **2151 columns** (wavelengths 350–2500 nm, 1 nm step).

**`raw_reflectance` query parameter:**
- `false` (default): values are already pseudo-absorbance `log(1/R)`.
- `true`: values are raw reflectance (0–1); `log(1/R)` is applied automatically.
  Values ≤ 0 are clipped to 1e-6 and reported in `warnings`.

Returns one prediction per spectrum, the total sample count, and any clipping warnings.
""",
)
async def predict_rf_csv(
    file: UploadFile = File(...),
    raw_reflectance: bool = False,
) -> BatchPredictionResponse:
    X, names = _parse_csv(await file.read())
    warnings = []
    if raw_reflectance:
        X, warnings = _to_absorbance(X, names)
    return _predict_batch("rf", X, names, warnings)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
