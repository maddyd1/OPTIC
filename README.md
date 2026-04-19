# 🎯 OPTIC — Ogden Parameter Tuning via Inverse Calibration

OPTIC is a locally-hosted, AI-assisted engineering tool that automates inverse calibration of hyperelastic material models from physical test data. Upload a force-displacement CSV from a compression test, and OPTIC returns the optimal N=3 Ogden parameters ready to drop into FEA software — no manual curve-fitting, no slow simulation loops.

All computation and AI analysis runs fully offline. No data leaves your machine.

---

## The Problem It Solves

Calibrating Ogden hyperelastic parameters for 3D-printed metal lattice structures (LPBF) traditionally requires either:
- Running dozens of expensive FEA simulations to manually bracket parameters, or
- Relying on curve-fitting tools with no physical insight into what's driving the result

OPTIC replaces that workflow with a Gaussian Process surrogate model trained on 28 LPBF simulation datasets, wrapped in a clean Streamlit interface with local LLM interpretation.

---

## How It Works

1. **Feature Extraction** — Parses the uploaded CSV and extracts a 9-feature mechanical fingerprint from the force-displacement curve (peak force, initial stiffness, energy absorbed, nonlinearity ratio, power law exponent, and quartile forces)
2. **GP Inverse Calibration** — A trained Gaussian Process surrogate maps the fingerprint back to the 6 Ogden parameters (Mu1, A1, Mu2, A2, Mu3, A3) via a 30-restart `scipy` optimization loop weighted by per-feature LOO accuracy
3. **Honest Confidence Scoring** — Reports mean leave-one-out cross-validation R² on the 4 most critical features — true out-of-sample accuracy, not training score
4. **Local LLM Interpretation** — Sends results to a locally-hosted model via LM Studio to generate a plain-English structural analysis highlighting which Ogden term dominates and flagging any physical anomalies
5. **Training Space Overlay** — Plots the uploaded curve against all 28 training datasets so you can immediately see if the test case is interpolating or extrapolating

---

## Features

- **Zero data leakage** — fully air-gapped, runs on local hardware
- **Out-of-bounds warnings** — flags parameters that fall outside the training envelope
- **Flexible CSV parsing** — handles varied column naming conventions automatically
- **Copy-ready parameter output** — formatted for direct paste into ABAQUS, ANSYS, or similar
- **Plotly dark-mode visualizations** — interactive, zoomable charts

---

## Project Structure

```
OPTIC/
├── app.py                          # Streamlit dashboard and UI
├── solver.py                       # GP surrogate model + scipy optimization engine
├── data_utils.py                   # CSV parsing and 9-feature fingerprint extraction
├── ai_engine.py                    # LM Studio API bridge (OpenAI-compatible)
├── rom_history_analysis_format.csv # 28-dataset training file (local only, gitignored)
└── Start_OPTIC.command             # macOS double-click launcher
```

---

## Prerequisites

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running locally on port `1234` with a model loaded (MoE architectures recommended — Gemma 4 27B or similar)
- Training data file: `rom_history_analysis_format.csv` placed in the project root (not included in repo — proprietary dataset)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/maddyd1/OPTIC.git
cd OPTIC

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install streamlit pandas numpy scipy scikit-learn plotly openai

# Launch
streamlit run app.py
```

Or on macOS, double-click `Start_OPTIC.command` after the first install.

---

## Usage

1. Start LM Studio and load a model
2. Launch OPTIC via `streamlit run app.py`
3. Upload a force-displacement CSV (columns: `Displacement` in mm, `Force` in N)
4. OPTIC extracts the mechanical fingerprint, runs inverse calibration, and returns Ogden parameters with confidence score
5. Copy parameters directly into your FEA material definition

---

## Tech Stack

| Component | Library |
|---|---|
| Dashboard | Streamlit |
| Surrogate model | scikit-learn `GaussianProcessRegressor` |
| Optimization | scipy `minimize` (L-BFGS-B, 30 restarts) |
| Feature extraction | NumPy, SciPy `curve_fit` |
| Visualization | Plotly |
| Local LLM | LM Studio (OpenAI-compatible API) |

---

## Limitations

- Surrogate trained on 28 LPBF datasets — extrapolation beyond the training envelope degrades reliability (flagged with warnings)
- Ogden model fixed at N=3 terms
- LLM interpretation requires LM Studio running; gracefully degrades to an offline message if unavailable

---

## Author

Built by [Maddie](https://github.com/maddyd1) — proton therapy engineer and accelerator systems specialist at The Christie NHS Foundation Trust.