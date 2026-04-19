# 🎯 OPTIC: Ogden Parameter Tuning via Inverse Calibration

OPTIC is a local, AI-powered Streamlit dashboard designed for materials scientists and engineers. It automates the extraction of physical features from force-displacement test data and uses a Machine Learning surrogate model to find the optimal N=3 Ogden hyperelastic simulation parameters. 

All computations and AI analyses run 100% locally to ensure strict data privacy for proprietary research.

## ✨ Features

*   **Automated Feature Extraction:** Reads raw physical test CSVs and extracts a 9-feature fingerprint (Peak Force, Initial Stiffness, Energy Absorbed, Nonlinearity, etc.).
*   **Gaussian Process Surrogate Model:** Replaces slow FEA simulations with a fast `scikit-learn` GP model trained on 28 LPBF simulation datasets.
*   **Honest Confidence Scoring:** Uses Leave-One-Out (LOO) cross-validation to report true out-of-sample prediction accuracy.
*   **Local AI Engineering Insights:** Integrates with **LM Studio** to provide plain-English, localized LLM analysis of the structural parameters and warnings about physical anomalies (e.g., buckling/yielding).
*   **Interactive Visualizations:** Powered by Plotly, featuring a "Training Space Overlay" to instantly see if test data falls outside the model's training boundaries.

## 📁 Project Structure

*   `app.py`: The main Streamlit dashboard and UI.
*   `solver.py`: The math engine containing the GP surrogate model and `scipy.optimize` loop.
*   `data_utils.py`: The data pipeline for parsing CSVs and extracting the 9-point fingerprint.
*   `ai_engine.py`: The OpenAI-compatible API bridge connecting the dashboard to LM Studio.
*   `Start_OPTIC.command`: A double-click executable to launch the virtual environment and app on macOS.
*   `rom_history_analysis_format.csv`: The 28-dataset training file (must be provided locally, ignored by Git).

## 🚀 Prerequisites

1.  **Python 3.x**
2.  **LM Studio** running locally on port `1234` with a loaded model (MoE architectures recommended for performance).
3.  **Required Python Libraries:** `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `plotly`, `openai`

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/maddyd1/OPTIC.git](https://github.com/maddyd1/OPTIC.git)
   cd OPTIC
