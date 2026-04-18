import numpy as np
import pandas as pd
import os
import warnings
from scipy.optimize import curve_fit, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
 
warnings.filterwarnings("ignore", category=ConvergenceWarning)
 
# ── Training data configuration ───────────────────────────────────────────────
# Path to the 28-dataset training CSV — adjust if needed
TRAINING_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "rom_history_analysis_format.csv"
)
 
PARAMS   = ["Mu1", "A1", "Mu2", "A2", "Mu3", "A3"]
FEATURES = [
    "peak_force", "initial_stiffness", "energy_absorbed", "max_displacement",
    "q25_force",  "mid_force",         "q75_force",
    "nonlinearity", "power_exp"
]
 
# Parameter bounds observed in the training data — used for out-of-bounds warnings
TRAINING_BOUNDS = {
    "Mu1": (0.0043, 1.5),
    "A1":  (-0.1,   1.5),
    "Mu2": (0.0043, 3.0),
    "A2":  (-0.1,   1.25),
    "Mu3": (0.25,   5.0),
    "A3":  (0.0165, 5.0),
}
 
 
def _parse_training_data():
    """Parse the multi-block training CSV into datasets."""
    df_raw = pd.read_csv(TRAINING_DATA_PATH, header=None)
    df_raw.columns = ["Mu1","A1","Mu2","A2","Mu3","A3",
                      "Displacement","Force","Stress","Strain"]
    header_idx = df_raw[df_raw["Mu1"] == "Mu1"].index.tolist()
 
    datasets = []
    for i, start in enumerate(header_idx):
        end = header_idx[i+1] if i+1 < len(header_idx) else len(df_raw)
        block = df_raw.iloc[start+1:end].copy()
        block = block[
            block["Displacement"].notna() & (block["Displacement"] != "")
        ].astype(float)
        p = {k: float(block.iloc[0][k]) for k in PARAMS}
        datasets.append({
            "params": p,
            "disp":   block["Displacement"].values,
            "force":  block["Force"].values,
        })
    return datasets
 
 
def _extract_training_features(disp, force):
    """Extract the 9-feature vector from a training curve."""
    from scipy.optimize import curve_fit
    n = len(disp)
 
    nonzero = np.where(disp > 0)[0]
    if len(nonzero) > 0:
        idx = nonzero[0]
        init_stiff = float(force[idx] / disp[idx]) if disp[idx] > 0 else 0.0
    else:
        init_stiff = 0.0
 
    peak_force   = float(force.max())
    energy       = float(np.trapezoid(force, x=disp))
    max_disp     = float(disp.max())
    q25          = float(force[max(0, int(n * 0.25))])
    mid          = float(force[n // 2])
    q75          = float(force[min(n-1, int(n * 0.75))])
 
    try:
        early = (force[2] - force[0]) / (disp[2] - disp[0]) if disp[2] != disp[0] else 1.0
        late  = (force[-1] - force[-3]) / (disp[-1] - disp[-3]) if disp[-1] != disp[-3] else 1.0
        nonlin = float(late / early) if early > 0 else 1.0
    except Exception:
        nonlin = 1.0
 
    try:
        popt, _ = curve_fit(
            lambda x, a, b: a * np.power(x + 1e-9, b),
            disp[1:], force[1:], p0=[10, 1.2], maxfev=2000
        )
        power_exp = float(popt[1])
    except Exception:
        power_exp = 1.0
 
    return [peak_force, init_stiff, energy, max_disp,
            q25, mid, q75, nonlin, power_exp]
 
 
def _build_surrogate():
    """
    Train one Gaussian Process surrogate per curve feature on the 28 training datasets.
    Returns: gp_models dict, scaler_X, scaler_ys dict, params_df for bounds checking.
    """
    datasets  = _parse_training_data()
    params_df = pd.DataFrame([d["params"] for d in datasets])
    feat_rows = [_extract_training_features(d["disp"], d["force"]) for d in datasets]
    feats_df  = pd.DataFrame(feat_rows, columns=FEATURES)
 
    X = params_df[PARAMS].values
    scaler_X  = StandardScaler().fit(X)
    X_scaled  = scaler_X.transform(X)
 
    gp_models  = {}
    scaler_ys  = {}
 
    for feat in FEATURES:
        y  = feats_df[feat].values.reshape(-1, 1)
        sy = StandardScaler().fit(y)
        ys = sy.transform(y).ravel()
 
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(6),
            length_scale_bounds=(1e-2, 1e4)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=False
        )
        gp.fit(X_scaled, ys)
        gp_models[feat] = gp
        scaler_ys[feat] = sy
 
    # LOO R² for honest confidence reporting
    from sklearn.model_selection import LeaveOneOut
    loo_scores = {}
    loo = LeaveOneOut()
    for feat in FEATURES:
        y_all = scaler_ys[feat].transform(feats_df[feat].values.reshape(-1,1)).ravel()
        actuals, preds = [], []
        for tr, te in loo.split(X_scaled):
            gp_loo = GaussianProcessRegressor(
                kernel=C(1.0,(1e-3,1e3))*RBF(np.ones(6),(1e-2,1e4)),
                n_restarts_optimizer=5
            )
            gp_loo.fit(X_scaled[tr], y_all[tr])
            yp = scaler_ys[feat].inverse_transform(
                gp_loo.predict(X_scaled[te]).reshape(-1,1)
            ).ravel()[0]
            actuals.append(feats_df[feat].values[te[0]])
            preds.append(yp)
        a, p = np.array(actuals), np.array(preds)
        r2 = 1 - np.sum((a-p)**2) / np.sum((a-a.mean())**2)
        loo_scores[feat] = max(0.0, round(r2, 3))
 
    return gp_models, scaler_X, scaler_ys, params_df, loo_scores
 
 
def _predict_features(params_6d, gp_models, scaler_X, scaler_ys):
    """Use trained GPs to predict all curve features for a given parameter set."""
    x  = np.array(params_6d).reshape(1, -1)
    xs = scaler_X.transform(x)
    out = {}
    for feat in FEATURES:
        ys, std = gp_models[feat].predict(xs, return_std=True)
        y = scaler_ys[feat].inverse_transform(ys.reshape(-1,1)).ravel()[0]
        out[feat] = (float(y), float(std[0]))
    return out
 
 
def check_out_of_bounds(params_dict):
    """
    Check whether recovered parameters fall outside the training data range.
    Returns a list of warning strings (empty if all in bounds).
    """
    warnings_list = []
    for k, v in params_dict.items():
        lo, hi = TRAINING_BOUNDS[k]
        if v < lo or v > hi:
            warnings_list.append(
                f"{k} = {v:.4f} is outside training range [{lo}, {hi}] — "
                f"prediction may be unreliable"
            )
    return warnings_list
 
 
# ── Module-level surrogate (built once on import) ─────────────────────────────
_surrogate = None
 
def _get_surrogate():
    global _surrogate
    if _surrogate is None:
        _surrogate = _build_surrogate()
    return _surrogate
 
 
def run_inverse_calibration(fingerprint):
    """
    Takes a 9-feature fingerprint of a target force-deflection curve and
    finds the Ogden parameters that best reproduce it using the GP surrogate.
 
    Returns:
        optimal_params  dict  {Mu1, A1, Mu2, A2, Mu3, A3}
        loo_confidence  float  mean LOO R² across key features (0–100%)
        bounds_warnings list   any parameters outside training range
    """
    gp_models, scaler_X, scaler_ys, params_df, loo_scores = _get_surrogate()
 
    target = {
        "peak_force":        fingerprint["peak_force"],
        "initial_stiffness": fingerprint["initial_stiffness"],
        "energy_absorbed":   fingerprint["energy_absorbed"],
        "max_displacement":  fingerprint["max_displacement"],
        "q25_force":         fingerprint.get("q25_force",    fingerprint["peak_force"] * 0.25),
        "mid_force":         fingerprint.get("mid_force",    fingerprint["peak_force"] * 0.50),
        "q75_force":         fingerprint.get("q75_force",    fingerprint["peak_force"] * 0.80),
        "nonlinearity":      fingerprint.get("nonlinearity", 1.2),
        "power_exp":         fingerprint.get("power_exp",    1.05),
    }
 
    # Weight by LOO R² so unreliable features count less
    weights = {f: loo_scores.get(f, 0.5) for f in FEATURES}
 
    def objective(x):
        pred = _predict_features(x, gp_models, scaler_X, scaler_ys)
        loss = 0.0
        for feat in FEATURES:
            p_val = pred[feat][0]
            t_val = target[feat]
            scale = max(abs(t_val), 1e-6)
            loss += weights[feat] * ((p_val - t_val) / scale) ** 2
        return loss
 
    # Parameter bounds from training data with 20% headroom
    bounds = []
    for p in PARAMS:
        lo, hi = TRAINING_BOUNDS[p]
        bounds.append((lo * 0.8, hi * 1.2))
 
    best_loss   = np.inf
    best_params = None
 
    # Multiple random restarts to avoid local minima
    np.random.seed(42)
    for _ in range(30):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        if res.fun < best_loss:
            best_loss   = res.fun
            best_params = res.x
 
    optimal_parameters = {
        "Mu1": round(float(best_params[0]), 4),
        "A1":  round(float(best_params[1]), 4),
        "Mu2": round(float(best_params[2]), 4),
        "A2":  round(float(best_params[3]), 4),
        "Mu3": round(float(best_params[4]), 4),
        "A3":  round(float(best_params[5]), 4),
    }
 
    # Honest confidence = mean LOO R² on the most important features
    key_feats = ["peak_force", "initial_stiffness", "energy_absorbed", "nonlinearity"]
    loo_confidence = round(
        np.mean([loo_scores.get(f, 0) for f in key_feats]) * 100, 1
    )
 
    bounds_warnings = check_out_of_bounds(optimal_parameters)
 
    return optimal_parameters, loo_confidence, bounds_warnings
 