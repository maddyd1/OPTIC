import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
 
 
def extract_curve_fingerprint(file_path):
    """
    Reads a physical test CSV and extracts a 9-feature fingerprint
    that describes the full shape of the force-deflection curve.
 
    Recognised displacement column names:
        Displacement, displacement, Deflection, deflection, Disp, disp, D, d
 
    Recognised force column names:
        Force, force, Force(N), force(n), F, f, Load, load
 
    Returns: fingerprint dict, disp array, force array.
    """
    df = pd.read_csv(file_path)
 
    # ── Column detection ──────────────────────────────────────────────────────
    DISP_NAMES  = {'displacement', 'disp', 'd', 'deflection'}
    FORCE_NAMES = {'force', 'f', 'load', 'force(n)'}
 
    col_map = {}
    for col in df.columns:
        normalised = col.strip().lower()
        if normalised in DISP_NAMES:
            col_map['Displacement'] = col
        if normalised in FORCE_NAMES:
            col_map['Force'] = col
 
    if 'Displacement' not in col_map or 'Force' not in col_map:
        raise ValueError(
            f"Could not find Displacement/Deflection and Force columns.\n"
            f"Columns found in file: {list(df.columns)}\n"
            f"Expected one of {DISP_NAMES} and one of {FORCE_NAMES}."
        )
 
    # ── Clean and sort ────────────────────────────────────────────────────────
    df['Displacement'] = pd.to_numeric(df[col_map['Displacement']], errors='coerce')
    df['Force']        = pd.to_numeric(df[col_map['Force']],        errors='coerce')
    df = df.dropna(subset=['Displacement', 'Force'])
    df = df.sort_values('Displacement').reset_index(drop=True)
 
    disp  = df['Displacement'].values
    force = df['Force'].values
    n     = len(disp)
 
    if n < 4:
        raise ValueError(
            f"Need at least 4 valid data points — only found {n}."
        )
 
    # ── Feature 1: Peak Force ─────────────────────────────────────────────────
    peak_force = float(np.max(force))
 
    # ── Feature 2: Initial Stiffness ─────────────────────────────────────────
    nonzero = np.where(disp > 0)[0]
    if len(nonzero) > 0:
        idx = nonzero[0]
        initial_stiffness = float(force[idx] / disp[idx])
    else:
        initial_stiffness = 0.0
 
    # ── Feature 3: Energy Absorbed (area under curve) ─────────────────────────
    energy_absorbed = float(np.trapezoid(force, x=disp))
 
    # ── Feature 4: Max Displacement ───────────────────────────────────────────
    max_displacement = float(np.max(disp))
 
    # ── Feature 5: Force at 25% of points ────────────────────────────────────
    q25_force = float(force[max(0, int(n * 0.25))])
 
    # ── Feature 6: Force at midpoint ─────────────────────────────────────────
    mid_force = float(force[n // 2])
 
    # ── Feature 7: Force at 75% of points ────────────────────────────────────
    q75_force = float(force[min(n - 1, int(n * 0.75))])
 
    # ── Feature 8: Nonlinearity (late slope / early slope) ───────────────────
    try:
        early_slope  = (force[2] - force[0]) / (disp[2] - disp[0]) if disp[2] != disp[0] else 1.0
        late_slope   = (force[-1] - force[-3]) / (disp[-1] - disp[-3]) if disp[-1] != disp[-3] else 1.0
        nonlinearity = float(late_slope / early_slope) if early_slope > 0 else 1.0
    except Exception:
        nonlinearity = 1.0
 
    # ── Feature 9: Power law exponent ─────────────────────────────────────────
    try:
        popt, _ = curve_fit(
            lambda x, a, b: a * np.power(x + 1e-9, b),
            disp[1:], force[1:],
            p0=[10, 1.2],
            maxfev=2000
        )
        power_exp = float(popt[1])
    except Exception:
        power_exp = 1.0
 
    # ── Package ───────────────────────────────────────────────────────────────
    fingerprint = {
        "peak_force":        round(peak_force,        4),
        "initial_stiffness": round(initial_stiffness, 4),
        "energy_absorbed":   round(energy_absorbed,   4),
        "max_displacement":  round(max_displacement,  4),
        "q25_force":         round(q25_force,         4),
        "mid_force":         round(mid_force,         4),
        "q75_force":         round(q75_force,         4),
        "nonlinearity":      round(nonlinearity,      4),
        "power_exp":         round(power_exp,         4),
    }
 
    return fingerprint, disp, force
 
 
# ── TEST BLOCK ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else "test_data.csv"
    try:
        fp, disp, force = extract_curve_fingerprint(test_file)
        print(f"SUCCESS — {len(disp)} data points read")
        print()
        for k, v in fp.items():
            print(f"  {k:<22} {v}")
    except Exception as e:
        print(f"ERROR: {e}")
 