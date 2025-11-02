# -*- coding: utf-8 -*-
"""
Transfer Curve Analysis for FETs — header-configurable revision
- User can specify header names for the first row (case-insensitive)
- Only DrainI (I_d) and GateV (V_g) are required; DrainV (V_d) and GateI (I_g) are optional
- Defaults (when no user input): DrainI, DrainV, GateV
- Robust auto-detection supports variants like "DrainI_A", "Gate V (V)", etc.
- Safe plotting and table writes

Tested with: pandas, numpy, matplotlib, xlrd, scipy
"""

###########################      user input      ############################

DATA_FILE: str = 'TR.xls'   # Path to the data file
SERIES_NAME: str = 'test'        # Series name for saving results
DEVICE_TYPE: str = 'p'               # 'n' for n-type FETs, 'p' for p-type FETs

# === NEW: Header configuration (case-insensitive) ===
# Fill these with the EXACT header texts in row 1 if your file differs.
# Leave as None to auto-detect. (Defaults: DrainI, DrainV, GateV)
CUSTOM_HEADERS = {
    'Id': 'DrainI'
,   # e.g., 'DrainI_A' or 'DrainI'
    'Vg': 'GateV'
,   # e.g., 'GateV' or 'Gate V (V)'
    'Vd': None,   # e.g., 'DrainV' (optional)
    'Ig': None,   # e.g., 'GateI'  (optional)
}

CONDITION_DATA: bool = False         # True if using a conditions file
if CONDITION_DATA:
    CONDITION_FILE: str = 'Conditions.xls'
else:
    CH_WIDTH: float = 1.0            # um
    CH_LENGTH: float = 1.0           # um
    GI_THICK: float = 0.09           # um

GI_EPS_R: float = 3.9                # 3.9 for SiO2

ID_AT_VTH: float = 1e-8              # A, current threshold for Vth (current-based)
LINEAR_WINDOW_FOR_VTH: float = 5.0   # V, window size when picking linear segment for interpolation Vth
PLOTTING: bool = True                # Save per-dataset plots

if PLOTTING:
    APPLY_ABSOLUTE: bool = True
    MANUAL_PLOT: bool = False
    if MANUAL_PLOT:
        VG_MIN: int = -20
        VG_MAX: int = +20
        ID_MIN: float = 1e-12
        ID_MAX: float = 1e-3
        FIGURE_SIZE_H: int = 8
        FIGURE_SIZE_V: int = 6

ANALYZE_PARAMETERS: bool = True     # Turn on to compute Vth/SS/mobility

############# Advanced control (change only if you know) #############
ANALYZE_VTH: bool = False
ANALYZE_SS: bool = False
ANALYZE_MOBILITY: bool = False
DENOISE_CURRENT: bool = False
REMOVE_OUTLIERS: bool = False
DIFFERENTIAL_ROUGHNESS: int = 2      # (1~4) recommended
LOG_THRESHOLD_FIND_SS: float = 1.2   # (1.1~2) recommended

if ANALYZE_PARAMETERS:
    ANALYZE_VTH = True
    ANALYZE_SS = True
    ANALYZE_MOBILITY = True
    DENOISE_CURRENT = False
    REMOVE_OUTLIERS = True
    DIFFERENTIAL_ROUGHNESS = 2
    LOG_THRESHOLD_FIND_SS = 1.2

################################################################################################
from datetime import datetime
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
from scipy.constants import epsilon_0
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import traceback
from typing import List, Union, Tuple, Optional, Dict

# ---- session setup ----
start_time: str = datetime.now().strftime("%H-%M-%S_%Y%m%d")
plt.rcParams['figure.dpi'] = 120
DEBUG: bool = False

# Output directory
if True:
    base_dir = f'./result/TransferCurve/{SERIES_NAME}_{DATA_FILE}_{start_time}/'
    directory: str = './Debug/' if DEBUG else base_dir
    os.makedirs(directory, exist_ok=True)
    # backup current code for reproducibility
    try:
        if not DEBUG:
            src_path = __file__
            backup_path: str = os.path.join(directory, 'code_backup.py')
            with open(src_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    except Exception as _:
        pass

# ---- helpers ----

def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())


def safe_min_positive(*arrs: np.ndarray, default: float = 1e-12) -> float:
    vals = []
    for a in arrs:
        a = np.asarray(a)
        a = a[np.isfinite(a)]
        a = a[a > 0]
        if a.size:
            vals.append(a.min())
    return min(vals) if vals else default


def lowpass_window(data: np.ndarray, window_size: int = 15) -> np.ndarray:
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def check_monotonic(array: np.ndarray) -> int:
    differences: np.ndarray = np.diff(array)
    is_increasing: bool = np.all(differences >= 0)
    is_decreasing: bool = np.all(differences <= 0)
    if is_increasing:
        return 1
    elif is_decreasing:
        return -1
    else:
        return 0


def get_segment_indices(arr: np.ndarray, n: int = -1) -> np.ndarray:
    if len(arr) < 2:
        return np.array([0]) if len(arr) > 0 else np.array([])

    trends: np.ndarray = np.sign(np.diff(arr))
    indices: List[np.ndarray] = []
    start: int = 0
    i: int = 0

    while i < len(trends):
        if trends[i] == 0:
            i += 1
            start = i
            continue
        direction: int = trends[i]
        j: int = i + 1
        while j < len(trends) and trends[j] == direction:
            j += 1
        end: int = j
        indices.append(np.arange(start, end + 1))
        i = j
        start = j

    if n == -1:
        return max(indices, key=len) if indices else np.array([])
    elif 0 <= n < len(indices):
        return indices[n]
    else:
        raise ValueError(f"n too big. (0 <= n <= {len(indices)-1})")


def interpolate_large_gaps(data: Union[List[float], np.ndarray], step: float, donot_interpolate: bool = False) -> Union[List[float], np.ndarray]:
    data = np.array(data)
    if step <= 0:
        donot_interpolate = True
    if donot_interpolate:
        return data.tolist()
    interpolated_data: List[float] = [data[0]]
    for i in range(1, len(data)):
        gap: float = data[i] - interpolated_data[-1]
        if gap > step:
            num_elements: int = int(np.ceil(gap / step)) - 1
            interpolated_values: List[float] = [interpolated_data[-1] + step * j for j in range(1, num_elements + 1)]
            interpolated_data.extend(interpolated_values)
        interpolated_data.append(data[i])
    return np.array(interpolated_data)


def find_nonzero_region(indices: np.ndarray) -> Tuple[int, int]:
    _start: int = indices[0]
    _end: int = indices[0]
    current_start: int = indices[0]
    max_length: int = 0
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            current_length: int = indices[i-1] - current_start + 1
            if current_length > max_length:
                max_length = current_length
                _start = current_start
                _end = indices[i-1]
            current_start = indices[i]
    current_length = indices[-1] - current_start + 1
    if current_length > max_length:
        _start = current_start
        _end = indices[-1]
    return _start, _end

# ---- load data ----
print("== Load workbook ==")
DATA_FILE = './' + DATA_FILE
if CONDITION_DATA:
    CONDITION_FILE = './' + CONDITION_FILE

try:
    if CONDITION_DATA:
        condition_workbook = xlrd.open_workbook(CONDITION_FILE)
        condition_sheet = condition_workbook.sheet_by_name('Conditions')
        condition_list: List[List[float]] = []
        for row in range(1, condition_sheet.nrows):
            row_data = condition_sheet.row_values(row, 1, 4)
            condition_list.append(row_data)
        condition_list = [[float(x) for x in y] for y in condition_list]
        print('conditions(CH_WIDTH, CH_LENGTH, GIThick):', condition_list)
except Exception:
    CONDITION_DATA = False
    print('Condition file not found — fallback to manual CH_WIDTH/LENGTH/GI_THICK')

workbook: xlrd.Book = xlrd.open_workbook(DATA_FILE)
sheet_names: List[str] = workbook.sheet_names()
settings_sheet: xlrd.sheet.Sheet = workbook.sheet_by_name('Settings')
settings_df: pd.DataFrame = pd.DataFrame([settings_sheet.row_values(i) for i in range(settings_sheet.nrows)])

# Data sheet selection (skip first 3 sheets if they are header/meta)
data_names: List[str] = sheet_names[:1] + sheet_names[3:]

# Build per-data settings slices
name_indices = settings_df[settings_df[0].isin(sheet_names)].index
dividing = settings_df[settings_df[0].isin(['=================================='])].index
start_idx = dividing[0::2]
name_idx = dividing[1::2]

settings_list: List[np.ndarray] = []
for i in range(len(data_names)):
    if data_names[i] == settings_df.iloc[name_indices[i], 0]:
        if i < len(data_names) - 1:
            selected = settings_df.iloc[start_idx[i]:start_idx[i+1], 0:4].values
        else:
            selected = settings_df.iloc[start_idx[i]:, 0:4].values
    else:
        raise ValueError('data name not matching')
    settings_list.append(selected)

if len(settings_list) != len(data_names):
    raise ValueError('some data missing')

# === Header auto-detection setup ===
DEFAULT_CANDIDATES: Dict[str, List[str]] = {
    'Id': ['DrainI', 'DrainI_A', 'Id', 'Drain I', 'Drain I (A)', 'Id(A)'],
    'Vg': ['GateV', 'Vg', 'Gate V', 'Gate V (V)', 'Vg(V)'],
    'Vd': ['DrainV', 'Vd', 'Drain V', 'Drain V (V)'],
    'Ig': ['GateI', 'Ig', 'Gate I', 'Gate I (A)', 'GateI_A'],
}

print("== Parse sheets ==")

def find_column_index(header_row: List[str], user_value: Optional[str], candidates: List[str]) -> Optional[int]:
    norm_headers = [normalize(h) for h in header_row]
    # 1) exact user-specified match
    if user_value:
        nv = normalize(user_value)
        for idx, nh in enumerate(norm_headers):
            if nh == nv:
                return idx
    # 2) candidate exact match
    for cand in candidates:
        nc = normalize(cand)
        for idx, nh in enumerate(norm_headers):
            if nh == nc:
                return idx
    # 3) relaxed prefix match
    pool = ([user_value] if user_value else []) + candidates
    pool = [p for p in pool if p]
    for cand in pool:
        nc = normalize(cand)
        for idx, nh in enumerate(norm_headers):
            if nh.startswith(nc) and len(nh) >= len(nc):
                return idx
    return None

# Extract raw arrays from each sheet

drainI_list: List[np.ndarray] = []
gateI_list: List[Optional[np.ndarray]] = []
gateV_list: List[np.ndarray] = []
drainV_list: List[Optional[np.ndarray]] = []

for nm in data_names:
    sh = workbook.sheet_by_name(nm)
    rows = [sh.row_values(r) for r in range(sh.nrows)]
    data_np = np.array(rows, dtype=object)
    header = [str(x) for x in data_np[0, :]]

    idx_Id = find_column_index(header, CUSTOM_HEADERS['Id'], DEFAULT_CANDIDATES['Id'])
    idx_Vg = find_column_index(header, CUSTOM_HEADERS['Vg'], DEFAULT_CANDIDATES['Vg'])
    idx_Vd = find_column_index(header, CUSTOM_HEADERS['Vd'], DEFAULT_CANDIDATES['Vd'])
    idx_Ig = find_column_index(header, CUSTOM_HEADERS['Ig'], DEFAULT_CANDIDATES['Ig'])

    # Require Id & Vg
    if idx_Id is None or idx_Vg is None:
        drainI_list.append(np.array([], dtype=float))
        gateV_list.append(np.array([], dtype=float))
        drainV_list.append(None)
        gateI_list.append(None)
        print(f"[{nm}] header not found: Id or Vg — Id_idx={idx_Id}, Vg_idx={idx_Vg}")
        continue

    # Convert columns to float arrays
    try:
        I_d_i: np.ndarray = np.array(data_np[1:, idx_Id], dtype=float)
        V_g_i: np.ndarray = np.array(data_np[1:, idx_Vg], dtype=float)
    except Exception:
        I_d_i = np.array([], dtype=float)
        V_g_i = np.array([], dtype=float)
    drainI_list.append(I_d_i)
    gateV_list.append(V_g_i)

    # optional columns
    if idx_Vd is not None:
        try:
            V_d_i = float(data_np[1, idx_Vd])  # we expect constant drain voltage in transfer curve
        except Exception:
            V_d_i = np.nan
    else:
        V_d_i = np.nan
    drainV_list.append(np.array([V_d_i]))

    if idx_Ig is not None:
        try:
            I_g_i: np.ndarray = np.array(data_np[1:, idx_Ig], dtype=float)
        except Exception:
            I_g_i = np.array([], dtype=float)
        gateI_list.append(I_g_i)
    else:
        gateI_list.append(None)

# Containers
transfercurve_cols: List[np.ndarray] = []  # will be stacked column-wise at the end
subthresholdswing_cols: List[np.ndarray] = []
mobilityFE_cols: List[np.ndarray] = []
error_files_list: List[str] = []
result_rows: List[List[Union[str, float]]] = [
    ['name', 'Drain Voltage', 'Vth_current', 'Vth_interpol', 'Vth_logderivative', 'onoff ratio', 'Subthreshold Swing', 'u_FE(max)']
]

gi_eps: float = GI_EPS_R * epsilon_0 * 1e-2  # F/cm

# ---- per-dataset processing ----
print("== Process datasets ==")
for i, name in enumerate(data_names):
    print("---- dataset:", name)
    try:
        I_d = np.array(drainI_list[i]).flatten()
        V_g = np.array(gateV_list[i]).flatten()
        I_g_opt = gateI_list[i] if gateI_list[i] is not None else None

        # Basic sanity check (only Id & Vg are required)
        if (len(I_d) != len(V_g)) or (len(I_d) == 0):
            error_files_list.append(name + ' (length mismatch or empty)')
            print('skip: length mismatch or empty')
            continue

        # Drain voltage (optional; if absent, try from settings else NaN)
        setting = settings_list[i]
        V_d = np.nan
        try:
            step_row = np.where(setting[:, 0] == 'Step')[0][0]
            terminal_row = np.where(setting[:, 0] == 'Device Terminal')[0][0]
            drain_col = np.where(setting[terminal_row, :] == 'Drain')[0][0]
            bias_row = np.where(setting[:, 0] == 'Start/Bias')[0][0]
            if (setting[step_row, drain_col]) == 'N/A':
                V_d = float(setting[bias_row, drain_col])
        except Exception:
            pass
        # If still NaN and user provided Vd column, prefer that
        if np.isnan(V_d) and drainV_list[i] is not None and drainV_list[i].size:
            V_d = float(drainV_list[i][0]) if np.isfinite(drainV_list[i][0]) else V_d

        # ---- per-dataset plotting (always, if PLOTTING) ----
        if PLOTTING:
            plt.figure(figsize=(10, 6) if not MANUAL_PLOT else (FIGURE_SIZE_H, FIGURE_SIZE_V))
            ax1 = plt.gca(); ax2 = ax1.twinx()
            if APPLY_ABSOLUTE:
                ax1.plot(V_g, np.abs(I_d), label='Drain Current ($I_d$) - Log', linewidth=1.2)
                if I_g_opt is not None and I_g_opt.size == I_d.size:
                    ax1.plot(V_g, np.abs(I_g_opt), linestyle='--', alpha=0.35, label='Gate Current ($I_g$) - Log', linewidth=1.0)
            else:
                ax1.plot(V_g, I_d, label='Drain Current ($I_d$) - Log', linewidth=1.2)
                if I_g_opt is not None and I_g_opt.size == I_d.size:
                    ax1.plot(V_g, I_g_opt, linestyle='--', alpha=0.35, label='Gate Current ($I_g$) - Log', linewidth=1.0)
            ax1.set_yscale('log')
            ax1.set_xlabel('Gate Voltage ($V_g$) [V]')
            ax1.set_ylabel('Current [A] - Log')
            ymin = safe_min_positive(I_d, I_g_opt if I_g_opt is not None else np.array([np.inf]), default=1e-14) * 0.1
            ymax = max(np.nanmax(np.abs(I_d)), np.nanmax(np.abs(I_g_opt)) if I_g_opt is not None and I_g_opt.size else 0) * 10
            ymax = ymax if np.isfinite(ymax) and ymax > 0 else 1
            ax1.set_ylim(max(ymin, 1e-14), ymax)
            if MANUAL_PLOT:
                ax1.set_xlim(VG_MIN, VG_MAX)
            else:
                ax1.set_xlim(V_g.min(), V_g.max())
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            ax2.plot(V_g, I_d, alpha=0.3, label='Drain Current ($I_d$) - Linear')
            ax2.set_ylabel('Drain Current [A] - Linear')
            ax2.set_ylim(np.nanmin(I_d), np.nanmax(I_d) if np.isfinite(np.nanmax(I_d)) else None)
            ax2.legend(loc='upper right')
            title_Vd = f" V_d={V_d}" if np.isfinite(V_d) else ""
            plt.title(f"TransferCurve_{name}{title_Vd}")
            plt.tight_layout()
            plt.savefig(os.path.join(directory, f"TR_{name}-Vd-{V_d if np.isfinite(V_d) else 'NA'}.png"))
            plt.close()

        # ---- CSV accumulation for overlay plots ----
        col0 = np.concatenate((np.array([f'V_g-TC', name], dtype=object), V_g.astype(object)))
        col1 = np.concatenate((np.array(['Id', name], dtype=object), I_d.astype(object)))
        transfercurve_cols.append(col0)
        transfercurve_cols.append(col1)

        # ---- Parameter analyses (optional) ----
        vth_current = np.nan
        vth_interpol = np.nan
        vth_logderivative = np.nan
        onoff_ratio = np.nan
        subthreshold_swing = np.nan
        max_mu_linear = np.nan

        if (ANALYZE_VTH or ANALYZE_SS or ANALYZE_MOBILITY):
            Vg_step_interpolate: float = 0.2
            I_d_abs = np.abs(I_d)
            spline = UnivariateSpline(V_g, I_d_abs, s=0)
            spline_log10 = UnivariateSpline(V_g, np.log10(np.clip(I_d_abs, 1e-20, None)), s=0)

            if np.diff(V_g).max() >= Vg_step_interpolate and Vg_step_interpolate > 0:
                V_g_fine = interpolate_large_gaps(V_g, Vg_step_interpolate)
            else:
                V_g_fine = np.array(V_g)
            I_d_fine = spline(V_g_fine)
            log10_I_d_fine = spline_log10(V_g_fine)

            # d log10(I_d)/dV_g
            d_logId_dVg = np.gradient(log10_I_d_fine, V_g_fine)
            if DEVICE_TYPE == 'p':
                d_logId_dVg *= -1
            d_logId_dVg = np.where(d_logId_dVg > 0, d_logId_dVg, 1e-12)

            subthreshold_indices = np.where(np.log10(d_logId_dVg + 1e-12) >= -LOG_THRESHOLD_FIND_SS)[0]
            if subthreshold_indices.size == 0:
                error_files_list.append(name + ' (subthreshold_indices=0)')
                print('subthreshold_indices=0 — consider lowering LOG_THRESHOLD_FIND_SS')
            else:
                ss_start, ss_end = find_nonzero_region(subthreshold_indices)
                V_g_cut = V_g_fine[ss_start:ss_end]
                I_d_cut = I_d_fine[ss_start:ss_end]
                dlog_cut = d_logId_dVg[ss_start:ss_end]

                if ANALYZE_VTH:
                    # current-based
                    vth_current = float(np.round(np.interp(ID_AT_VTH, I_d_abs, V_g, left=np.nan, right=np.nan), 3))
                    # interpolation-based (linear window search)
                    step_base = Vg_step_interpolate if Vg_step_interpolate > 0 else max(np.diff(V_g).min(), 1e-3)
                    window_size = max(int(LINEAR_WINDOW_FOR_VTH / step_base), 3)
                    offset = int(3 / step_base)
                    safe_end = min(ss_end + offset, len(V_g_fine) - 1)
                    valid_idx = np.where((V_g_fine >= V_g_fine[ss_start]) & (V_g_fine <= V_g_fine[safe_end]))[0]
                    if valid_idx.size:
                        search_start, search_end = valid_idx[0], valid_idx[-1]
                        max_corr = -1.0
                        best_start = search_start
                        for st in range(search_start, search_end + 1):
                            ed = st + window_size
                            if ed > len(V_g_fine):
                                break
                            Vw = V_g_fine[st:ed]
                            Iw = I_d_fine[st:ed]
                            if Vw.size < 2:
                                continue
                            slope, intercept, r_value, *_ = linregress(Vw, Iw)
                            corr = abs(r_value)
                            if corr > max_corr:
                                max_corr = corr
                                best_start = st
                        lin_mask = np.zeros_like(V_g_fine, dtype=bool)
                        lin_mask[best_start:best_start + window_size] = True
                        V_lin = V_g_fine[lin_mask]
                        I_lin = I_d_fine[lin_mask]
                        if V_lin.size >= 2:
                            slope, intercept, *_ = linregress(V_lin, I_lin)
                            if slope != 0:
                                vth_interpol = float(np.round(-intercept / slope, 3))
                        # log-derivative based Vth
                        max_slope_idx = int(np.argmax(dlog_cut))
                        vth_logderivative = float(np.round(V_g_cut[max_slope_idx], 3))
                        onoff_ratio = float(np.max(np.abs(I_d_fine)) / max(safe_min_positive(I_d_fine), 1e-20))

                if ANALYZE_SS:
                    SS_values = 1000.0 / d_logId_dVg
                    SS_cut = SS_values[ss_start:ss_end]
                    SS_sorted = np.sort(SS_cut)
                    ge60 = SS_sorted[SS_sorted >= 60]
                    if ge60.size:
                        min_over_60 = ge60[0]
                        chosen = SS_sorted[SS_sorted <= min_over_60]
                        if chosen.size:
                            subthreshold_swing = float(np.round(chosen.max(), 3))
                    # Save SS columns
                    col_ss0 = np.concatenate((np.array(['V_g-SS', name], dtype=object), np.round(V_g_cut.astype(float), 6).astype(object)))
                    col_ss1 = np.concatenate((np.array(['SS(mV/dec)', name], dtype=object), np.round(SS_cut.astype(float), 6).astype(object)))
                    subthresholdswing_cols.append(col_ss0)
                    subthresholdswing_cols.append(col_ss1)

                if ANALYZE_MOBILITY:
                    V_g_rough = V_g_fine.copy(); I_d_rough = I_d_fine.copy()
                    if DENOISE_CURRENT:
                        I_d_rough = gaussian_filter1d(I_d_fine, 1)
                        fx = interpolate.interp1d(V_g_rough, I_d_rough, kind='cubic')
                        V_g_rough = np.arange(np.min(V_g_rough), np.max(V_g_fine), DIFFERENTIAL_ROUGHNESS)
                        I_d_rough = fx(V_g_rough)
                        rs = UnivariateSpline(V_g_rough, I_d_rough, s=0)
                        V_g_rough = np.arange(V_g_rough.min(), V_g_rough.max(), 1)
                        I_d_rough = rs(V_g_rough)

                    g_m = np.abs(np.gradient(I_d_rough, V_g_rough))
                    if REMOVE_OUTLIERS and g_m.size:
                        try:
                            thr = np.max(g_m) * 5e-2
                            spline_gm = UnivariateSpline(V_g_rough, g_m, s=np.max((g_m) * 5e-6))
                            diff = np.abs(g_m - spline_gm(V_g_rough))
                            keep = diff <= thr
                            g_m = g_m[keep]; V_g_rough = V_g_rough[keep]; I_d_rough = I_d_rough[keep]
                        except Exception:
                            pass

                    # geometry in cm
                    L = CH_LENGTH * 1e-4
                    W = CH_WIDTH * 1e-4
                    gi_thick_cm: float = GI_THICK * 1e-4
                    gi_cap: float = GI_EPS_R * epsilon_0 * 1e-2 / gi_thick_cm

                    vt = vth_interpol if np.isfinite(vth_interpol) else np.nan
                    mu_linear = (L / (W * gi_cap * max(V_d, 1e-6))) * g_m
                    mu_eff = (L / (W * gi_cap * max(V_d, 1e-6))) * (I_d_rough / np.maximum(np.abs(V_g_rough - vt), 1e-6))
                    mu_sat = (L / (W * gi_cap * max(V_d, 1e-6))) * (2 * I_d_rough / np.maximum((V_g_rough - vt) ** 2, 1e-6))

                    max_mu_linear = float(np.max(mu_linear)) if mu_linear.size else np.nan
                    # Save mobility cols
                    col_mu0 = np.concatenate((np.array(['V_g-u_FE', name], dtype=object), np.round(V_g_rough.astype(float), 6).astype(object)))
                    col_mu1 = np.concatenate((np.array(['mu_linear(cm^2/Vs)', name], dtype=object), np.round(mu_linear.astype(float), 6).astype(object)))
                    mobilityFE_cols.append(col_mu0)
                    mobilityFE_cols.append(col_mu1)

            result_rows.append([name, V_d, vth_current, vth_interpol, vth_logderivative, onoff_ratio, subthreshold_swing, max_mu_linear])

        # ---- Per-dataset analysis figure (optional) ----
        if (ANALYZE_VTH or ANALYZE_SS or ANALYZE_MOBILITY):
            fig, axs = plt.subplots(3, 3, figsize=(25, 12))
            try:
                if ANALYZE_VTH and np.isfinite(vth_interpol):
                    axs[0, 0].plot(V_g, I_d, label='I_d', linewidth=1.2)
                    axs[0, 0].axvline(vth_interpol, color='green', linestyle='--', label=f'Vth_interp={vth_interpol:.2f} V')
                    axs[0, 0].set_ylim(0, None)
                    axs[0, 0].legend(); axs[0, 0].grid(True); axs[0, 0].set_title('Transfer (linear)')
                else:
                    axs[0, 0].axis('off')

                if ANALYZE_VTH:
                    axs[0, 1].plot(V_g, np.abs(I_d) if APPLY_ABSOLUTE else I_d)
                    axs[0, 1].set_yscale('log'); axs[0, 1].grid(True); axs[0, 1].set_title('Transfer (log)')
                else:
                    axs[0, 1].axis('off')

                if ANALYZE_SS and 'V_g_cut' in locals():
                    axs[1, 1].plot(V_g_cut, 1000.0 / dlog_cut, label='SS (mV/dec)')
                    axs[1, 1].legend(); axs[1, 1].grid(True); axs[1, 1].set_title('SS vs Vg')
                    axs[1, 0].plot(V_g_cut, dlog_cut, label='d log10(I_d)/dVg')
                    axs[1, 0].legend(); axs[1, 0].grid(True)
                else:
                    axs[1, 0].axis('off'); axs[1, 1].axis('off')

                if ANALYZE_MOBILITY and 'mu_linear' in locals():
                    axs[2, 0].plot(V_g_rough, mu_linear, label='mu_linear')
                    axs[2, 0].legend(); axs[2, 0].grid(True); axs[2, 0].set_title('Mobility (linear)')
                else:
                    axs[2, 0].axis('off')
                axs[0, 2].axis('off'); axs[1, 2].axis('off'); axs[2, 1].axis('off'); axs[2, 2].axis('off')
            finally:
                plt.tight_layout()
                plt.savefig(os.path.join(directory, f"Analysis_{name}.png"))
                plt.close()

    except Exception as e:
        print('Process Error:', e)
        traceback.print_exc()
        error_files_list.append(name)

# ---- finalize CSV writes (after loop) ----
print("== Finalize & save tables ==")
# TransferCurve.csv
if len(transfercurve_cols) > 0:
    maxlen = max(len(c) for c in transfercurve_cols)
    padded_cols = [np.pad(c, (0, maxlen - len(c)), constant_values='') for c in transfercurve_cols]
    transfercurve_save = np.stack(padded_cols, axis=1)
    np.savetxt(os.path.join(directory, 'TransferCurve.csv'), transfercurve_save, delimiter=',', fmt='%s')
    print('Saved:', os.path.join(directory, 'TransferCurve.csv'))
else:
    print('No transfer curves to save.')

# SubthresholdSwing.csv
if len(subthresholdswing_cols) > 0:
    maxlen = max(len(c) for c in subthresholdswing_cols)
    padded_cols = [np.pad(c, (0, maxlen - len(c)), constant_values='') for c in subthresholdswing_cols]
    subthresholdswing_save = np.stack(padded_cols, axis=1)
    np.savetxt(os.path.join(directory, 'SubthresholdSwing.csv'), subthresholdswing_save, delimiter=',', fmt='%s')
    print('Saved:', os.path.join(directory, 'SubthresholdSwing.csv'))

# MobilityFE.csv
if len(mobilityFE_cols) > 0:
    maxlen = max(len(c) for c in mobilityFE_cols)
    padded_cols = [np.pad(c, (0, maxlen - len(c)), constant_values='') for c in mobilityFE_cols]
    mobilityFE_save = np.stack(padded_cols, axis=1)
    np.savetxt(os.path.join(directory, 'MobilityFE.csv'), mobilityFE_save, delimiter=',', fmt='%s')
    print('Saved:', os.path.join(directory, 'MobilityFE.csv'))

# results.csv
if len(result_rows) > 1:
    np.savetxt(os.path.join(directory, 'results.csv'), np.array(result_rows, dtype=object), delimiter=',', fmt='%s')
    print('Saved:', os.path.join(directory, 'results.csv'))
    
# ---- overlay plots (after loop) ----
print("\n== Overlay plots ==")
TC_path = os.path.join(directory, 'TransferCurve.csv')
if os.path.exists(TC_path):
    # 0행 = 역할명, 1행 = 데이터셋 이름(범례에 쓰기 좋음)
    hdr_role = pd.read_csv(TC_path, header=None, nrows=1).values[0]
    hdr_name = pd.read_csv(TC_path, header=None, skiprows=1, nrows=1).values[0]

    # 본 데이터는 2행부터
    df = pd.read_csv(TC_path, header=None, skiprows=2, dtype=str)
    num_cols = df.shape[1]
    if num_cols % 2 != 0:
        num_cols -= 1  # 홀수면 마지막 컬럼은 버림(불완전 페어)

    # linear overlay
    plt.figure(figsize=(10, 6))
    for i in range(0, num_cols, 2):
        x = pd.to_numeric(df.iloc[:, i], errors='coerce')
        y = pd.to_numeric(df.iloc[:, i + 1], errors='coerce')
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            continue
        label = str(hdr_name[i + 1]) if (i + 1) < len(hdr_name) else f'col{i+1}'
        plt.plot(x[mask].values, y[mask].values, label=label)
    plt.xlabel('Gate Voltage (V)'); plt.ylabel('Drain Current (A)')
    # plt.title('Overlayed Curves (linear)'); plt.legend(); plt.grid(True)
    plt.title('Overlayed Curves (linear)'); plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(directory, 'Total_linear.png')); plt.close()

    # log overlay
    plt.figure(figsize=(10, 6))
    for i in range(0, num_cols, 2):
        x = pd.to_numeric(df.iloc[:, i], errors='coerce')
        y = pd.to_numeric(df.iloc[:, i + 1], errors='coerce').abs()
        mask = x.notna() & y.notna() & (y > 0)
        if mask.sum() < 2:
            continue
        label = str(hdr_name[i + 1]) if (i + 1) < len(hdr_name) else f'col{i+1}'
        plt.plot(x[mask].values, y[mask].values, label=label)
    plt.xlabel('Gate Voltage (V)'); plt.ylabel('Abs Drain Current (A)')
    plt.yscale('log'); plt.ylim(1e-14, None)
    # plt.title('Overlayed Curves (log)'); plt.legend(); plt.grid(True)
    plt.title('Overlayed Curves (log)'); plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(directory, 'Total_log.png')); plt.close()
else:
    print('TransferCurve.csv not found — skip overlay plots')

# ---- summary ----
if len(error_files_list) > 0:
    print('[WARN] Check these datasets:', error_files_list)
else:
    print('All datasets processed without critical errors.')