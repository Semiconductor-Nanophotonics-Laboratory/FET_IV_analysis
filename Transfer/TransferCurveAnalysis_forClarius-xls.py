'''
    ### Description ###
    # =====================================================================================================
    # Transfer Curve Analysis for FETs
    #
    # Key Calculations:
    #
    # 1. **Threshold Voltage (V_th)**
    #    - Determines the voltage at which the device turns on.
    #    - Three methods:
    #      1) **Current-based (V_th_current)**: V_g at a specific I_d threshold.
    #      2) **Interpolation-based (V_th_interpol)**: Linear regression in the linear region.
    #      3) **Log-derivative (V_th_logderivative)**: Maximum d(log(I_d)) / dV_g.
    #
    # 2. **Subthreshold Swing (SS)**
    #    - Measures how efficiently the transistor switches.
    #    - `SS = 1000 / (d log(I_d) / dV_g) (mV/dec)`.
    #
    # 3. **Field-Effect Mobility (μ)**
    #    - Determines charge carrier transport efficiency.
    #    - **Linear region:** `μ_linear = (L / (W * C_ox * V_d)) * g_m`
    #    - **Effective mobility:** `μ_eff = (L / (W * C_ox * V_d)) * (I_d / |V_g - V_th|)`
    #    - **Saturation mobility:** `μ_sat = (L / (W * C_ox * V_d)) * (2 * I_d / (V_g - V_th)^2)`
    #    - For **p-type FETs**, calculations use `|V_g - V_th|` where needed.
    #
    # 4. **On/Off Ratio**
    #    - `On/Off Ratio = max(I_d) / min(I_d)`, indicates switching performance.
    # =====================================================================================================
'''

###########################      user input      ############################       

DATA_FILE = 'TC_example2.xls'  # Path to the data file

SERIES_NAME = 'IGZO_project1'  # Name of the series for saving results

DEVICE_TYPE = 'n'  # 'n' for n-type FETs, 'p' for p-type FETs

CONDITION_DATA = False  # Set to True if using a conditions file
if CONDITION_DATA:
    CONDITION_FILE = 'Conditions.xls'  # Path to the conditions file
else:
    CH_WIDTH = 1  # Size of Channel (um)
    CH_LENGTH = 1  # Size of Channel (um)
    GI_THICK = 0.09  # Thickness of gate insulator (um)

GI_EPS_R = 3.9  # Relative permittivity (3.9 for SiO2)

ID_AT_VTH = 1e-8  # A  # Drain current threshold for Vth calculation
LINEAR_WINDOW_FOR_VTH = 5  # V  # Window size for linear region in Vth calculation
PLOTTING = True  # Set to True to plot transfer curves

if PLOTTING:
    APPLY_ABSOLUTE = True  # Use absolute values for current in plots
    MANUAL_PLOT = False  # Set to True for manual plot
    if MANUAL_PLOT:
        VG_MIN = -20  # Minimum gate voltage for manual plot
        VG_MAX = +20  # Maximum gate voltage for manual plot
        ID_MIN = 1e-12  # Minimum drain current for manual plot
        ID_MAX = 1e-3  # Maximum drain current for manual plot
        FIGURE_SIZE_H = 8  # Horizontal size of the figure
        FIGURE_SIZE_V = 6  # Vertical size of the figure

ANALYZE_PARAMETERS = True  # Set to True to analyze Vth, SS, and mobility

############# Specified control !!! (change ONLY IF you know what you are doing) !!! #############
if ANALYZE_PARAMETERS:
    ANALYZE_VTH = True  # Calculate Vth using three methods
    ANALYZE_SS = True  # Calculate Subthreshold Swing
    ANALYZE_MOBILITY = True  # Calculate field-effect Mobility
    
    DENOISE_CURRENT = False  # NOT recommended: denoise I_d for calculating mobility
    REMOVE_OUTLIERS = True  # Remove outliers from data
    
    DIFFERENTIAL_ROUGHNESS = 2  # (1~4) recommended
    LOG_THRESHOLD_FIND_SS = 1.2  # (1.1~2) recommended for Subthreshold Swing calculation

################################################################################################

from datetime import datetime
start_time = datetime.now().strftime("%H-%M-%S_%Y%m%d")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120
import xlrd
from scipy.constants import epsilon_0
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import traceback
DEBUG = False

if True:
    directory = './Debug/' if DEBUG else f'./result/TransferCurve/{SERIES_NAME}_{DATA_FILE}_{start_time}/'
    os.makedirs(directory, exist_ok=True)
    if not DEBUG:
        backup_path = os.path.join(directory, 'code_backup.py')
        with open(__file__, 'r', encoding='utf-8') as src, \
            open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())

if True:   ### prepare for data analysis ###
    DATA_FILE = './' + DATA_FILE
    
    if CONDITION_DATA:
        try:
            CONDITION_FILE = './' + CONDITION_FILE
            condition_workbook = xlrd.open_workbook(CONDITION_FILE)
            condition_sheet = condition_workbook.sheet_by_name('Conditions')
            condition_list = []
            for row in range(1,condition_sheet.nrows):
                row_data = condition_sheet.row_values(row, 1, 4)
                condition_list.append(row_data)
            condition_list = [[float(x) for x in y] for y in condition_list]
            print('\nconditions(CH_WIDTH, CH_LENGTH, GIThick):\n',condition_list)
        except Exception as e:
            CONDITION_DATA = False
            print('Condition file not found (using manual input)\n CH_WIDTH, CH_LENGTH, GI_THICK')
            
    workbook = xlrd.open_workbook(DATA_FILE)
    sheet_names = workbook.sheet_names()
    data_names = sheet_names[:1]+sheet_names[3:]
    settings_sheet = workbook.sheet_by_name('Settings')
    settings_df = pd.DataFrame([settings_sheet.row_values(i) for i in range(settings_sheet.nrows)])
    data_name_indicies = settings_df[settings_df[0].isin(sheet_names)].index
    dividing_indicies = settings_df[settings_df[0].isin(['=================================='])].index
    start_dividing_index = dividing_indicies[0::2]
    name_dividing_index = dividing_indicies[1::2]
    
    settings_list = []
    for i in range(len(data_names)):
        if data_names[i] == settings_df.iloc[data_name_indicies[i], 0]:
            if i < len(data_names)-1:
                selected_data = settings_df.iloc[start_dividing_index[i]:start_dividing_index[i+1], 0:4].values
            else:
                selected_data = settings_df.iloc[start_dividing_index[i]:, 0:4].values
        else:
            raise ValueError('data name not matching')
        settings_list.append(selected_data)
        # print('\n\n\nselected data:',selected_data)
        del selected_data
        
    if len(settings_list) != len(data_names):
        raise ValueError('some data missing')
    
    drainI_list = []
    # drainV_list = []
    gateI_list = []
    gateV_list = []
    for i in range(len(data_names)):
        sheet_i = workbook.sheet_by_name(data_names[i])
        data_i = []
        for row_idx in range(sheet_i.nrows):
            row = sheet_i.row_values(row_idx)
            data_i.append(row)
        data_i = np.array(data_i)
        I_d_i = np.array(data_i)[1:,data_i[0,:] == 'DrainI'].astype(float)
        drainI_list.append(I_d_i)
        I_g_i = np.array(data_i)[1:,data_i[0,:] == 'GateI'].astype(float)
        gateI_list.append(I_g_i)
        # V_d_i = np.array(data_i)[1:,1]
        # drainV_list.append(V_d_i)
        V_g_i = np.array(data_i)[1:,data_i[0,:] == 'GateV'].astype(float)
        gateV_list.append(V_g_i)
        del sheet_i, data_i, I_d_i,I_g_i, V_g_i #, V_d_i
        
    transfercurve_list = []
    subthresholdswing_list = []
    mobilityFE_list = []
    mobilitySat_list = []
    error_files_list = []
    result_list = [['name', 'Drain Voltage', 'Vth_current', 'Vth_interpol', 'Vth_logderivative', 'onoff ratio', 'Subthreshold Swing', 'u_FE(max)']]
    
    gi_eps = GI_EPS_R *epsilon_0 *1e-2  # Permitivity of GI (F/cm)

if True:   ### pre-define functions for data analysis ###
    def lowpass_window(data, window_size=15):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    def check_monotonic(array):
        differences = np.diff(array)
        is_increasing = np.all(differences >= 0)
        is_decreasing = np.all(differences <= 0)
        if is_increasing:
            return 1
        elif is_decreasing:
            return -1
        else:
            return 0
    
    # def split_by_trend(arr):
    #     if len(arr) < 2:
    #         return [arr]
    #     trends = np.sign(np.diff(arr))
    #     segments = []
    #     start = 0
    #     for i in range(1, len(trends)):
    #         if trends[i] != trends[i - 1]:
    #             segments.append(arr[start:i+1])
    #             start = i
    #     segments.append(arr[start:])
    #     return segments

    def get_segment_indices(arr, n=-1):
        if len(arr) < 2:
            return np.array([0]) if len(arr) > 0 else np.array([])

        trends = np.sign(np.diff(arr))
        indices = []
        start = 0
        i = 0

        while i < len(trends):
            # flat(0) 건너뛰기
            if trends[i] == 0:
                i += 1
                start = i
                continue
            # 현재 방향 유지되는 동안 진행
            direction = trends[i]
            j = i + 1
            while j < len(trends) and trends[j] == direction:
                j += 1
            end = j
            indices.append(np.arange(start, end + 1))  # arr 인덱스 기준
            i = j
            start = j

        if n == -1:
            return max(indices, key=len) if indices else np.array([])
        elif 0 <= n < len(indices):
            return indices[n]
        else:
            raise ValueError(f"n too big. (0 <= n <= {len(indices)-1})")
        
    
    def interpolate_large_gaps(data, step, donot_interpolate=False):
        data = np.array(data)
        if step <= 0:
            donot_interpolate = True
        if donot_interpolate:
            return data.tolist()
        interpolated_data = [data[0]]
        for i in range(1, len(data)):
            gap = data[i] - interpolated_data[-1]
            if gap > step:
                num_elements = int(np.ceil(gap / step)) - 1
                interpolated_values = [
                    interpolated_data[-1] + step * j
                    for j in range(1, num_elements + 1)
                ]
                interpolated_data.extend(interpolated_values)
            interpolated_data.append(data[i])
        return np.array(interpolated_data)
    
    def find_nonzero_region(indices):
        _start = indices[0]
        _end = indices[0]
        current_start = indices[0]
        max_length = 0
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                current_length = indices[i-1] - current_start + 1
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

if True:   ### data analysis (calculate each single datasheet) ###
    for i in range(len(data_names)):
        print('\n\n**** ==== **** ==== ****\nProcessing data:',data_names[i])
        
        try :
            CalculateVth = ANALYZE_VTH
            CalculateSS = ANALYZE_SS
            CalculateMobility = ANALYZE_MOBILITY
            
            ###################################
            if CONDITION_DATA:
                CH_WIDTH = float(condition_list[i][0])
                CH_LENGTH = float(condition_list[i][1])
                GI_THICK = float(condition_list[i][2])
                gi_thick_cm = GI_THICK *1e-4
                DEVICE_TYPE = str(condition_list[i][3])
            else:
                gi_thick_cm = GI_THICK *1e-4  # cm
            gi_cap = gi_eps /gi_thick_cm #F/cm^2
            
            name = data_names[i]
            setting = settings_list[i]
            # I_d = np.abs(drainI_list[i].astype(float)).flatten()
            # I_g = np.abs(gateI_list[i].astype(float)).flatten()
            I_d = np.array(drainI_list[i].astype(float)).flatten()
            I_g = np.array(gateI_list[i].astype(float)).flatten()
            V_g = np.array(gateV_list[i].astype(float)).flatten()
            
            if check_monotonic(V_g) >0:
                print('Positive single sweep')
            elif check_monotonic(V_g) <0:
                print('Negative single sweep')
                I_d = I_d[::-1]
                I_g = I_g[::-1]
                V_g = V_g[::-1]
            else:
                print('Not a single sweep')
                if CalculateVth == True or CalculateSS == True or CalculateMobility == True:
                    seg_num = 1   # 0 means longest segment-1
                    idx_anl = get_segment_indices(V_g, n=seg_num)  
                    I_d = I_d[idx_anl]
                    I_g = I_g[idx_anl]
                    V_g = V_g[idx_anl]
                    if check_monotonic(V_g) < 0:
                        print('Negative single sweep segment')
                        I_d = I_d[::-1]
                        I_g = I_g[::-1]
                        V_g = V_g[::-1]
                    # print(V_g)
                    print('Selected segment:', seg_num, f'with {len(I_d)} points')
                    
                else:
                    error_files_list.append(name+' (not a single sweep)')
            print('CalculateVth:',CalculateVth,'\nCalculateSS:',CalculateSS,'\nCalculateMobility:',CalculateMobility)
            
            if (len(I_d) != len(I_g) or len(I_d) != len(V_g)) or (len(I_d) == 0 or len(I_g) == 0 or len(V_g) == 0):
                error_files_list.append(name)
                continue
            
            step_row = np.where(setting[:,0] == 'Step')[0][0]
            terminal_row = np.where(setting[:,0] == 'Device Terminal')[0][0]
            gate_col = np.where(setting[terminal_row,:] == 'Gate')[0][0]
            drain_col = np.where(setting[terminal_row,:] == 'Drain')[0][0]
            bias_row = np.where(setting[:,0] == 'Start/Bias')[0][0]
            
            if (setting[step_row, gate_col]) != 'N/A':
                V_g_step = float(setting[step_row, gate_col])
            else:
                V_g_step = False
            try:
                if (setting[step_row, drain_col]) == 'N/A':
                    V_d = float(setting[bias_row, drain_col])
                else:
                    raise ValueError('Drain Voltage not constant(: not a transfer curve)')
            except ValueError as error:
                print(error)
                error_files_list.append(name)
                continue
            
            ##########################################
            if PLOTTING:
                plt.figure(figsize=(FIGURE_SIZE_H if MANUAL_PLOT else 10, FIGURE_SIZE_V if MANUAL_PLOT else 6))
                ax1 = plt.gca() 
                ax2 = ax1.twinx()
                if APPLY_ABSOLUTE:
                    ax1.plot(V_g, np.abs(I_d), label='Drain Current ($I_d$) - Log Scale', color='blue')
                    ax1.plot(V_g, np.abs(I_g), color='red', linestyle='--', alpha=0.3, label='Gate Current ($I_g$) - Log Scale')
                else:
                    ax1.plot(V_g, I_d, label='Drain Current ($I_d$) - Log Scale', color='blue')
                    ax1.plot(V_g, I_g, color='red', linestyle='--', alpha=0.3, label='Gate Current ($I_g$) - Log Scale')
                ax1.set_yscale('log')
                ax1.set_xlabel("Gate Voltage ($V_g$) [V]")
                ax1.set_ylabel("Current [A] - Log Scale")
                ax1.legend(loc='upper left')
                ax1.grid(True)
                ax2.plot(V_g, I_d, label='Drain Current ($I_d$) - Linear Scale', color='green',alpha=0.3)
                ax2.set_ylabel("Drain Current [A] - Linear Scale")
                ax2.legend(loc='upper right')
                ax1.set_xticks(np.arange(V_g.min(), V_g.max() + 1, 5))
                if MANUAL_PLOT:
                    ax1.set_xlim(VG_MIN, VG_MAX)
                    ax1.set_ylim(ID_MIN, ID_MAX)
                    ax2.set_ylim(ID_MIN, ID_MAX)
                else:
                    ax1.set_xlim(V_g.min(), V_g.max())
                    ymin = min(I_d.min(), I_g.min()) * 1e-1
                    ymax = max(I_d.max(), I_g.max()) * 1e+1
                    if ymin <= 0:
                        ymin = 1e-12
                    ax1.set_ylim(ymin, ymax)
                    # ax1.set_ylim(min(I_d.min(), I_g.min())*1e-1, max(I_d.max(), I_g.max())*1e+1)
                    ax2.set_ylim(I_d.min(), I_d.max())
                plt.title(f"TransferCurve_{name} V_d-{V_d}")
                plt.savefig(directory + f"TR_{name}-V_d-{V_d}.png")
                plt.cla()
                plt.clf()
                plt.close()
            
            ###################################
            if CalculateVth or CalculateSS or CalculateMobility:
                Vg_step_interpolate = 0.2
                
                I_d = np.abs(I_d)
                I_g = np.abs(I_g)
                
                spline = UnivariateSpline(V_g, I_d, s=0)
                spline_log = UnivariateSpline(V_g, np.log(I_d), s=0)
                
                if np.diff(V_g).max() >= Vg_step_interpolate and Vg_step_interpolate > 0:
                    V_g_fine = interpolate_large_gaps(V_g, Vg_step_interpolate)
                    print('interpolated with {}V'.format(Vg_step_interpolate))
                elif Vg_step_interpolate < 0:
                    raise ValueError('Vg_step_interpolate should be positive')
                else:
                    V_g_fine = np.array(V_g)
                    print('no interpolation')
                I_d_fine = spline(V_g_fine)
                log_I_d_fine = spline_log(V_g_fine)
                
                if DEVICE_TYPE == 'n':
                    d_log_I_d_dV_g_fine = np.gradient(log_I_d_fine, V_g_fine)
                    d_log_I_d_dV_g_fine = np.where(d_log_I_d_dV_g_fine > 0, d_log_I_d_dV_g_fine, 1e-12)
                elif DEVICE_TYPE == 'p':
                    d_log_I_d_dV_g_fine = np.gradient(log_I_d_fine, V_g_fine)*-1
                    d_log_I_d_dV_g_fine = np.where(d_log_I_d_dV_g_fine > 0, d_log_I_d_dV_g_fine, 1e-12)
                
                if DEVICE_TYPE == 'n':
                    subthreshold_indices = np.where(np.log10(d_log_I_d_dV_g_fine +1e-9) >= (LOG_THRESHOLD_FIND_SS)*-1)[0]
                elif DEVICE_TYPE == 'p':
                    subthreshold_indices = np.where(np.log10(d_log_I_d_dV_g_fine +1e-9) >= (LOG_THRESHOLD_FIND_SS)*-1)[0]
                
                if len(subthreshold_indices) == 0:
                    error_files_list.append(name+' (subthreshold_indices = 0)')
                    print('subthreshold_indices = 0\nenlarge - "LOG_THRESHOLD_FIND_SS"')
                    continue
                
                if DEBUG:
                    def plot_d_log_I_d(V_g_fine, d_log_I_d_dV_g_fine):        # for debugging & parameter tuning
                        plt.figure(figsize=(8, 5))
                        plt.plot(V_g_fine, d_log_I_d_dV_g_fine, marker='o', linestyle='-', color='b', label='d_log(I_d) / dV_g')
                        plt.xlabel('V_g_fine (V)', fontsize=12)
                        plt.ylabel('d log(I_d) / dV_g', fontsize=12)
                        plt.title('Subthreshold Slope (d log(I_d) / dV_g vs V_g)', fontsize=14)
                        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.legend()
                        plt.show()
                        plt.figure(figsize=(8, 5))
                        plt.plot(V_g_fine, np.log10(d_log_I_d_dV_g_fine +1e-9), marker='o', linestyle='-', color='r', label='d_log(I_d) / dV_g')
                        plt.axhline(LOG_THRESHOLD_FIND_SS*-1, color='r', linestyle='--', linewidth=1.5, label=f'LOG_THRESHOLD_FIND_SS = {(LOG_THRESHOLD_FIND_SS*-1)}')
                        plt.xlabel('V_g_fine (V)', fontsize=12)
                        plt.ylabel('log--d log(I_d) / dV_g', fontsize=12)
                        plt.title('Subthreshold Slope (d log(I_d) / dV_g vs V_g)', fontsize=14)
                        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.legend()
                        plt.show()
                    plot_d_log_I_d(V_g_fine, d_log_I_d_dV_g_fine)
                
                ss_start, ss_end = find_nonzero_region(subthreshold_indices)
                
                V_g_fine_cut = V_g_fine[ss_start:ss_end]
                I_d_fine_cut = I_d_fine[ss_start:ss_end]
                d_log_I_d_dV_g_cut = d_log_I_d_dV_g_fine[ss_start:ss_end]
                print('\n------ *** -----\nV_d:', V_d)

            
            if CalculateVth:
                vth_current = np.round(np.interp(ID_AT_VTH, I_d, V_g, left=np.nan, right=np.nan), 3)

                step_base = Vg_step_interpolate if Vg_step_interpolate > 0 else (V_g_step if V_g_step else np.diff(V_g).min())
                window_size = int(LINEAR_WINDOW_FOR_VTH / step_base)

                offset = int(3 / step_base)
                if DEVICE_TYPE == 'n':
                    safe_end = min(ss_end + offset, len(V_g_fine) - 1)
                    valid_indices = np.where(
                        (V_g_fine >= V_g_fine[ss_start]) &
                        (V_g_fine <= V_g_fine[safe_end])
                    )[0]
                elif DEVICE_TYPE == 'p':
                    safe_start = max(ss_start - offset, 0)
                    valid_indices = np.where(
                        (V_g_fine <= V_g_fine[ss_end]) &
                        (V_g_fine >= V_g_fine[safe_start])
                    )[0]

                if len(valid_indices) == 0:
                    error_files_list.append(name + ' (may be 1. too large "LOG_THRESHOLD_FIND_SS" // 2. failed FET)')
                    print('may be 1. too large "LOG_THRESHOLD_FIND_SS" // 2. failed FET')
                    continue

                search_start, search_end = valid_indices[0], valid_indices[-1]
                max_corr = 0
                best_start = search_start
                for start in range(search_start, search_end):
                    end = start + window_size
                    if end > len(V_g_fine):  # 범위 초과 방지
                        break
                    V_g_window = V_g_fine[start:end]
                    I_d_window = I_d_fine[start:end]
                    slope, intercept, r_value, _, _ = linregress(V_g_window, I_d_window)
                    corr = abs(r_value)
                    if corr > max_corr:
                        max_corr = corr
                        best_start = start

                linear_region_mask = np.zeros_like(V_g_fine, dtype=bool)
                linear_region_mask[best_start:best_start + window_size] = True
                V_g_linear = V_g_fine[linear_region_mask]
                I_d_linear = I_d_fine[linear_region_mask]

                slope, intercept, _, _, _ = linregress(V_g_linear, I_d_linear)
                vth_interpol = np.round(-intercept / slope, 3)

                max_slope_index_fine = np.argmax(d_log_I_d_dV_g_cut)
                vth_logderivative = np.round(V_g_fine_cut[max_slope_index_fine], 3)

                onoff_ratio = np.max(np.abs(I_d_fine)) / np.min(np.abs(I_d_fine))

                print(f"\nvth_current(at {ID_AT_VTH}A): {vth_current} V")
                print(f"vth_interpol: {vth_interpol} V")
                print(f"vth_logderivative: {vth_logderivative} V")
                print(f"onoff_ratio: {np.round(onoff_ratio)}")
            else:
                vth_current = np.nan
                vth_interpol = np.nan
                vth_logderivative = np.nan
                onoff_ratio = np.nan

            
            if CalculateSS:
                SS_values = 1000 / d_log_I_d_dV_g_fine
                SS_values_cut = SS_values[ss_start:ss_end]
                SS_values_cut_sorted = np.sort(SS_values_cut)
                min_over_60 = SS_values_cut_sorted[SS_values_cut_sorted >= 60][0]
                results = SS_values_cut_sorted[SS_values_cut_sorted <= min_over_60]
                subthreshold_swings = np.round(results,3)
                subthreshold_swing = np.max(subthreshold_swings)
                print('\nsubthreshold_swings: {} (mV/dec)'.format(subthreshold_swings))
                print('subthreshold_swing (over 60): {} (mV/dec)'.format(subthreshold_swing))
            else:
                subthreshold_swing = np.nan
            
            if CalculateMobility:
                
                if DENOISE_CURRENT:
                    I_d_rough = gaussian_filter1d(I_d_fine, 1)
                    V_g_rough = V_g_fine
                    fx_roughen = interpolate.interp1d(V_g_rough,I_d_rough,kind='cubic')
                    V_g_rough = np.arange(np.min(V_g_rough),np.max(V_g_fine),DIFFERENTIAL_ROUGHNESS)
                    I_d_rough = fx_roughen(V_g_rough)
                    print('Roughened with {}V'.format(DIFFERENTIAL_ROUGHNESS))
                    roughspline = UnivariateSpline(V_g_rough, I_d_rough, s=0)   #hyperparameter
                    V_g_rough = np.arange(V_g_rough.min(), V_g_rough.max(), 1)
                    I_d_rough = roughspline(V_g_rough)
                else:
                    V_g_rough = V_g_fine
                    I_d_rough = I_d_fine
                    print('no roughening')
                
                g_m = np.abs(np.gradient(I_d_rough, V_g_rough))  # dI_d/dV_g
                
                if REMOVE_OUTLIERS:
                    threshold = np.max(g_m)*5e-2     #hyperparameter
                    spline_gm = UnivariateSpline(V_g_rough, g_m, s=np.max((g_m)*5e-6))     #hyperparameter    
                    diff = np.abs(g_m - spline_gm(V_g_rough))
                    outliers = diff > threshold
                    g_m = g_m[~outliers]
                    V_g_rough = V_g_rough[~outliers]
                    I_d_rough = I_d_rough[~outliers]
                
                if DEVICE_TYPE == 'n':
                    mu_linear = (CH_LENGTH / (CH_WIDTH * gi_cap * V_d)) * g_m
                    mu_eff = (CH_LENGTH / (CH_WIDTH * gi_cap * V_d)) * (I_d_rough / (V_g_rough - vth_interpol))
                    mu_sat = (CH_LENGTH / (CH_WIDTH * gi_cap * V_d)) * (2 * I_d_rough / (V_g_rough - vth_interpol) ** 2)

                    cond_indices = np.where(V_g_rough > V_g_fine[ss_end])[0]
                    if cond_indices.size == 0:
                        print("[Mobility calc - n-type] No saturation region found (V_g_rough > V_g_fine[ss_end])")
                        idx_sat_reigion = None
                    else:
                        idx_sat_reigion = cond_indices[0]

                elif DEVICE_TYPE == 'p':
                    mu_linear = (CH_LENGTH / (CH_WIDTH * gi_cap * V_d)) * np.abs(g_m)
                    mu_eff = (CH_LENGTH / (CH_WIDTH * gi_cap * V_d)) * (I_d_rough / np.abs(V_g_rough - vth_interpol))
                    mu_sat = (CH_LENGTH / (CH_WIDTH * gi_cap * V_d)) * (2 * I_d_rough / (V_g_rough - vth_interpol) ** 2)

                    cond_indices = np.where(V_g_rough > V_g_fine[ss_start])[0]
                    if cond_indices.size == 0:
                        print("[Mobility calc - p-type] No saturation region found (V_g_rough > V_g_fine[ss_start])")
                        idx_sat_reigion = None
                    else:
                        idx_sat_reigion = cond_indices[0]
                
                max_mu_linear = np.max(mu_linear)
                
                print("\nu_FE(max): ", np.round(max_mu_linear,3))
            else:
                max_mu_linear = np.nan
            
            #############################################################################
            if CalculateVth or CalculateSS or CalculateMobility:
                fig, axs = plt.subplots(3, 3, figsize=(25, 12))
                if CalculateVth:
                    axs[0, 0].plot(V_g, I_d, label='Drain Current ($I_d$)', color='blue')
                    axs[0, 0].plot(V_g_linear, I_d_linear, 'o', label='Linear Region', color='orange', alpha=0.3)
                    axs[0, 0].axvline(vth_interpol, color='green', linestyle='--', label=f'Threshold Voltage ($V_th$): {vth_interpol:.2f} V')
                    V_g_extrap = np.linspace(vth_interpol, V_g_linear.max(), 10)
                    I_d_extrap = slope * V_g_extrap + intercept
                    axs[0, 0].plot(V_g_extrap, I_d_extrap, color='red', linestyle='--', label='Extrapolated Line')
                    axs[0, 0].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[0, 0].set_ylabel("Drain Current ($I_d$) [A]")
                    axs[0, 0].legend()
                    axs[0, 0].grid(True)
                    axs[0, 0].set_title("Transfer Curve: Linear Scale")
                    axs[0, 0].set_ylim(0, None)
                    axs[0, 1].plot(V_g, I_d, 'o', label='Original Drain Current ($I_d$)', color='cyan', alpha=0.5)
                    axs[0, 1].plot(V_g_fine, np.exp(log_I_d_fine), '-', label='Interpolated Drain Current', color='blue')
                    axs[0, 1].axvline(vth_logderivative, color='green', linestyle='--', label=f'Threshold Voltage ($V_th$): {vth_logderivative:.2f} V')
                    axs[0, 1].set_yscale('log')
                    axs[0, 1].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[0, 1].set_ylabel("Drain Current ($I_d$) [A]")
                    axs[0, 1].legend()
                    axs[0, 1].grid(True)
                    axs[0, 1].set_title("Transfer Curve: Log Scale")
                else:
                    axs[0, 0].axis('off')
                    axs[0, 1].axis('off')
                    
                if CalculateSS:
                    axs[1, 0].plot(V_g_fine, d_log_I_d_dV_g_fine, label=r'd($\log_{10}(I_d)$)/d$V_g$', color='purple')
                    axs[1, 0].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[1, 0].set_ylabel(r"d($\log_{10}(I_d)$)/d$V_g$")
                    axs[1, 0].axvspan(V_g_fine[ss_start], V_g_fine[ss_end], color='yellow', alpha=0.3, label=r'ss region')
                    axs[1, 0].set_title(r"$\log_{10}(I_d)$ vs ($V_g$)")
                    axs[1, 0].legend()
                    axs[1, 0].grid(True)
                    # axs[1, 1].plot(V_g_fine_cut, d_log_I_d_dV_g_cut, label=r'$\log_{10}(I_d)$', color='purple')
                    # axs[1, 1].set_xlabel("Gate Voltage ($V_g$) [V]")
                    # axs[1, 1].set_ylabel(r"$\log_{10}(I_d)$")
                    # axs[1, 1].set_title(r"$\log_{10}(I_d)$ vs ($V_g$) Cut")
                    # axs[1, 1].legend()
                    # axs[1, 1].grid(True)
                    axs[1, 1].plot(V_g_fine_cut, SS_values_cut, label='SS', color='blue')
                    axs[1, 1].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[1, 1].set_ylabel("SS (mV/decade)")
                    axs[1, 1].set_title("SS vs ($V_g$)")
                    axs[1, 1].legend(loc='upper right')
                    axs[1, 1].grid(True)
                    
                else:
                    axs[1, 0].axis('off')
                    axs[1, 1].axis('off')
                
                if CalculateMobility:
                    axs[0, 2].plot(V_g_rough, g_m, label='Transconductance ($g_m$)', marker='o', color='purple')
                    axs[0, 2].plot(V_g_rough, spline_gm(V_g_rough), '-', label='spline', color='green', alpha=0.5)
                    axs[0, 2].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[0, 2].set_ylabel("Transconductance ($g_m$) [A/V]")
                    axs[0, 2].set_title("Transconductance vs Gate Voltage")
                    axs[0, 2].legend()
                    axs[0, 2].grid(True)
                    
                    if DENOISE_CURRENT:
                        axs[1, 2].plot(V_g_rough, I_d_rough, label='Linear Region Mobility', marker='o',color='blue')
                        axs[1, 2].set_xlabel("Gate Voltage ($V_g$) [V]")
                        axs[1, 2].set_ylabel("Drain Current ($I_d$) [A]")
                        axs[1, 2].set_title("Smoothed I_d vs Gate Voltage")
                        axs[1, 2].legend()
                        axs[1, 2].grid(True)
                    else:
                        axs[1, 2].axis('off')
                    
                    axs[2, 0].plot(V_g_rough, mu_linear, label='Linear Region Mobility', marker='o',color='blue')
                    axs[2, 0].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[2, 0].set_ylabel(r"Mobility ($\mu$) [cm^2/V·s]")
                    axs[2, 0].set_title("Linear Mobility vs Gate Voltage")
                    axs[2, 0].legend()
                    axs[2, 0].grid(True)
                    
                    if DEVICE_TYPE == 'n':
                        axs[2, 1].plot(V_g_rough[idx_sat_reigion:], mu_eff[idx_sat_reigion:], label='Effective Mobility', marker='o',color='green')
                    elif DEVICE_TYPE == 'p':
                        axs[2, 1].plot(V_g_rough[:idx_sat_reigion], mu_eff[:idx_sat_reigion], label='Effective Mobility', marker='o',color='green')
                    axs[2, 1].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[2, 1].set_ylabel(r"Mobility ($\mu$) [cm^2/V·s]")
                    axs[2, 1].set_title("Effective Mobility vs Gate Voltage")
                    axs[2, 1].legend()
                    axs[2, 1].grid(True)
                    
                    if DEVICE_TYPE == 'n':
                        axs[2, 2].plot(V_g_rough[idx_sat_reigion:], mu_sat[idx_sat_reigion:], label='Saturation Mobility', marker='o',color='red')
                    elif DEVICE_TYPE == 'p':
                        axs[2, 2].plot(V_g_rough[:idx_sat_reigion], mu_sat[:idx_sat_reigion], label='Saturation Mobility', marker='o',color='red')
                    axs[2, 2].set_xlabel("Gate Voltage ($V_g$) [V]")
                    axs[2, 2].set_ylabel(r"Mobility ($\mu$) [cm^2/V·s]")
                    axs[2, 2].set_title("Saturation Mobility vs Gate Voltage")
                    axs[2, 2].legend()
                    axs[2, 2].grid(True)
                    
                else:
                    axs[0, 2].axis('off')
                    axs[2, 0].axis('off')
                    axs[2, 1].axis('off')
                    axs[2, 2].axis('off')
                    axs[1, 2].axis('off')
                plt.tight_layout()
                plt.savefig(directory +"Analysis_{}.png".format(name))
                plt.cla()
                plt.clf()
                plt.close()
                
                result_list.append([name, V_d, vth_current, vth_interpol, vth_logderivative, onoff_ratio, subthreshold_swing, max_mu_linear])
                np.savetxt(directory + 'results.csv', np.array(result_list), delimiter=',', fmt='%s')
            
            if PLOTTING:
                _transfercurve = np.vstack((np.array([['V_g-TC', name]], dtype=str), np.array([V_g, I_d], dtype=float).T)).T
                transfercurve_list.extend(_transfercurve)
                TC_maxlength = max(len(arr) for arr in transfercurve_list)
                transfercurve_save = np.array([np.pad(arr, (0, TC_maxlength - len(arr)), 'constant') for arr in transfercurve_list]).T
            np.savetxt(directory + 'TransferCurve.csv', transfercurve_save, delimiter=',', fmt='%s')
            if CalculateVth:
                _subthresholdswing = np.vstack((np.array([['V_g-SS', name]], dtype=str), np.round(np.array([V_g_fine_cut, SS_values_cut], dtype=float),6).T)).T
                subthresholdswing_list.extend(_subthresholdswing)
                SS_maxlength = max(len(arr) for arr in subthresholdswing_list)
                subthresholdswing_save = np.array([np.pad(arr, (0, SS_maxlength - len(arr)), 'constant') for arr in subthresholdswing_list]).T
                np.savetxt(directory + 'SubthresholdSwing.csv', subthresholdswing_save, delimiter=',', fmt='%s')
            if CalculateMobility:
                _mobilityFE = np.vstack((np.array([['V_g-u_FE', name]], dtype=str), np.round(np.array([V_g_rough, mu_linear], dtype=float),6).T)).T
                mobilityFE_list.extend(_mobilityFE)
                MF_maxlength = max(len(arr) for arr in mobilityFE_list)
                mobilityFE_save = np.array([np.pad(arr, (0, MF_maxlength - len(arr)), 'constant') for arr in mobilityFE_list]).T
                np.savetxt(directory + 'MobilityFE.csv', mobilityFE_save, delimiter=',', fmt='%s')
            
            def clear_variables():
                for var in [
                    "V_g_rough", "V_g_fine", "I_d_fine", "log_I_d_fine", "d_log_I_d_dV_g_fine",
                    "subthreshold_indices", "V_g_linear", "I_d_linear", "I_d_rough", "g_m",
                    "mu_linear", "mu_eff", "mu_sat", "spline_gm", "diff", "outliers",
                    "valid_indices", "search_start", "search_end", "best_start", "start", "end",
                    "V_g_window", "I_d_window", "corr", "max_corr", "V_g_extrap", "I_d_extrap",
                    "slope", "intercept", "threshold", "max_mu_linear", "idx_sat_reigion"
                ]:
                    if var in locals():
                        del locals()[var]
            clear_variables()
        
        except Exception as e:
            print('Process Error (Unknown):',e)
            traceback.print_exc()
            error_files_list.append(name)
    
    if PLOTTING:
        df = pd.read_csv(directory +'TransferCurve.csv', header=None, skiprows=1)
        header_row = pd.read_csv(directory + 'TransferCurve.csv', header=None, nrows=1).values[0]
        num_columns = df.shape[1]
        plt.figure(figsize=(10, 6))
        for i in range(0, num_columns, 2):
            x = df.iloc[:, i]
            y = df.iloc[:, i + 1]
            label = str(header_row[i + 1])
            plt.plot(x, y, label=label)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Overlayed Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(directory +"Total_linear.png")
        plt.cla()
        plt.clf()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        for i in range(0, num_columns, 2):
            x = df.iloc[:, i]
            y = np.abs(df.iloc[:, i + 1])
            label = str(header_row[i + 1])
            plt.plot(x, y, label=label)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.yscale('log')
        plt.ylim(1e-14, None)
        plt.title('Overlayed Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(directory +"Total_log.png")
        plt.cla()
        plt.clf()
        plt.close()
    
    if len(error_files_list) > 0:
        print('\n\n\n\n******\nsomething happened in files (you shall check error):',error_files_list)