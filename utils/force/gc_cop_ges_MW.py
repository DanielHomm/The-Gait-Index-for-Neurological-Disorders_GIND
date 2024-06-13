import pandas as pd
from scipy.signal import savgol_filter

def construct_gc_cop_ges(gc_counter, gc_counter_ges, num_gc_ges, gc_xvec, Strike, ToeOff, fto_gc, trial, ana, cut_off_frame):
    columns_cop = ['Trial', 'x-Werte 0-100%', 'x Frames', 'strikes', 'toeoffs', 
                   'x CoP', 'y CoP', 'x CoP filt', 'y CoP filt', 'x-Werte 0-100% gekürzt']
    gc_cop_ges = pd.DataFrame(columns=columns_cop)
    
    for i in range(len(gc_counter)):
        trial_nn = f"{trial}_{gc_counter[i]}" 
        row_index = gc_counter_ges + i 
        label = trial_nn 
        x_Werte = gc_xvec.iloc[num_gc_ges + gc_counter[i], 8] #'x-Werte in % GC analog' 
        x_Frames = gc_xvec.iloc[num_gc_ges + gc_counter[i], 7] # 'Frames GC analog' 
        strikes = 5 * Strike[gc_counter[i]:gc_counter[i]+2] - 5 * Strike[gc_counter[i]] + 1
        toeoffs = 5 * ToeOff[fto_gc-1+gc_counter[i]] - 5 * Strike[gc_counter[i]] + 1
  
        x_frame = int(round(gc_xvec.loc[num_gc_ges + gc_counter[i] + 1, 'Frame zum Zeitpunkt x=20% für GRFz Auswahl'])) 
        if ana['Force.Fz1'][5 * x_frame] <= ana['Force.Fz2'][5 * x_frame]: 
            Fz = ana['Force.Fz1'][gc_xvec.loc[num_gc_ges + gc_counter[i] + 1, 'Frames GC analog'] - 1] 
            My = ana['Moment.My1'][gc_xvec.loc[num_gc_ges + gc_counter[i] + 1, 'Frames GC analog'] - 1] 
            Mx = ana['Moment.Mx1'][gc_xvec.loc[num_gc_ges + gc_counter[i] + 1, 'Frames GC analog'] - 1] 
        else: 
            Fz = ana['Force.Fz2'][x_Frames - 1]
            My = ana['Moment.My2'][x_Frames - 1] 
            Mx = ana['Moment.Mx2'][x_Frames - 1] 

        x_CoP = My / Fz 
        y_CoP = Mx / Fz 

        x_CoP_filt = savgol_filter(x_CoP, window_length=27, polyorder=3) 
        y_CoP_filt = savgol_filter(y_CoP, window_length=27, polyorder=3) 
  
        x_CoP_filt = x_CoP_filt[cut_off_frame:-cut_off_frame-1] - x_CoP_filt[cut_off_frame] 
        y_CoP_filt = y_CoP_filt[cut_off_frame:-cut_off_frame-1] - y_CoP_filt[cut_off_frame] 

        if y_CoP_filt[int(len(y_CoP_filt) * 3 / 4)] < 0: 
            x_CoP_filt = -x_CoP_filt 
            y_CoP_filt = -y_CoP_filt 

        x_values = x_Werte[cut_off_frame:-cut_off_frame-1]
        gc_cop_ges.loc[row_index] = [label, x_Werte, x_Frames, strikes, toeoffs, x_CoP, y_CoP, x_CoP_filt, y_CoP_filt, x_values]

    return gc_cop_ges


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def process_cop_data(gc_cop_ges, gc_ang_ges, gz_counter_ges, cut_off_frame, side="R"):

    if side not in ["R", "L"]:
        raise ValueError("side must be either 'R' or 'L'")

    columns = [
        'Label', "", 'x CoP Winkel-MW', 'y CoP Winkel-MW',
        'Angles MW (FP3)', 'x CoP Winkel-konti', 'y CoP Winkel-konti',
        'Angles konti (FP3)', 'x CoP unberichtigt', 'y CoP unberichtigt',
        'strikes', 'toe offs'
    ]
    gc_cop_ges_MW = pd.DataFrame(columns=columns)
    # Initialize the DataFrame with empty and mean/std rows
    for i in range(gz_counter_ges + 3):
        if i < gz_counter_ges:
            gc_cop_ges_MW.loc[i] = ['' for _ in range(len(columns))]
        elif i == gz_counter_ges:
            gc_cop_ges_MW.loc[i] = ['MW'] + [None] * (len(columns) - 1)
        elif i == gz_counter_ges + 1:
            gc_cop_ges_MW.loc[i] = ['MW-std'] + [None] * (len(columns) - 1)
        elif i == gz_counter_ges + 2:
            gc_cop_ges_MW.loc[i] = ['MW+std'] + [None] * (len(columns) - 1)

    for n in range(gz_counter_ges):
        strikes_curr = gc_cop_ges.iloc[n, 3]
        gc_cop_ges_MW.iloc[n, 11] = gc_cop_ges.iloc[n, 4] / strikes_curr[1] * 100
        gc_cop_ges_MW.iloc[n, 10] = strikes_curr

        cop_old_x = gc_cop_ges.iloc[n, 7]
        cop_old_y = gc_cop_ges.iloc[n, 8]
        x_range_old = np.linspace(np.min(cop_old_x), np.max(cop_old_x), len(cop_old_x))
        x_range_new = np.linspace(np.min(cop_old_x), np.max(cop_old_x), 201)
        y_range_old = np.linspace(np.min(cop_old_y), np.max(cop_old_y), len(cop_old_y))
        y_range_new = np.linspace(np.min(cop_old_y), np.max(cop_old_y), 201)

        cop_new_x = interp1d(x_range_old, cop_old_x, kind='cubic', fill_value='extrapolate')(x_range_new)
        cop_new_y = interp1d(y_range_old, cop_old_y, kind='cubic', fill_value='extrapolate')(y_range_new)

        gc_cop_ges_MW.iloc[n, 8] = cop_new_x
        gc_cop_ges_MW.iloc[n, 9] = cop_new_y

        CoP_xy_rot = np.array([cop_new_x, cop_new_y])
        if side == "R":
            foot_internal_rotation = -gc_ang_ges.iloc[n, 16][:gc_ang_ges.iloc[n, 42]]
        elif side == "L":
            foot_internal_rotation = gc_ang_ges.iloc[n, 16][:gc_ang_ges.iloc[n, 42]]
            
        foot_internal_rotation = foot_internal_rotation[cut_off_frame//5:len(foot_internal_rotation)-cut_off_frame//5]
        foot_internal_rotation_MW = np.mean(foot_internal_rotation)

        RM_curr = np.array([
            [np.cos(np.radians(foot_internal_rotation_MW)), -np.sin(np.radians(foot_internal_rotation_MW))],
            [np.sin(np.radians(foot_internal_rotation_MW)), np.cos(np.radians(foot_internal_rotation_MW))]
        ])
        CoP_xy_rot2 = np.dot(RM_curr, CoP_xy_rot)
        gc_cop_ges_MW.iloc[n, 2] = CoP_xy_rot2[0, :]
        gc_cop_ges_MW.iloc[n, 3] = CoP_xy_rot2[1, :]
        gc_cop_ges_MW.iloc[n, 4] = foot_internal_rotation_MW
        CoP_xy_rot = np.dot(RM_curr, CoP_xy_rot)

        f_i_r_range_old = np.arange(len(foot_internal_rotation))
        f_i_r_range_new = np.linspace(0, len(foot_internal_rotation) - 1, len(cop_new_x))
        foot_internal_rotation = interp1d(f_i_r_range_old, foot_internal_rotation, kind='linear', fill_value='extrapolate')(f_i_r_range_new)

        for k in range(len(CoP_xy_rot[0])):
            RM_curr = np.array([
                [np.cos(np.radians(foot_internal_rotation[k])), -np.sin(np.radians(foot_internal_rotation[k]))],
                [np.sin(np.radians(foot_internal_rotation[k])), np.cos(np.radians(foot_internal_rotation[k]))]
            ])
            CoP_xy_rot[:, k] = np.dot(RM_curr, CoP_xy_rot[:, k])

        gc_cop_ges_MW.iloc[n, 5] = CoP_xy_rot[0, :]
        gc_cop_ges_MW.iloc[n, 6] = CoP_xy_rot[1, :]
        gc_cop_ges_MW.iloc[n, 7] = foot_internal_rotation

    for i in range(2, len(gc_cop_ges_MW.columns) - 2):
        column_data = gc_cop_ges_MW.iloc[0:gz_counter_ges, i].to_numpy()[0]
        mean_value = np.mean(column_data)
        std_value = np.std(column_data)
        gc_cop_ges_MW.iloc[gz_counter_ges, i] = mean_value
        gc_cop_ges_MW.iloc[gz_counter_ges + 1, i] = mean_value - std_value
        gc_cop_ges_MW.iloc[gz_counter_ges + 2, i] = mean_value + std_value

    return gc_cop_ges_MW
