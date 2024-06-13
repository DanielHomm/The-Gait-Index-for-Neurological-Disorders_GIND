import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def construct_gc_grf_ges(trial, force, gz_counter, gz_counter_ges, gc_xvec, fto_gc, grf_key, norm_grf_key, strike, toe_off, num_gc_ges, mass_Korr):
    """
    This function 
    """
    # Define columns for the DataFrame
    columns_grf = ['Trial', 'x-Werte', 'x GRF force', 'y GRF force', 'z GRF force',
                'normalized x GRF force', 'normalized y GRF force', 'normalized z GRF force',
               'strikes', 'toe offs', 'frames x']

    # Initialize DataFrames for R_gc_grf_ges and L_gc_grf_ges
    gc_grf_ges = pd.DataFrame(columns=columns_grf)

    if grf_key in force:
        for i in range(len(gz_counter)):
            trial_nn = f"{trial}_{gz_counter[i]}"
            row_index = gz_counter_ges + i

            label = trial_nn
            x_Werte = gc_xvec.iloc[num_gc_ges + gz_counter[i], 3]  # 'x-Werte'

            if force[grf_key][round(gc_xvec.iloc[num_gc_ges + gz_counter[i], 6]), 1] <= 0:
                x_GRF = force[grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
                y_GRF = force[grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            else:
                x_GRF = -force[grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
                y_GRF = -force[grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr

            z_GRF = force[grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
            normalized_x_GRF = force[norm_grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            normalized_y_GRF = force[norm_grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            normalized_z_GRF = force[norm_grf_key][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            strikes = strike[gz_counter[i]:gz_counter[i]+2] - strike[gz_counter[i]] + 1
            toeoffs = toe_off[fto_gc - 1 + gz_counter[i]] - strike[gz_counter[i]] + 1
            frames_x = gc_xvec.iloc[num_gc_ges + gz_counter[i], 1] - strike[gz_counter[i]] + 1

            gc_grf_ges.loc[row_index] = [label, x_Werte, x_GRF, y_GRF, z_GRF,
                                         normalized_x_GRF, normalized_y_GRF, normalized_z_GRF,
                                         strikes, toeoffs, frames_x]
    return gc_grf_ges


def process_grf_data(gz_counter_ges, gc_grf_ges):

    columns_grf = ['Label', 'x-Werte', 'x GRF force', 'y GRF force', 'z GRF force',
               'normalized x GRF force', 'normalized y GRF force', 'normalized z GRF force', 'toe offs']

    # Initialize DataFrames for R_gc_grf_ges_MW and L_gc_grf_ges_MW
    gc_grf_ges_MW = pd.DataFrame(columns=columns_grf)

    for n in range(gz_counter_ges):
        grf_old_x = gc_grf_ges.iloc[n, 2]
        grf_old_y = gc_grf_ges.iloc[n, 3]
        grf_old_z = gc_grf_ges.iloc[n, 4]
        grf_old_norm_x = gc_grf_ges.iloc[n, 5]
        grf_old_norm_y = gc_grf_ges.iloc[n, 6]
        grf_old_norm_z = gc_grf_ges.iloc[n, 7]
        strikes_curr = gc_grf_ges.iloc[n, 8]

        x_range_old = np.linspace(np.min(grf_old_x), np.max(grf_old_x), len(grf_old_x))
        x_range_new = np.linspace(np.min(grf_old_x), np.max(grf_old_x), 201)
        y_range_old = np.linspace(np.min(grf_old_y), np.max(grf_old_y), len(grf_old_y))
        y_range_new = np.linspace(np.min(grf_old_y), np.max(grf_old_y), 201)

        grf_new_x = interp1d(x_range_old, grf_old_x, kind='cubic', fill_value='extrapolate')(x_range_new)
        grf_new_y = interp1d(y_range_old, grf_old_y, kind='cubic', fill_value='extrapolate')(y_range_new)
        grf_new_z = interp1d(x_range_old, grf_old_z, kind='cubic', fill_value='extrapolate')(x_range_new)
        grf_new_norm_x = interp1d(x_range_old, grf_old_norm_x, kind='cubic', fill_value='extrapolate')(x_range_new)
        grf_new_norm_y = interp1d(y_range_old, grf_old_norm_y, kind='cubic', fill_value='extrapolate')(y_range_new)
        grf_new_norm_z = interp1d(x_range_old, grf_old_norm_z, kind='cubic', fill_value='extrapolate')(x_range_new)

        gc_grf_ges_MW.loc[n] = ["", np.arange(0, 100.5, 0.5), grf_new_x, grf_new_y, grf_new_z, grf_new_norm_x, grf_new_norm_y, grf_new_norm_z, gc_grf_ges.iloc[n, 9] / strikes_curr[1] * 100]

        print()

        gc_grf_ges_MW.loc[gz_counter_ges] = ["MW", None] + [np.mean(gc_grf_ges_MW.iloc[n, i]) for i in range(2, len(columns_grf))]
        gc_grf_ges_MW.loc[gz_counter_ges+1] = ["MW-std", None] + [np.mean(gc_grf_ges_MW.iloc[n, i])-np.std(gc_grf_ges_MW.iloc[n, i]) for i in range(2, len(columns_grf))]   
        gc_grf_ges_MW.loc[gz_counter_ges+2] = ["MW+std", None] + [np.mean(gc_grf_ges_MW.iloc[n, i])+np.std(gc_grf_ges_MW.iloc[n, i]) for i in range(2, len(columns_grf))]

    return gc_grf_ges_MW
