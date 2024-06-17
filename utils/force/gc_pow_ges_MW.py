import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def construct_pow_ges(gz_counter, num_gc_ges, gc_xvec, Strike, ToeOff, fto_gc, trial, power, mass_Korr, prefix):
    """
    This function reads and prepares the Ankle, Knee, Hip Power data for the right and left foot.

    args: gz_counter: list, num_gc_ges: int, gc_xvec: DataFrame, Strike: list, ToeOff: list, fto_gc: int, trial: str, power: dict, mass_Korr: float, prefix: str

    return: pow_ges: DataFrame
    """
    columns_pow = ['Trial', 'x-Werte', 'Ankle Power', 'Knee Power', 'Hip Power', 'Waist Power',
               'Shoulder Power', 'Elbow Power', 'Wrist Power', 'strikes', 'toe offs', 'frames x']
    
    data = []
    if f'{prefix}AnklePower' in power:
        for i in range(len(gz_counter)):
            trial_nn = f"{trial}_{gz_counter[i]}"
            row_index = num_gc_ges + i

            # Initialize a row with Nones
            row = [None] * len(columns_pow)

            row[0] = trial_nn
            row[1] = gc_xvec.iloc[num_gc_ges + gz_counter[i], 3]  # 'x-Werte'

            row[2] = power[f'{prefix}AnklePower'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
            row[3] = power[f'{prefix}KneePower'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
            row[4] = power[f'{prefix}HipPower'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
            row[5] = power[f'{prefix}WaistPower'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
            row[6] = power[f'{prefix}ShoulderPower'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
            row[7] = power[f'{prefix}ElbowPower'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
            row[8] = power[f'{prefix}WristPower'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[9] = (Strike[gz_counter[i]:gz_counter[i]+2] - Strike[gz_counter[i]] + 1).tolist()
            row[10] = ToeOff[fto_gc - 1 + gz_counter[i]] - Strike[gz_counter[i]] + 1
            row[11] = gc_xvec.iloc[num_gc_ges + gz_counter[i], 1] - Strike[gz_counter[i]] + 1

            # Append the row to the data list
            data.append(row)

    # Convert the list to a DataFrame
    pow_ges = pd.DataFrame(data, columns=columns_pow)
    return pow_ges

def process_pow_data(gc_pow_ges, gz_counter_ges):
    """
    This function further processes the Ankle, Knee, Hip Power data for the right and left foot.
    Also calculates the mean and standard deviation of the Ankle, Knee, Hip Power data.

    args: gc_pow_ges: DataFrame, gz_counter_ges: int

    return: gc_pow_ges_MW: DataFrame
    """

    columns_pow_MW = ['label', 'x-Werte', 'Ankle Power', 'Knee Power', 'Hip Power', 'Waist Power',
                      'Shoulder Power', 'Elbow Power', 'Wrist Power', 'toe offs']

    # Initialize the DataFrame for gc_pow_ges_MW
    gc_pow_ges_MW = pd.DataFrame(columns=columns_pow_MW)

    # Add labels and initialize rows
    for i in range(gz_counter_ges + 3):
        row = [None] * len(columns_pow_MW)
        if i == gz_counter_ges:
            row[0] = 'MW'
        elif i == gz_counter_ges + 1:
            row[0] = 'MW-std'
        elif i == gz_counter_ges + 2:
            row[0] = 'MW+std'
        gc_pow_ges_MW.loc[i] = row

    # Setting interpolation axis
    gc_pow_ges_MW.iloc[0, 1] = np.linspace(0, 100, num=201).tolist()  # Achse auf die die Werte interpoliert werden

    # Interpolating and calculating mean and standard deviation
    for i in range(2, 9):
        if gc_pow_ges.iloc[0, i] is not None:
            for n in range(gz_counter_ges):
                strikes_curr = np.array(gc_pow_ges.iloc[n, 9])
                gc_pow_ges_MW.iloc[n, 9] = gc_pow_ges.iloc[n, 10] / strikes_curr[1] * 100
                time_curr_norm_old = np.array(gc_pow_ges.iloc[n, 1])
                R_pow_old = np.array(gc_pow_ges.iloc[n, i])
                gc_pow_ges_MW.iloc[n, i] = interp1d(time_curr_norm_old, R_pow_old, kind='cubic', fill_value='extrapolate')(np.array(gc_pow_ges_MW.iloc[0, 1]))

            mean_value = np.mean(gc_pow_ges_MW.iloc[:gz_counter_ges, i].to_numpy()[0])
            std_value = np.std(gc_pow_ges_MW.iloc[:gz_counter_ges, i].to_numpy()[0])

            gc_pow_ges_MW.iloc[gz_counter_ges, i] = mean_value
            gc_pow_ges_MW.iloc[gz_counter_ges + 1, i] = mean_value - std_value
            gc_pow_ges_MW.iloc[gz_counter_ges + 2, i] = mean_value + std_value

            mean_strikes = np.mean(gc_pow_ges_MW.iloc[:gz_counter_ges, 9].to_numpy()[0])
            std_strikes = np.std(gc_pow_ges_MW.iloc[:gz_counter_ges, 9].to_numpy()[0])

            gc_pow_ges_MW.iloc[gz_counter_ges, 9] = mean_strikes
            gc_pow_ges_MW.iloc[gz_counter_ges + 1, 9] = mean_strikes - std_strikes
            gc_pow_ges_MW.iloc[gz_counter_ges + 2, 9] = mean_strikes + std_strikes

    return gc_pow_ges_MW
