import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Function to construct the DataFrame
def construct_akh_ges(gz_counter, num_gc_ges, gc_xvec, Strike, ToeOff, fto_gc, trial, force, mom, mass_Korr, prefix):
    """
    This function reads and prepares the Ankle, Knee, Hip Moments and Forces for the right and left foot.

    args: gz_counter: list, num_gc_ges: int, gc_xvec: DataFrame, Strike: list, ToeOff: list, fto_gc: int, trial: str, force: dict, mom: dict, mass_Korr: float

    return: akh_ges: DataFrame
    """
    columns_akh = ['Trial', 'x-Werte', '1 (y) Ankle Force', '2 (x) Ankle Force', '3 (z) Ankle Force',
               '1 (y) Knee Force', '2 (x) Knee Force', '3 (z) Knee Force',
               '1 (y) Hip Force', '2 (x) Hip Force', '3 (z) Hip Force',
               '1 (y) Waist Force', '2 (x) Waist Force', '3 (z) Waist Force',
               '1 (y) Neck Force', '2 (x) Neck Force', '3 (z) Neck Force',
               '1 (y) Shoulder Force', '2 (x) Shoulder Force', '3 (z) Shoulder Force',
               '1 (y) Elbow Force', '2 (x) Elbow Force', '3 (z) Elbow Force',
               '1 (y) Wrist Force', '2 (x) Wrist Force', '3 (z) Wrist Force',
               '1 (y) Ankle Moment', '2 (x) Ankle Moment', '3 (z) Ankle Moment',
               '1 (y) Knee Moment', '2 (x) Knee Moment', '3 (z) Knee Moment',
               '1 (y) Hip Moment', '2 (x) Hip Moment', '3 (z) Hip Moment',
               '1 (y) Waist Moment', '2 (x) Waist Moment', '3 (z) Waist Moment',
               '1 (y) Neck Moment', '2 (x) Neck Moment', '3 (z) Neck Moment',
               '1 (y) Shoulder Moment', '2 (x) Shoulder Moment', '3 (z) Shoulder Moment',
               '1 (y) Elbow Moment', '2 (x) Elbow Moment', '3 (z) Elbow Moment',
               '1 (y) Wrist Moment', '2 (x) Wrist Moment', '3 (z) Wrist Moment',
               'strikes', 'toe offs', 'frames x'
            ]
    
    data = []
    if f'{prefix}GroundReactionForce' in force:
        for i in range(len(gz_counter)):
            trial_nn = f"{trial}_{gz_counter[i]}"
            row_index = num_gc_ges + i

            # Initialize a row with Nones
            row = [None] * len(columns_akh)

            row[0] = trial_nn
            row[1] = gc_xvec.iloc[num_gc_ges + gz_counter[i], 3]  # 'x-Werte'

            row[2] = force[f'{prefix}AnkleForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[3] = force[f'{prefix}AnkleForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[4] = force[f'{prefix}AnkleForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[5] = force[f'{prefix}KneeForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[6] = force[f'{prefix}KneeForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[7] = force[f'{prefix}KneeForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[8] = force[f'{prefix}HipForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[9] = force[f'{prefix}HipForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[10] = force[f'{prefix}HipForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[11] = force[f'{prefix}WaistForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[12] = force[f'{prefix}WaistForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[13] = force[f'{prefix}WaistForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            if f'{prefix}NeckForce' in force:
                row[14] = force[f'{prefix}NeckForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
                row[15] = force[f'{prefix}NeckForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
                row[16] = force[f'{prefix}NeckForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr
                row[38] = mom[f'{prefix}NeckMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
                row[39] = mom[f'{prefix}NeckMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
                row[40] = mom[f'{prefix}NeckMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[17] = force[f'{prefix}ShoulderForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[18] = force[f'{prefix}ShoulderForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[19] = force[f'{prefix}ShoulderForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[20] = force[f'{prefix}ElbowForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[21] = force[f'{prefix}ElbowForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[22] = force[f'{prefix}ElbowForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[23] = force[f'{prefix}WristForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[24] = force[f'{prefix}WristForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[25] = force[f'{prefix}WristForce'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[26] = mom[f'{prefix}AnkleMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[27] = mom[f'{prefix}AnkleMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[28] = mom[f'{prefix}AnkleMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[29] = mom[f'{prefix}KneeMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[30] = mom[f'{prefix}KneeMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[31] = mom[f'{prefix}KneeMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[32] = mom[f'{prefix}HipMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[33] = mom[f'{prefix}HipMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[34] = mom[f'{prefix}HipMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[35] = mom[f'{prefix}WaistMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[36] = mom[f'{prefix}WaistMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[37] = mom[f'{prefix}WaistMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[41] = mom[f'{prefix}ShoulderMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[42] = mom[f'{prefix}ShoulderMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[43] = mom[f'{prefix}ShoulderMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[44] = mom[f'{prefix}ElbowMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[45] = mom[f'{prefix}ElbowMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[46] = mom[f'{prefix}ElbowMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[47] = mom[f'{prefix}WristMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 0] * mass_Korr
            row[48] = mom[f'{prefix}WristMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 1] * mass_Korr
            row[49] = mom[f'{prefix}WristMoment'][gc_xvec.iloc[num_gc_ges + gz_counter[i], 1], 2] * mass_Korr

            row[50] = (Strike[gz_counter[i]:gz_counter[i]+2] - Strike[gz_counter[i]] + 1).tolist()
            row[51] = ToeOff[fto_gc - 1 + gz_counter[i]] - Strike[gz_counter[i]] + 1
            row[52] = gc_xvec.iloc[num_gc_ges + gz_counter[i], 1] - Strike[gz_counter[i]] + 1

            # Append the row to the data list
            data.append(row)

    # Convert the list to a DataFrame
    akh_ges = pd.DataFrame(data, columns=columns_akh)
    return akh_ges

# Interpolation AKH Kräfte und Momente für Mittelwertskurve
def process_akh_data(gc_akh_ges, gz_counter_ges):
    """
    This function further processes the Ankle, Knee, Hip Moments and Forces for the right and left foot.
    Also calculates the mean and standard deviation of the Ankle, Knee, Hip Moments and Forces.

    args: gc_akh_ges: DataFrame, gz_counter_ges: int

    return: gc_akh_ges_MW: DataFrame
    """

    columns_akh_mw = ['label', 'x-Werte', '1 (y) Ankle Force', '2 (x) Ankle Force', '3 (z) Ankle Force',
                      '1 (y) Knee Force', '2 (x) Knee Force', '3 (z) Knee Force', '1 (y) Hip Force',
                      '2 (x) Hip Force', '3 (z) Hip Force', '1 (y) Waist Force', '2 (x) Waist Force',
                      '3 (z) Waist Force', '1 (y) Neck Force', '2 (x) Neck Force', '3 (z) Neck Force',
                      '1 (y) Shoulder Force', '2 (x) Shoulder Force', '3 (z) Shoulder Force', '1 (y) Elbow Force',
                      '2 (x) Elbow Force', '3 (z) Elbow Force', '1 (y) Wrist Force', '2 (x) Wrist Force',
                      '3 (z) Wrist Force', '1 (y) Ankle Moment', '2 (x) Ankle Moment', '3 (z) Ankle Moment',
                      '1 (y) Knee Moment', '2 (x) Knee Moment', '3 (z) Knee Moment', '1 (y) Hip Moment',
                      '2 (x) Hip Moment', '3 (z) Hip Moment', '1 (y) Waist Moment', '2 (x) Waist Moment',
                      '3 (z) Waist Moment', '1 (y) Neck Moment', '2 (x) Neck Moment', '3 (z) Neck Moment',
                      '1 (y) Shoulder Moment', '2 (x) Shoulder Moment', '3 (z) Shoulder Moment', '1 (y) Elbow Moment',
                      '2 (x) Elbow Moment', '3 (z) Elbow Moment', '1 (y) Wrist Moment', '2 (x) Wrist Moment',
                      '3 (z) Wrist Moment', 'toe offs']

    # Initialize the DataFrame for gc_akh_ges_MW
    gc_akh_ges_MW = pd.DataFrame(columns=columns_akh_mw)

    # Add labels and initialize rows
    for i in range(gz_counter_ges + 3):
        row = [None] * len(columns_akh_mw)
        if i == gz_counter_ges:
            row[0] = 'MW'
        elif i == gz_counter_ges + 1:
            row[0] = 'MW-std'
        elif i == gz_counter_ges + 2:
            row[0] = 'MW+std'
        gc_akh_ges_MW.loc[i] = row

    # Setting interpolation axis
    gc_akh_ges_MW.iloc[0, 1] = np.linspace(0, 100, num=201).tolist()  # Achse auf die die Werte interpoliert werden

    # Interpolating and calculating mean and standard deviation
    for i in range(2, 50):
        if gc_akh_ges.iloc[0, i] is not None:
            for n in range(gz_counter_ges):
                strikes_curr = np.array(gc_akh_ges.iloc[n, 50])
                gc_akh_ges_MW.iloc[n, 50] = gc_akh_ges.iloc[n, 51] / strikes_curr[1] * 100
                time_curr_norm_old = np.array(gc_akh_ges.iloc[n, 1])
                R_akh_old = np.array(gc_akh_ges.iloc[n, i])
                gc_akh_ges_MW.iloc[n, i] = interp1d(time_curr_norm_old, R_akh_old, kind='cubic', fill_value='extrapolate')(np.array(gc_akh_ges_MW.iloc[0, 1]))

            mean_value = np.mean(gc_akh_ges_MW.iloc[:gz_counter_ges, i].to_numpy()[0])
            std_value = np.std(gc_akh_ges_MW.iloc[:gz_counter_ges, i].to_numpy()[0])

            gc_akh_ges_MW.iloc[gz_counter_ges, i] = mean_value
            gc_akh_ges_MW.iloc[gz_counter_ges + 1, i] = mean_value - std_value
            gc_akh_ges_MW.iloc[gz_counter_ges + 2, i] = mean_value + std_value

            mean_strikes = np.mean(gc_akh_ges_MW.iloc[:gz_counter_ges, 50].to_numpy()[0])
            std_strikes = np.std(gc_akh_ges_MW.iloc[:gz_counter_ges, 50].to_numpy()[0])

            gc_akh_ges_MW.iloc[gz_counter_ges, 50] = mean_strikes
            gc_akh_ges_MW.iloc[gz_counter_ges + 1, 50] = mean_strikes - std_strikes
            gc_akh_ges_MW.iloc[gz_counter_ges + 2, 50] = mean_strikes + std_strikes

    return gc_akh_ges_MW
