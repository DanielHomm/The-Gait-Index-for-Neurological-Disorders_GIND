import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def construct_gc_com_ges(gc_xvec, num_gc, Strike, ToeOff, trial, mark, fto_gc):
    """
    This function reads and prepares the center of mass data for the right and left foot.

    args: gc_xvec: DataFrame, num_gc: int, Strike: list, ToeOff: list, trial: str, mark: dict, fto_gc: int

    return: com_ges: DataFrame
    """
    
    columns_com = ['Trial', 'x-Werte', 'x CoM', 'y CoM', 'z CoM', 'strikes', 'toe offs', 'frames x']
    com_ges = pd.DataFrame(columns=columns_com)

    for i in range(num_gc):
        trial_nn = f"{trial}_{i+1}"
        row = [None] * len(columns_com)
        row[0] = trial_nn
        row[1] = gc_xvec.iloc[i, 3]

        com_x_curr = mark['CentreOfMass'][gc_xvec.iloc[i, 1], 0] - mark['CentreOfMass'][gc_xvec.iloc[i, 1], 0][0]
        com_y_curr = mark['CentreOfMass'][gc_xvec.iloc[i, 1], 1] - mark['CentreOfMass'][gc_xvec.iloc[i, 1], 1][0]

        com_length = len(com_y_curr)

        if com_y_curr[com_length - 1] >= 0:
            row[3] = com_y_curr
            row[2] = -com_x_curr
        else:
            row[3] = -com_y_curr
            row[2] = com_x_curr

        row[4] = mark['CentreOfMass'][gc_xvec.iloc[i, 1], 2]
        row[5] = (Strike[i:i+2] - Strike[i] + 1).tolist()
        row[6] = ToeOff[fto_gc - 1 + i] - Strike[i] + 1
        row[7] = gc_xvec.iloc[i, 1] - Strike[i] + 1

        com_ges.loc[i] = row

    return com_ges


def process_com_data(com_ges, num_gc_ges):
    """
    This function further processes the center of mass data for the right and left foot.
    Also calculates the mean and standard deviation of the center of mass data.

    args: com_ges: DataFrame, num_gc_ges: int

    return: com_ges_MW: DataFrame
    """

    columns_com_MW = ['Label', 'x-Werte', 'x CoM', 'y CoM', 'z CoM', 'toe offs']
    com_ges_MW = pd.DataFrame(columns=columns_com_MW)

    # Initialize the DataFrame
    for i in range(num_gc_ges + 3):
        row = [''] * len(columns_com_MW)
        if i == num_gc_ges:
            row[0] = 'MW'
        elif i == num_gc_ges + 1:
            row[0] = 'MW-std'
        elif i == num_gc_ges + 2:
            row[0] = 'MW+std'
        com_ges_MW.loc[i] = row

    # Setting interpolation axis
    com_ges_MW.iloc[0, 1] = np.linspace(0, 100, num=201).tolist()  # Achse auf die die Werte interpoliert werden

    # Interpolating and calculating mean and standard deviation
    for i in range(2, 5):
        interpolated_values = []
        for n in range(num_gc_ges):
            strikes_curr = np.array(com_ges.iloc[n, 5])
            com_ges_MW.iloc[n, 5] = com_ges.iloc[n, 6] / strikes_curr[1] * 100
            time_curr_norm_old = np.array(com_ges.iloc[n, 1])
            com_old = np.array(com_ges.iloc[n, i])
            interpolated = interp1d(time_curr_norm_old, com_old, kind='cubic', fill_value='extrapolate')(np.array(com_ges_MW.iloc[0, 1]))
            com_ges_MW.iloc[n, i] = interpolated
            interpolated_values.append(interpolated)

        mean_value = np.mean(interpolated_values[0], axis=0)
        std_value = np.std(interpolated_values[0], axis=0)

        com_ges_MW.iloc[num_gc_ges, i] = mean_value
        com_ges_MW.iloc[num_gc_ges + 1, i] = mean_value - std_value
        com_ges_MW.iloc[num_gc_ges + 2, i] = mean_value + std_value

    toe_offs_values = [row for row in com_ges_MW.iloc[:num_gc_ges, 5] if isinstance(row, (int, float))]
    mean_strikes = np.mean(toe_offs_values)
    std_strikes = np.std(toe_offs_values)
    com_ges_MW.iloc[num_gc_ges, 5] = mean_strikes
    com_ges_MW.iloc[num_gc_ges + 1, 5] = mean_strikes - std_strikes
    com_ges_MW.iloc[num_gc_ges + 2, 5] = mean_strikes + std_strikes

    return com_ges_MW
