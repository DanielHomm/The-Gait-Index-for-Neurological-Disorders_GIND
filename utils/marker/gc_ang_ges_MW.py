import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def construct_gc_ang_ges(gc_xvec, num_gc_ges, Strike, ToeOff, ang, prefix, trial):
    """
    This function reads and prepares the angle data for the right and left foot.

    args: gc_xvec: DataFrame, num_gc_ges: int, Strike: list, ToeOff: list, ang: dict, prefix: str, trial: str

    return: gc_ang_ges: DataFrame
    """

    columns = [
        'label', 'x-Werte', 'Ankle 1 Flexion', 'Ankle 2 Inversion', 'Ankle 3 Rotation',
        'Knee 1 Flexion', 'Knee 2 Adduktion', 'Knee 3 Rotation', 'Hip 1 Flexion',
        'Hip 2 Adduktion', 'Hip 3 Rotation', 'Pelvis 1 Vorneigung', 'Pelvis 2 Seitneigung',
        'Pelvis 3 Rotation', 'Foot Progress 1 Flexion', 'Foot Progress 2 Inversion',
        'Foot Progress 3 Rotation', 'Ankle Abs 1', 'Ankle Abs 2', 'Ankle Abs 3',
        'Spine 1 Vorwärtsneigung Thorax', 'Spine 2 Rechtsneigung Thorax', 'Spine 3 Rechtsrotation Thorax',
        'Neck 1 Vorwärtsneigung', 'Neck 2 Rechtsneigung', 'Neck 3 Rechtsrotation', 'Shoulder 1 Flexion',
        'Shoulder 2 Adduktion', 'Shoulder 3 Rotation', 'Elbow 1 Flexion', 'Elbow 2 ', 'Elbow 3 ',
        'Wrist 1 Ulnardeviation ', 'Wrist 2 Ausstrecken', 'Wrist 3 Rotation', 'Thorax 1 Vorwärtsneigung',
        'Thorax 2 Linksneigung', 'Thorax 3 Linksrotation', 'Head 1 Rückneigung', 'Head 2 Linksneigung',
        'Head 3 Linksrotation', 'strikes', 'toe offs', 'frames'
    ]
    gc_ang_ges = pd.DataFrame(columns=columns)
    num_gc = len(Strike) - 1

    if f'{prefix}AnkleAngles' in ang:
        for i in range(num_gc):
            trial_nn = f"{trial}_{i}"
            row = [None] * len(columns)
            row[0] = trial_nn
            row[1] = gc_xvec.iloc[num_gc_ges + i, 3]
            row[2] = ang[f'{prefix}AnkleAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[3] = ang[f'{prefix}AnkleAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[4] = ang[f'{prefix}AnkleAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[5] = ang[f'{prefix}KneeAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[6] = ang[f'{prefix}KneeAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[7] = ang[f'{prefix}KneeAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[8] = ang[f'{prefix}HipAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[9] = ang[f'{prefix}HipAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[10] = ang[f'{prefix}HipAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[11] = ang[f'{prefix}PelvisAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[12] = ang[f'{prefix}PelvisAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[13] = ang[f'{prefix}PelvisAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[14] = ang[f'{prefix}FootProgressAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[15] = ang[f'{prefix}FootProgressAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[16] = ang[f'{prefix}FootProgressAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[17] = ang[f'{prefix}AbsAnkleAngle'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[18] = ang[f'{prefix}AbsAnkleAngle'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[19] = ang[f'{prefix}AbsAnkleAngle'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[20] = ang[f'{prefix}SpineAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[21] = ang[f'{prefix}SpineAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[22] = ang[f'{prefix}SpineAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[26] = ang[f'{prefix}ShoulderAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[27] = ang[f'{prefix}ShoulderAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[28] = ang[f'{prefix}ShoulderAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[29] = ang[f'{prefix}ElbowAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[30] = ang[f'{prefix}ElbowAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[31] = ang[f'{prefix}ElbowAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[32] = ang[f'{prefix}WristAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[33] = ang[f'{prefix}WristAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[34] = ang[f'{prefix}WristAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[35] = ang[f'{prefix}ThoraxAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
            row[36] = ang[f'{prefix}ThoraxAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
            row[37] = ang[f'{prefix}ThoraxAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            if f'{prefix}HeadAngles' in ang:
                row[38] = ang[f'{prefix}HeadAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
                row[39] = ang[f'{prefix}HeadAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
                row[40] = ang[f'{prefix}HeadAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
                row[23] = ang[f'{prefix}NeckAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 0]
                row[24] = ang[f'{prefix}NeckAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 1]
                row[25] = ang[f'{prefix}NeckAngles'][gc_xvec.iloc[num_gc_ges + i, 1], 2]
            row[41] = Strike[i:(i + 2)] - Strike[i] + 1
            row[42] = ToeOff[i - 1] - Strike[i] + 1
            row[43] = gc_xvec.iloc[num_gc_ges + i, 1] - Strike[i] + 1

            gc_ang_ges.loc[num_gc_ges + i] = row

    return gc_ang_ges

def process_ang_data(ang_ges, num_gc_ges):
    """
    This function further processes the angle data for the right and left foot.
    Also calculates the mean and standard deviation of the angle data.

    args: ang_ges: DataFrame, num_gc_ges: int

    return: ang_ges_MW: DataFrame
    """

    columns_ang_MW = [
        'Label', 'x-Werte', 'Ankle 1 Flexion', 'Ankle 2 Inversion', 'Ankle 3 Rotation',
        'Knee 1 Flexion', 'Knee 2 Adduktion', 'Knee 3 Rotation', 'Hip 1 Flexion', 'Hip 2 Adduktion',
        'Hip 3 Rotation', 'Pelvis 1 Vorneigung', 'Pelvis 2 Seitneigung', 'Pelvis 3 Rotation',
        'Foot Progress 1 Flexion', 'Foot Progress 2 Inversion', 'Foot Progress 3 Rotation',
        'Ankle Abs 1', 'Ankle Abs 2', 'Ankle Abs 3', 'Spine 1 Vorwärtsneigung Thorax',
        'Spine 2 Rechtsneigung Thorax', 'Spine 3 Rechtsrotation Thorax', 'Neck 1 Vorwärtsneigung',
        'Neck 2 Rechtsneigung', 'Neck 3 Rechtsrotation', 'Shoulder 1 Flexion', 'Shoulder 2 Adduktion',
        'Shoulder 3 Rotation', 'Elbow 1 Flexion', 'Elbow 2', 'Elbow 3', 'Wrist 1 Ulnardeviation',
        'Wrist 2 Ausstrecken', 'Wrist 3 Rotation', 'Thorax 1 Vorwärtsneigung', 'Thorax 2 Linksneigung',
        'Thorax 3 Linksrotation', 'Head 1 Rückneigung', 'Head 2 Linksneigung', 'Head 3 Linksrotation',
        'toe offs'
    ]
    ang_ges_MW = pd.DataFrame(columns=columns_ang_MW)

    # Initialize the DataFrame
    for i in range(num_gc_ges + 3):
        row = [''] * len(columns_ang_MW)
        if i == num_gc_ges:
            row[0] = 'MW'
        elif i == num_gc_ges + 1:
            row[0] = 'MW-std'
        elif i == num_gc_ges + 2:
            row[0] = 'MW+std'
        ang_ges_MW.loc[i] = row

    # Setting interpolation axis
    ang_ges_MW.iloc[0, 1] = np.linspace(0, 100, num=201).tolist()  # Achse auf die die Werte interpoliert werden

    # Determine the last angle column based on the existence of 'HeadAngles'
    last_angle = len(ang_ges.columns)

    # Interpolating and calculating mean and standard deviation
    for i in range(2, last_angle - 3):
        interpolated_values = []
        if not ang_ges.iloc[0, i] is None:
            for n in range(num_gc_ges):
                strikes_curr = np.array(ang_ges.iloc[n, 41])
                ang_ges_MW.iloc[n, 41] = ang_ges.iloc[n, 42] / strikes_curr[1] * 100
                time_curr_norm_old = np.array(ang_ges.iloc[n, 1])
                ang_old = np.array(ang_ges.iloc[n, i])
                interpolated = interp1d(time_curr_norm_old, ang_old, kind='cubic', fill_value='extrapolate')(np.array(ang_ges_MW.iloc[0, 1]))
                ang_ges_MW.iloc[n, i] = interpolated
                interpolated_values.append(interpolated)

            mean_value = np.mean(interpolated_values[0], axis=0)
            std_value = np.std(interpolated_values[0], axis=0)

            ang_ges_MW.iloc[num_gc_ges, i] = mean_value
            ang_ges_MW.iloc[num_gc_ges + 1, i] = mean_value - std_value
            ang_ges_MW.iloc[num_gc_ges + 2, i] = mean_value + std_value

    toe_offs_values = [row for row in ang_ges_MW.iloc[:num_gc_ges, 41] if isinstance(row, (int, float))]
    mean_strikes = np.mean(toe_offs_values)
    std_strikes = np.std(toe_offs_values)
    ang_ges_MW.iloc[num_gc_ges, 41] = mean_strikes
    ang_ges_MW.iloc[num_gc_ges + 1, 41] = mean_strikes - std_strikes
    ang_ges_MW.iloc[num_gc_ges + 2, 41] = mean_strikes + std_strikes

    return ang_ges_MW
