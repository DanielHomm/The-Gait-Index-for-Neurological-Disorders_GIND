import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def construct_gc_ang_bew_ges(ang_ges, num_gc_ges, trial):
    columns_bew = [
        'Trial', '', 'Ankle 1 Flexion', 'Ankle 2 Inversion', 'Ankle 3 Rotation',
        'Knee 1 Flexion', 'Knee 2 Adduktion', 'Knee 3 Rotation', 'Hip 1 Flexion', 'Hip 2 Adduktion',
        'Hip 3 Rotation', 'Pelvis 1 Vorneigung', 'Pelvis 2 Seitneigung', 'Pelvis 3 Rotation',
        'Foot Progress 1 Flexion', 'Foot Progress 2 Inversion', 'Foot Progress 3 Rotation',
        'Ankle Abs 1', 'Ankle Abs 2', 'Ankle Abs 3', 'Spine 1 Vorwärtsneigung Thorax',
        'Spine 2 Rechtsneigung Thorax', 'Spine 3 Rechtsrotation Thorax', 'Neck 1 Vorwärtsneigung',
        'Neck 2 Rechtsneigung', 'Neck 3 Rechtsrotation', 'Shoulder 1 Flexion', 'Shoulder 2 Adduktion',
        'Shoulder 3 Rotation', 'Elbow 1 Flexion', 'Elbow 2', 'Elbow 3', 'Wrist 1 Ulnardeviation',
        'Wrist 2 Ausstrecken', 'Wrist 3 Rotation', 'Thorax 1 Vorwärtsneigung', 'Thorax 2 Linksneigung',
        'Thorax 3 Linksrotation', 'Head 1 Rückneigung', 'Head 2 Linksneigung', 'Head 3 Linksrotation'
    ]
    ang_bew_ges = pd.DataFrame(columns=columns_bew)

    # Initialize the DataFrame with trial numbers
    for i in range(num_gc_ges):
        trial_nn = f"{trial}_{i+1}"
        row = [None] * len(columns_bew)
        row[0] = trial_nn
        ang_bew_ges.loc[i] = row

    # Calculate the range of motion for each joint angle
    for i in range(num_gc_ges):
        for n in range(2, len(columns_bew)):
            ang_bew_ges.iloc[i, n] = np.linalg.norm(
                np.max(ang_ges.iloc[i, n]) - np.min(ang_ges.iloc[i, n])
            )

    # Adding rows for Mean and Standard Deviation
    mw_row = ['MW'] + [None] * (len(columns_bew) - 1)
    std_row = ['Std'] + [None] * (len(columns_bew) - 1)
    ang_bew_ges.loc[num_gc_ges] = mw_row
    ang_bew_ges.loc[num_gc_ges + 1] = std_row

    # Calculate Mean and Standard Deviation for each joint angle
    for n in range(2, len(columns_bew)):
        values = ang_bew_ges.iloc[:num_gc_ges, n].dropna().astype(float)
        ang_bew_ges.iloc[num_gc_ges, n] = values.mean()
        ang_bew_ges.iloc[num_gc_ges + 1, n] = values.std()

    return ang_bew_ges
