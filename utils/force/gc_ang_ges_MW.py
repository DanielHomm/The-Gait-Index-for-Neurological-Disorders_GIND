import pandas as pd

def construct_gc_ang_ges(gc_xvec, num_gc_ges, Strike, ToeOff, ang, prefix, trial):
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