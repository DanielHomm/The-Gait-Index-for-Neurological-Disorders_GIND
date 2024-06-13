# Lin 398
import numpy as np
import pandas as pd

def construct_gc_para(gc_xvec, num_gc, Strike, ToeOff, trial, mark, fto_gc, other_Strike, other_ToeOff, mark_freq, side='R'):
    # Initialize the columns
    columns_para = [
        'Trial', 'Standphase [%]', 'Doppelschrittlänge [Frames]', 'Doppelschrittlänge [mm], in Bezug auf HEEL',
        'Schrittlänge [mm]', 'Spurbreite [mm]', 'Einzelschrittzeit [s]', 'Doppelschrittzeit [s]',
        'Einzelstützzeit [s]', 'Doppelstützzeit [s]', 'Einzelstandphase [%]', 'Doppelstandphase [%]',
        'Kontrolle Standzeiten, sollte 0 sein', 'Ganggeschwindigkeit [m/s]', 'Kadenz [1/min]',
        'Schwungphase [%]', 'Schwungzeit [s]', 'Standzeit [s]', 'Verhältnis Schwung- zu Standzeit'
    ]

    gc_para = pd.DataFrame(columns=columns_para)

    # Determine which Strike and ToeOff of the other foot are relevant
    if Strike[0] < other_Strike[0]:
        f_gc_other = 0
    else:
        f_gc_other = 1

    if Strike[0] < other_ToeOff[0]:
        f_to_other = 0
    else:
        f_to_other = 1

    for i in range(num_gc):
        trial_nn = f"{trial}_{i+1}"
        row = [None] * len(columns_para)
        row[0] = trial_nn

        # Standphase [%]
        row[1] = (ToeOff[fto_gc + i - 1] - Strike[i]) / (Strike[i + 1] - Strike[i]) * 100

        # Doppelschrittlänge [Frames]
        row[2] = Strike[i + 1] - Strike[i]

        # Doppelschrittlänge [mm], in Bezug auf HEEL
        heel = 'RHEE' if side == 'R' else 'LHEE'
        row[3] = abs(mark[heel][Strike[i + 1], 1] - mark[heel][Strike[i], 1])

        # Schrittlänge [mm]
        other_heel = 'LHEE' if side == 'R' else 'RHEE'
        row[4] = abs(mark[heel][Strike[i + 1], 1] - mark[other_heel][other_Strike[f_gc_other + i], 1])

        # Spurbreite [mm]
        row[5] = abs(mark[heel][Strike[i + 1], 0] - mark[other_heel][other_Strike[f_gc_other + i], 0])

        # Einzelschrittzeit [s]
        row[6] = (Strike[i + 1] - other_Strike[f_gc_other + i]) / mark_freq

        # Doppelschrittzeit [s]
        row[7] = (Strike[1 + i] - Strike[i]) / mark_freq

        # Einzelstützzeit [s]
        row[8] = (other_Strike[f_gc_other + i] - other_ToeOff[f_to_other + i]) / mark_freq

        # Doppelstützzeit [s]
        row[9] = (other_ToeOff[f_to_other + i] - Strike[i] + ToeOff[fto_gc + i - 1] - other_Strike[f_gc_other + i]) / mark_freq

        # Einzelstandphase [%]
        row[10] = (other_Strike[f_gc_other + i] - other_ToeOff[f_to_other + i]) / (Strike[i + 1] - Strike[i]) * 100

        # Doppelstandphase [%]
        row[11] = (other_ToeOff[f_to_other + i] - Strike[i] + ToeOff[fto_gc + i - 1] - other_Strike[f_gc_other + i]) / (Strike[i + 1] - Strike[i]) * 100

        # Kontrolle Standzeiten, sollte 0 sein
        row[12] = row[10] + row[11] - row[1]

        # Ganggeschwindigkeit [m/s]
        row[13] = 0.001 * abs(mark[heel][Strike[i + 1], 1] - mark[heel][Strike[i], 1]) / ((Strike[1 + i] - Strike[i]) / mark_freq)

        # Kadenz [1/min]
        row[14] = 60 / ((Strike[1 + i] - Strike[i]) / mark_freq) * 2

        # Schwungphase [%]
        row[15] = 100 - row[1]

        # Schwungzeit [s]
        row[16] = (Strike[i + 1] - ToeOff[fto_gc + i - 1]) / mark_freq

        # Standzeit [s]
        row[17] = (ToeOff[fto_gc + i - 1] - Strike[i]) / mark_freq

        # Verhältnis Schwung- zu Standzeit
        row[18] = row[16] / row[17]

        gc_para.loc[i] = row

    num_par = 19

    # Mittelwert and Standardabweichung
    mean_row = ['Mittelwert'] + gc_para.iloc[:, 1:num_par].mean().tolist()
    std_row = ['Standardabweichung'] + gc_para.iloc[:, 1:num_par].std().tolist()

    gc_para.loc[num_gc] = mean_row
    gc_para.loc[num_gc + 1] = std_row

    return gc_para

# Example usage

print("R_gc_para and L_gc_para are created successfully.")
