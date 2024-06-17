import pandas as pd
import numpy as np

def construct_norm_gc_para():
    """
    This function constructs the normal gait cycle parameters DataFrame using predefined values.
    
    args: None

    returns: norm_gc_para: DataFrame
    """

    columns = [
        'Parameter', 'Standphase [%]', 'Doppelschrittlänge [Frames]', 'Doppelschrittlänge [mm], in Bezug auf RHEE',
        'Schrittlänge [mm]', 'Spurbreite [mm]', 'Einzelschrittzeit [s]', 'Doppelschrittzeit [s]', 'Einzelstützzeit [s]',
        'Doppelstützzeit [s]', 'Einzelstandphase [%]', 'Doppelstandphase [%]', 'Kontrolle Standzeiten, sollte 0 sein',
        'Ganggeschwindigkeit [m/s]', 'Kadenz [1/min]', 'Standzeit [s]', 'Schwungphase [%]'
    ]

    norm_gc_para = pd.DataFrame(columns=columns)

    # Add labels and initialize rows
    norm_gc_para.loc[0] = [''] + [''] * (len(columns) - 1)
    norm_gc_para.loc[1] = ['untere Grenze'] + [None] * (len(columns) - 1)
    norm_gc_para.loc[2] = ['obere Grenze'] + [None] * (len(columns) - 1)
    # Lower bounds
    norm_gc_para.iloc[1, 1:] = [62, 0, 1300, 550, 30, 0, 0, 0, 0, 38, 24, 0, 1.23, 100, 62, 38]

    # Upper bounds
    norm_gc_para.iloc[2, 1:] = [62, 0, 1500, 750, 130, 0, 0, 0, 0, 38, 24, 0, 1.39, 130, 62, 38]

    # Calculate mean and error for plot later
    for i in range(1, len(columns)):
        lower_bound = norm_gc_para.iloc[1, i]
        upper_bound = norm_gc_para.iloc[2, i]
        mean_value = np.mean([lower_bound, upper_bound])
        error_value = upper_bound - mean_value

        norm_gc_para.loc[3, columns[i]] = mean_value  # Einzuzeichnender MW
        norm_gc_para.loc[4, columns[i]] = error_value  # Einzuzeichnender err

    norm_gc_para.loc[3, 'Parameter'] = 'MW'
    norm_gc_para.loc[4, 'Parameter'] = 'err'

    return norm_gc_para