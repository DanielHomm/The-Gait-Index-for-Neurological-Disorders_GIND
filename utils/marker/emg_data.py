import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import numpy as np

def construct_emg_roh(trial, Strike, ToeOff, ana, num_gc, mark_freq, prefix):
    if num_gc < 1:
        return pd.DataFrame()

    columns_emg = [
        'Trial', 'x-Werte Frames', 'x-Werte [s]', 'GM L', 'GM R', 'RF L', 'RF R',
        'BF L', 'BF R', 'Tensor L', 'Tensor R', 'Number Strikes', 'Times strikes', 'Times toe offs'
    ]
    emg_roh = pd.DataFrame(columns=columns_emg)

    frames = np.arange(5 * Strike[0] -1, 5 * Strike[-1])
    time_seconds = (frames + 1) / (5 * mark_freq)

    row = [None] * len(columns_emg)
    row[0] = trial
    row[1] = frames
    row[2] = time_seconds
    row[3] = ana[f'EMG.GM_L'][frames]
    row[4] = ana[f'EMG.GM_R'][frames]
    row[5] = ana[f'EMG.RF_L'][frames]
    row[6] = ana[f'EMG.RF_R'][frames]
    row[7] = ana[f'EMG.BF_L'][frames]
    row[8] = ana[f'EMG.BF_R'][frames]

    if f'EMG.Tensor_L' in ana:
        row[9] = ana[f'EMG.Tensor_L'][frames]
        row[10] = ana[f'EMG.Tensor_R'][frames]
    elif f'EMG.TF_L' in ana:
        row[9] = ana[f'EMG.TF_L'][frames]
        row[10] = ana[f'EMG.TF_R'][frames]

    row[11] = num_gc
    row[12] = Strike / mark_freq
    row[13] = ToeOff / mark_freq

    emg_roh.loc[0] = row

    return emg_roh


def butter_filter(data, cutoff, btype, order=2):
    b, a = butter(order, cutoff, btype=btype)
    return filtfilt(b, a, data)

def construct_emg_bear(emg_roh, fcuthigh, fcutlow, fcutenv, force_sr=1000, n=0):
    emg_bear = emg_roh.copy()
    force_sr = force_sr

    # Nyquist frequency
    fnyq = force_sr / 2

    Wn = [fcuthigh / fnyq, fcutlow / fnyq, fcutenv / fnyq]
    processed_data = []

    row = emg_roh.iloc[n].copy()
    for i in range(3, 11):
        emg_data = row[i]

        if isinstance(emg_data, np.ndarray):
            # High-pass filter
            b, a = butter(2, Wn[0], btype="high")
            bear_curr = filtfilt(b, a, emg_data)
            # Low-pass filter
            emg_data_lp = butter_filter(bear_curr, Wn[1], 'low')

            # Rectification
            emg_data_rec = np.abs(emg_data_lp - np.mean(emg_data_lp))

            # Linear envelope using low-pass filter
            emg_data_env = butter_filter(emg_data_rec, Wn[2], 'low')
            # Store the processed data
            row[i] = emg_data_env.tolist()
        else:
            print(i, " is not an array")

    processed_data.append(row)
    emg_bear = pd.DataFrame(processed_data, columns=emg_roh.columns)

    return emg_bear

def construct_emg_new(emg_bear, mark_freq, gz_counter_ges):
    # Define the columns for the DataFrame
    columns = ['Label', 'x-Werte normalisiert [s]', 'GM L', 'GM R', 'RF L', 'RF R', 'BF L', 'BF R', 'Tensor L', 'Tensor R', 'Times toe offs']
    emg_new = pd.DataFrame(columns=columns)
    
    # Initialize the DataFrame and set labels
    for i in range(gz_counter_ges + 3):
        row = [None] * len(columns)
        if i == gz_counter_ges:
            row[0] = 'MW'
        elif i == gz_counter_ges + 1:
            row[0] = 'MW-std'
        elif i == gz_counter_ges + 2:
            row[0] = 'MW+std'
        emg_new.loc[i] = row

    # Set the interpolation axis
    time_curr_norm_new = np.arange(0, 100.5, 0.5)
    emg_new.iloc[0, 1] = time_curr_norm_new

    trial_number = len(emg_bear)
    
    # Interpolate and populate the DataFrame
    for i in range(2, len(columns)):
        for n in range(trial_number):
            strikes_curr = np.round(np.array(emg_bear.iloc[n, 12]) * mark_freq * 5).astype(int)
            toeoff_curr = np.round(np.array(emg_bear.iloc[n, 13]) * mark_freq * 5).astype(int)
            for r in range(len(strikes_curr) - 1):
                time_curr = np.arange(strikes_curr[r], strikes_curr[r+1])
                time_curr_length = len(time_curr) - 1
                time_curr_norm_old = (time_curr - strikes_curr[r]) / time_curr_length * 100
                emg_old = np.array(emg_bear.iloc[n, i])
                frames_curr = time_curr - strikes_curr[0] + 1
                #interp_func = interp1d(time_curr_norm_old, emg_old[frames_curr-1], kind='cubic', fill_value='extrapolate')
                interp_func = interp1d(time_curr_norm_old, emg_old[frames_curr], kind='cubic', fill_value='extrapolate')
                if i < len(columns) - 1:
                    emg_new.iloc[r, i] = interp_func(time_curr_norm_new)
                else:
                    if toeoff_curr[r] > strikes_curr[r]:
                        toeoff_norm_old = (toeoff_curr[r] - strikes_curr[r]) / time_curr_length * 100
                    else:
                        toeoff_norm_old = (toeoff_curr[r+1] - strikes_curr[r]) / time_curr_length * 100
                    emg_new.iloc[r, len(columns) - 1] = toeoff_norm_old

        mean_value = np.mean(emg_new.iloc[:gz_counter_ges, i].to_numpy()[0])
        std_value = np.std(emg_new.iloc[:gz_counter_ges, i].to_numpy()[0])

        emg_new.iloc[gz_counter_ges, i] = mean_value
        emg_new.iloc[gz_counter_ges + 1, i] = mean_value - std_value
        emg_new.iloc[gz_counter_ges + 2, i] = mean_value + std_value

        #mean_col = np.mean(emg_new.iloc[:gz_counter_ges, i])
        #std_col = np.std(emg_new.iloc[:gz_counter_ges, i])

        #emg_new.iloc[gz_counter_ges, i] = mean_col
        #emg_new.iloc[gz_counter_ges + 1, i] = mean_col - std_col
        #emg_new.iloc[gz_counter_ges + 2, i] = mean_col + std_col

    return emg_new



def construct_emg_newonoff(emg_new, num_gc_ges):
    # Initialize the DataFrame with the given structure
    columns = ['Label', 'x-Werte normalisiert [s]', 'GM L', 'GM R', 'RF L', 'RF R', 'BF L', 'BF R', 'Tensor L', 'Tensor R', 'Times toe offs']
    emg_newonoff = pd.DataFrame(columns=columns)
    
    emg_newonoff.loc[0] = [''] * len(columns)
    emg_newonoff.iloc[0, 1] = emg_new.loc[0, 'x-Werte normalisiert [s]']
    emg_newonoff.iloc[0, -1] = emg_new.loc[num_gc_ges - 1, 'Times toe offs']
    
    for i in range(2, len(columns) - 1):
        klein = np.min(emg_new.loc[0, columns[i]]) # num_gc_ges-1 instead of 0 possible
        gross = np.max(emg_new.loc[0, columns[i]]) # num_gc_ges-1 instead of 0 possible
        schwelle = 0.25 * (gross - klein) + klein
        
        messwerte = np.array(emg_new.loc[0, columns[i]]) # num_gc_ges-1 instead of 0 possible
        messwerte = np.where(messwerte >= schwelle, 1, 0)
        
        emg_newonoff.iloc[0, i] = messwerte

    return emg_newonoff