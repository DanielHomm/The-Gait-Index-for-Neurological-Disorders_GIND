# Required libraries
import pandas as pd
import kineticstoolkit.lab as ktk
import numpy as np
import os
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from utils.force.gc_grf_ges_mw import construct_gc_grf_ges, process_grf_data
from utils.force.gc_cop_ges_MW import construct_gc_cop_ges, process_cop_data
from utils.force.gc_akh_ges_MW import construct_akh_ges, process_akh_data
from utils.force.gc_pow_ges_MW import construct_pow_ges, process_pow_data
from utils.force.norm_gc_para import construct_norm_gc_para
from utils.marker.gc_para import construct_gc_para
from utils.marker.gc_ang_ges_MW import construct_gc_ang_ges, process_ang_data
from utils.marker.gc_com_ges_MW import construct_gc_com_ges, process_com_data
from utils.marker.gc_ang_bew_ges import construct_gc_ang_bew_ges
from utils.marker.emg_data import construct_emg_roh, construct_emg_bear, construct_emg_new, construct_emg_newonoff

# https://github.com/Biomechanical-ToolKit/BTKPython/issues/2
# https://github.com/conda-forge/btk-feedstock
# can be installed with conda install -c conda-forge btk
import btk

def create_data_dicts_btk(c3d_file):
    """
    This function reads the c3d file and stores its data in dictionaries
    
    args: c3d_file: path to c3d dynamic file
    
    returns: V_analysis: dictionary with walk-parameters
            meta_data: meta data of the c3d file
            ff: first frame
    """
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(c3d_file)
    reader.Update()
    acq = reader.GetOutput()

    # get analysis object automatic calcualted walk-parameters (black box from Vicon)
    meta_data = acq.GetMetaData()
    ff = acq.GetFirstFrame()
    analysis = None
    for i in range (meta_data.GetChildNumber()):
        if meta_data.GetChild(i).GetLabel().lower() == 'analysis':
            analysis = meta_data.GetChild(i)
            break
    if analysis is None:
        print('No analysis data found in c3d file')
    else:
        print('Analysis data found in c3d file')
        # the names of all walk-parameters should be stored in index 1
        names = analysis.GetChild(1).GetInfo().ToString()
        # the coresponding values of all walk-parameters should be stored in index 6
        values = analysis.GetChild(6).GetInfo().ToDouble()
        V_analysis = {names[i]: values[i] for i in range(len(names))}
    return V_analysis, meta_data, ff

def create_data_dicts_kinetics(c3d_file, c3d_file_stat):
    """
    This function reads the c3d files and stores its data in dictionaries
    
    args: c3d_file: path to c3d dynamic file
            c3d_file_stat: path to c3d static file
    
    return: ang: dictionary with angles
                ev: dictionary with events
                force: dictionary with forces
                force_sr: sampling rate of the forces
                mark: dictionary with markers
                mark_freq: sampling rate of the markers
                V_analysis: dictionary with walk-parameters
                ana: dictionary with analog data
                mom: dictionary with moments
                power: dictionary with power
                ana_stat: dictionary with static analog data
    """
    markers = ktk.read_c3d(c3d_file, convert_point_unit=False)
    markers_stat = ktk.read_c3d(c3d_file_stat, convert_point_unit=False)

    # Get all data with labels containing the word angle
    ang = {label: markers["Points"].data[label][:, 0:3] for label in markers["Points"].data.keys() if "Angle" in label}
    # Get all foot of and foot strike events -> have been defined specificly in Vicon Nexus
    ev = markers["Points"].events
    # Condition: Left foot strikes first on the ground !!!!!!!Needs to be adapted for multiple Gait-Cycles!!!!!!! - Kati fragen, wie das beschrieben werden soll für unterschiedliche Beinzyklen
    ev = {"Left_Foot_Strike": np.array([ev[2].time, ev[6].time]), "Left_Foot_Off": np.array([ev[1].time, ev[5].time]), "Right_Foot_Strike": np.array([ev[0].time, ev[4].time]), "Right_Foot_Off": np.array([ev[3].time])}
    # Get all data with labels containing the word force
    force = {label: markers["Points"].data[label][:, 0:3] for label in markers["Points"].data.keys() if "Force" in label or "NormalisedGRF" in label}
    # Is the analogous frequency really 1000Hz?
    force_sr = 1 / (markers["Analogs"].time[1] - markers["Analogs"].time[0])
    mark = {label: markers["Points"].data[label][:, 0:3] for label in markers["Points"].data.keys()}
    # Is the markers frequency really 200Hz?
    mark_freq = 1 / (markers["Points"].time[1] - markers["Points"].time[0])
    # V_analaysis is the same as above
    ana = markers["Analogs"].data
    mom = {label: markers["Points"].data[label][:, 0:3] for label in markers["Points"].data.keys() if "Moment" in label}
    power = {label: markers["Points"].data[label][:, 0:3] for label in markers["Points"].data.keys() if "Power" in label}

    # only static data that is used
    ana_stat = markers_stat["Analogs"].data
    return ang, ev, force, force_sr, mark, mark_freq, ana, mom, power, ana_stat


def patient_data(m_eing=88, h_eing=1860):
    m_eing=m_eing     # in Nexus hinterlegtes Körpergewicht
    h_eing=h_eing     # in Nexus hinterlegte Körpergröße
    return m_eing, h_eing

def get_strikes(ev, mark_freq, ff):
    """
    This function determines the strikes and toe offs of the right and left foot

    args: ev: dictionary with events

    returns: 
        R_Strike: right foot strikes
        R_ToeOff: right foot toe offs
        L_Strike: left foot strikes
        L_ToeOff: left foot toe offs
        fto_gc_R: first toe off right foot
        fto_gc_L: first toe off left foot
    """
    # if save force data == 1
    R_Strike = np.round(ev["Right_Foot_Strike"] * mark_freq).astype(int) - ff
    R_Strike[R_Strike <= 0] = 1
    R_ToeOff = np.round(ev["Right_Foot_Off"] * mark_freq).astype(int) - ff

    L_Strike = np.round(ev["Left_Foot_Strike"] * mark_freq).astype(int) - ff
    L_Strike[L_Strike <= 0] = 1
    L_ToeOff = np.round(ev["Left_Foot_Off"] * mark_freq).astype(int) - ff

    # Determine the first toe off
    if R_Strike[0] < R_ToeOff[0]:
        fto_gc_R = 1
    else:
        fto_gc_R = 2

    if L_Strike[0] < L_ToeOff[0]:
        fto_gc_L = 1
    else:
        fto_gc_L = 2
    return R_Strike, R_ToeOff, L_Strike, L_ToeOff, fto_gc_R, fto_gc_L


def create_xvec(trial, strikes, toe_offs, num_gc, fto_gc, num_gc_ges):
    """
    This function creates the x-vector for the gait cycle

    args: trial: trial name, strikes: foot strikes, toe_offs: toe offs,
      num_gc: number of gait cycles, fto_gc: first toe off, num_gc_ges: number of gait cycles

    return gc_xvec: DataFrame with the x-vector for the gait cycle
    """

    columns = ['Trial', 'Frames des entsprechenden GC', 'Länge des GC-Vektors', 'x-Werte auf 1-100% skaliert',
           'Phasenwechsel Stand- zur Schwungphase', 'Frame zum Zeitpunkt x=20% für GRFz Auswahl',
           'Frame zum Zeitpunkt x=10% für GRFxy Richtungszuweisung', 'Frames GC analog', 'x-Werte in % GC analog']   
    
    # Initialize DataFrames for R_gc_xvec and L_gc_xvec
    gc_xvec = pd.DataFrame(columns=columns)

    for i in range(num_gc):
        trial_nn = f"{trial}_{i+1}"
        frames_gc = np.arange(strikes[i] - 1, strikes[i + 1])
        length_gc = len(frames_gc)
        x_values = (frames_gc - (strikes[i] - 1)) / (length_gc - 1) * 100
        phase_change = (toe_offs[fto_gc + i - 1] - strikes[i]) / (strikes[i + 1] - strikes[i]) * 100
        frame_20 = 0.2 * (length_gc - 1) + strikes[i]
        frame_10 = 0.1 * (length_gc - 1) + strikes[i]
        frames_analog = np.arange(5 * (strikes[i] - 1) + 5, 5 * toe_offs[fto_gc + i - 1] + 1)
        x_values_analog = (frames_analog - 5 * (strikes[i] - 1) - 5) / (len(np.arange(5 * (strikes[i] - 1) + 5, 5 * (strikes[i + 1] - 1) + 5 + 1)) - 1) * 100

        gc_xvec.loc[num_gc_ges + i + 1] = [trial_nn, frames_gc, length_gc, x_values, phase_change, frame_20, frame_10, frames_analog, x_values_analog]

    return gc_xvec

def create_gz_counter(force_key, gc_xvec, num_gc, num_gc_ges, force, SW_GRFz=1.0):
    gz_counter = []
    if force_key in force:
        for i in range(num_gc):
            frame_idx = int(round(gc_xvec.loc[num_gc_ges + i + 1, 'Frame zum Zeitpunkt x=20% für GRFz Auswahl']))
            if force[force_key][frame_idx, 2] >= SW_GRFz:
                gz_counter.append(i)
    return gz_counter

def load_data(c3d_file, c3d_file_stat):
    # variables definition
    SW_GRFz = 1
    R_num_gc_ges = 0
    L_num_gc_ges = 0

    trial = "dyn01"

    # variables stored in the matlab file
    # where are these defined as 3? I thought they are defined as 0!?!?!?
    gz_counter_R_ges = 0
    gz_counter_L_ges = 0

    m_eing, h_eing = patient_data()
    ang, ev, force, force_sr, mark, mark_freq, ana, mom, power, ana_stat = create_data_dicts_kinetics(c3d_file, c3d_file_stat)
    V_analysis, meta_data, ff = create_data_dicts_btk(c3d_file)

    return ang, ev, force, force_sr, mark, mark_freq, V_analysis, ana, mom, power, ana_stat, m_eing, h_eing, R_num_gc_ges, L_num_gc_ges, gz_counter_R_ges, gz_counter_L_ges, SW_GRFz, trial, meta_data, ff


def read_force_data(c3d_file, c3d_file_stat):
    ang, ev, force, force_sr, mark, mark_freq, V_analysis, ana, mom, power, ana_stat, m_eing, h_eing, R_num_gc_ges, L_num_gc_ges, gz_counter_R_ges, gz_counter_L_ges, SW_GRFz, trial, meta_data, ff = load_data(c3d_file, c3d_file_stat)
    # Assuming ana_stat is a dictionary containing the necessary data
    mass_Korr = 1  # Default value if no mass determination through static
    if 'Force.Fz1' in ana_stat:
        mass = max(-np.mean(ana_stat['Force.Fz1']), -np.mean(ana_stat['Force.Fz2'])) / 9.81  # Body weight calculation
        mass_Korr = m_eing / mass  # Correction factor for the forces based on calibrated body weight

    R_Strike, R_ToeOff, L_Strike, L_ToeOff, fto_gc_R, fto_gc_L = get_strikes(ev, mark_freq, ff)

    
    # Define the number of gait cycles
    num_gc_R = len(R_Strike) - 1
    num_gc_L = len(L_Strike) - 1

    R_gc_xvec = create_xvec(trial, R_Strike, R_ToeOff, num_gc_R, fto_gc_R, R_num_gc_ges)
    L_gc_xvec = create_xvec(trial, L_Strike, L_ToeOff, num_gc_L, fto_gc_L, L_num_gc_ges)

    # Matlab code line: 605
    gz_counter_R = create_gz_counter('RGroundReactionForce', R_gc_xvec, num_gc_R, R_num_gc_ges, force, SW_GRFz=SW_GRFz)
    gz_counter_L = create_gz_counter('LGroundReactionForce', L_gc_xvec, num_gc_L, L_num_gc_ges, force, SW_GRFz=SW_GRFz)

    # Line 690
    R_gc_grf_ges = construct_gc_grf_ges(trial, force, gz_counter_R, gz_counter_R_ges, R_gc_xvec, fto_gc_R, 'RGroundReactionForce', 'RNormalisedGRF', R_Strike, R_ToeOff, R_num_gc_ges, mass_Korr)
    L_gc_grf_ges = construct_gc_grf_ges(trial, force, gz_counter_L, gz_counter_L_ges, L_gc_xvec, fto_gc_L, 'LGroundReactionForce', 'LNormalisedGRF', L_Strike, L_ToeOff, L_num_gc_ges, mass_Korr)
    
    # Line 815
    cut_off_frame = 40 - 1
    if 'RGroundReactionForce' in force:
        R_gc_cop_ges = construct_gc_cop_ges(gz_counter_R, gz_counter_R_ges, R_num_gc_ges, R_gc_xvec, R_Strike, R_ToeOff, fto_gc_R, trial, ana, cut_off_frame)
    if 'LGroundReactionForce' in force:
        L_gc_cop_ges = construct_gc_cop_ges(gz_counter_L, gz_counter_L_ges, L_num_gc_ges, L_gc_xvec, L_Strike, L_ToeOff, fto_gc_L, trial, ana, cut_off_frame)
    
    # Line 936
    R_gc_akh_ges = construct_akh_ges(gz_counter_R, R_num_gc_ges, R_gc_xvec, R_Strike, R_ToeOff, fto_gc_R, trial, force, mom, mass_Korr, 'R')
    L_gc_akh_ges = construct_akh_ges(gz_counter_L, L_num_gc_ges, L_gc_xvec, L_Strike, L_ToeOff, fto_gc_L, trial, force, mom, mass_Korr, 'L')

    # Line 1166
    R_gc_pow_ges = construct_pow_ges(gz_counter_R, R_num_gc_ges, R_gc_xvec, R_Strike, R_ToeOff, fto_gc_R, trial, power, mass_Korr, 'R')
    L_gc_pow_ges = construct_pow_ges(gz_counter_L, L_num_gc_ges, L_gc_xvec, L_Strike, L_ToeOff, fto_gc_L, trial, power, mass_Korr, 'L')


    # Line 1227
    R_gc_ang_ges = construct_gc_ang_ges(R_gc_xvec, R_num_gc_ges, R_Strike, R_ToeOff, ang, 'R', trial)
    L_gc_ang_ges = construct_gc_ang_ges(L_gc_xvec, L_num_gc_ges, L_Strike, L_ToeOff, ang, 'L', trial)
    
    # Line 1740
    R_num_gc_ges = R_num_gc_ges+num_gc_R
    L_num_gc_ges = L_num_gc_ges+num_gc_L
    gz_counter_R_ges= len(gz_counter_R)+gz_counter_R_ges
    gz_counter_L_ges= len(gz_counter_L)+gz_counter_L_ges

    # Line 1790
    norm_gc_para = construct_norm_gc_para()

    # Line 2130
    R_gc_grf_ges_MW = process_grf_data(gz_counter_R_ges, R_gc_grf_ges)
    L_gc_grf_ges_MW = process_grf_data(gz_counter_L_ges, L_gc_grf_ges)

    # Line 2426
    cut_off_frame = 40 - 1
    R_gc_cop_ges_MW = process_cop_data(R_gc_cop_ges, R_gc_ang_ges, gz_counter_R_ges, cut_off_frame, side="R")
    L_gc_cop_ges_MW = process_cop_data(L_gc_cop_ges, L_gc_ang_ges, gz_counter_L_ges, cut_off_frame, side='L')

    # Line 2766
    R_gc_akh_ges_MW = process_akh_data(R_gc_akh_ges, gz_counter_R_ges, side='R')
    L_gc_akh_ges_MW = process_akh_data(L_gc_akh_ges, gz_counter_L_ges, side='L')

    # Line 2950
    R_gc_pow_ges_MW = process_pow_data(R_gc_pow_ges, len(gz_counter_R))
    L_gc_pow_ges_MW = process_pow_data(L_gc_pow_ges, len(gz_counter_L))
    
    gc_count_force={'gz_counter_R_ges': gz_counter_R_ges,'gz_counter_L_ges': gz_counter_L_ges}

    force_data = {"R_gc_cop_ges_MW": R_gc_cop_ges_MW, "L_gc_cop_ges_MW": L_gc_cop_ges_MW, "R_gc_grf_ges_MW": R_gc_grf_ges_MW, "L_gc_grf_ges_MW": L_gc_grf_ges_MW, "R_gc_akh_ges_MW": R_gc_akh_ges_MW, "L_gc_akh_ges_MW": L_gc_akh_ges_MW, "norm_gc_para": norm_gc_para, "gc_count_force": gc_count_force, "R_gc_pow_ges_MW": R_gc_pow_ges_MW, "L_gc_pow_ges_MW": L_gc_pow_ges_MW}

    return force_data


def read_marker_data(c3d_file, c3d_file_stat):
    ang, ev, force, force_sr, mark, mark_freq, V_analysis, ana, mom, power, ana_stat, m_eing, h_eing, R_num_gc_ges, L_num_gc_ges, gz_counter_R_ges, gz_counter_L_ges, SW_GRFz, trial, meta_data, ff = load_data(c3d_file, c3d_file_stat)

    R_Strike, R_ToeOff, L_Strike, L_ToeOff, fto_gc_R, fto_gc_L = get_strikes(ev, mark_freq, ff)
    
    # Define the number of gait cycles
    num_gc_R = len(R_Strike) - 1
    num_gc_L = len(L_Strike) - 1

    R_gc_xvec = create_xvec(trial, R_Strike, R_ToeOff, num_gc_R, fto_gc_R, R_num_gc_ges)
    L_gc_xvec = create_xvec(trial, L_Strike, L_ToeOff, num_gc_L, fto_gc_L, L_num_gc_ges)

    # Line 398
    R_gc_para = construct_gc_para(R_gc_xvec, num_gc_R, R_Strike, R_ToeOff, trial, mark, fto_gc_R, L_Strike, L_ToeOff, mark_freq, side='R')
    L_gc_para = construct_gc_para(L_gc_xvec, num_gc_L, L_Strike, L_ToeOff, trial, mark, fto_gc_L, R_Strike, R_ToeOff, mark_freq, side='L')

    # Line 605
    gz_counter_R = create_gz_counter('RGroundReactionForce', R_gc_xvec, num_gc_R, R_num_gc_ges, force, SW_GRFz=SW_GRFz)
    gz_counter_L = create_gz_counter('LGroundReactionForce', L_gc_xvec, num_gc_L, L_num_gc_ges, force, SW_GRFz=SW_GRFz)

    # Line 1227
    R_gc_ang_ges = construct_gc_ang_ges(R_gc_xvec, R_num_gc_ges, R_Strike, R_ToeOff, ang, 'R', trial)
    L_gc_ang_ges = construct_gc_ang_ges(L_gc_xvec, L_num_gc_ges, L_Strike, L_ToeOff, ang, 'L', trial)

    # Line 1420
    R_gc_ang_bew_ges = construct_gc_ang_bew_ges(R_gc_ang_ges, num_gc_R, trial)
    L_gc_ang_bew_ges = construct_gc_ang_bew_ges(L_gc_ang_ges, num_gc_L, trial)

    # Line 1529
    EMG_R_roh = construct_emg_roh(trial, R_Strike, R_ToeOff, ana, num_gc_R, mark_freq, 'R')
    EMG_L_roh = construct_emg_roh(trial, L_Strike, L_ToeOff, ana, num_gc_L, mark_freq, 'L')

    # Line 1643
    R_gc_com_ges = construct_gc_com_ges(R_gc_xvec, num_gc_R, R_Strike, R_ToeOff, trial, mark, fto_gc_R)
    L_gc_com_ges = construct_gc_com_ges(L_gc_xvec, num_gc_L, L_Strike, L_ToeOff, trial, mark, fto_gc_L)

    # Line 1740
    R_num_gc_ges = R_num_gc_ges+num_gc_R
    L_num_gc_ges = L_num_gc_ges+num_gc_L
    gz_counter_R_ges= len(gz_counter_R)+gz_counter_R_ges
    gz_counter_L_ges= len(gz_counter_L)+gz_counter_L_ges

    # Line 3158
    R_gc_ang_ges_MW = process_ang_data(R_gc_ang_ges, num_gc_R)
    L_gc_ang_ges_MW = process_ang_data(L_gc_ang_ges, num_gc_L)

    # Line 3455
    # Cutoff frequencies
    fcuthigh = 20
    fcutlow = 450
    fcutenv = 9

    EMG_R_bear = construct_emg_bear(EMG_R_roh, fcuthigh, fcutlow, fcutenv, force_sr=force_sr, n=0)
    EMG_L_bear = construct_emg_bear(EMG_L_roh, fcuthigh, fcutlow, fcutenv, force_sr=force_sr, n=0)

    # Line 3933
    right_EMG_new = construct_emg_new(EMG_R_bear, mark_freq, gz_counter_R_ges)
    left_EMG_new = construct_emg_new(EMG_L_bear, mark_freq, gz_counter_L_ges)

    # Line 4005
    # Kati fragen SChwellwerte werden gleich berechnet (Klein = Groß), komische implementierung
    right_EMG_newonoff = construct_emg_newonoff(right_EMG_new, R_num_gc_ges)
    left_EMG_newonoff = construct_emg_newonoff(left_EMG_new, L_num_gc_ges)
    
    # Line 4110
    R_gc_com_ges_MW = process_com_data(R_gc_com_ges, num_gc_R)
    L_gc_com_ges_MW = process_com_data(L_gc_com_ges, num_gc_L)

    gc_count_mark={'R_num_gc_ges': R_num_gc_ges,'L_num_gc_ges': L_num_gc_ges}
    
    marker_data = {"R_gc_com_ges_MW": R_gc_com_ges_MW, "L_gc_com_ges_MW": L_gc_com_ges_MW, "R_gc_ang_ges_MW": R_gc_ang_ges_MW, "L_gc_ang_ges_MW": L_gc_ang_ges_MW, "R_gc_ang_bew_ges": R_gc_ang_bew_ges, "L_gc_ang_bew_ges": L_gc_ang_bew_ges, "mark": mark, "right_EMG_new": right_EMG_new, "left_EMG_new": left_EMG_new, "EMG_R_roh": EMG_R_roh, "EMG_L_roh": EMG_L_roh, "EMG_R_bear": EMG_R_bear, "EMG_L_bear": EMG_L_bear, "R_gc_para": R_gc_para, "L_gc_para": L_gc_para, "right_EMG_newonoff": right_EMG_newonoff, "left_EMG_newonoff": left_EMG_newonoff, "gc_count_mark": gc_count_mark}

    return marker_data


if __name__ == "__main__":

    laptop= r'D:\TUM-HIWI\Messdaten\Masterarbeit Data\Data final\Christian\Gait FullBody'
    pc =  r'C:\UNI\HIWI\TUM\HIWI_SPM\Messdaten\Masterarbeit Data\Data final\Christian\Gait FullBody'

    c3d_file = os.path.join(pc, 'dyn01.c3d')
    # path c3d file with static data
    c3d_file_stat = os.path.join(pc, 'stat01.c3d')

    force_data = create_force_data(c3d_file, c3d_file_stat)
    #marker_data = create_marker_data(c3d_file, c3d_file_stat)
