# Required libraries
import pandas as pd
import kineticstoolkit.lab as ktk
import numpy as np
import os

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
# can be installed with conda install -c conda-forge btk, currently only available for Python < 3.12
import btk

def create_data_dicts_btk(c3d_file):
    """
    This function reads the c3d file and stores its data in Python dictionaries
    
    args: c3d_file path to c3d dynamic file: str
    
    returns: V_analysis: dictionary with walk-parameters
            meta_data, meta data of the c3d file: btkMetaData
            ff, first frame: int
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
        # the names of all walk-parameters should be stored in index 1
        names = analysis.GetChild(1).GetInfo().ToString()
        # the coresponding values of all walk-parameters should be stored in index 6
        values = analysis.GetChild(6).GetInfo().ToDouble()
        V_analysis = {names[i]: values[i] for i in range(len(names))}
    return V_analysis, meta_data, ff

def create_data_dicts_kinetics(c3d_file, c3d_file_stat, event_idx, first_foot=None):
    """
    This function reads the c3d files and stores its data in dictionaries
    
    args: c3d_file, path to c3d dynamic file: str
            c3d_file_stat, path to c3d static file: str
    
    return: ang: dictionary with angles
                ev: dictionary with events
                force: dictionary with forces
                force_sr: sampling rate of the forces
                mark: dictionary with markers
                mark_freq: sampling rate of the markers (int)
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
    events = markers["Points"].events
    first_foot = first_foot.lower() if first_foot is not None else first_foot
    if first_foot is None or (first_foot != "right" and first_foot != "left"):
        try:
            ev = {"Left_Foot_Strike": np.array([events[i].time for i in event_idx["Left_Foot_Strike"]]), 
                  "Left_Foot_Off": np.array([events[i].time for i in event_idx["Left_Foot_Off"]]), 
                  "Right_Foot_Strike": np.array([events[i].time for i in event_idx["Right_Foot_Strike"]]),
                  "Right_Foot_Off": np.array([events[i].time for i in event_idx["Right_Foot_Off"]])}
        except:
            raise KeyError("Please check if you put the correct indeces for the strikes and toe offs." +
                            "If you don't know the exact indeces please provide for the variable first_foot the *left* or *right* to identify the first strike")
    else:
        length = len(events)
        ev = {"Left_Foot_Strike": [], "Left_Foot_Off": [], "Right_Foot_Strike": [], "Right_Foot_Off": []}
        if first_foot == "right":
            ev["Right_Foot_Strike"].append(events[0].time)
            for i in range(0, length, 4):
                if i + 1 < length:
                    ev["Left_Foot_Off"].append(events[i + 1].time)
                if i + 2 < length:
                    ev["Left_Foot_Strike"].append(events[i + 2].time)
                if i + 3 < length:
                    ev["Right_Foot_Off"].append(events[i + 3].time)
                if i + 4 < length:
                    ev["Right_Foot_Strike"].append(events[i + 4].time)
        elif first_foot == "left":
            ev["Left_Foot_Strike"].append(events[0].time)
            for i in range(length):
                if i + 1 < length:
                    ev["Right_Foot_Off"].append(events[i + 1].time)
                if i + 2 < length:
                    ev["Right_Foot_Strike"].append(events[i + 2].time)
                if i + 3 < length:
                    ev["Left_Foot_Off"].append(events[i + 3].time)
                if i + 4 < length:
                    ev["Left_Foot_Strike"].append(events[i + 4].time)
        
        for key in ev:
            ev[key] = np.array(ev[key])
        print(ev)

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


def participant_data(m_eing=88, h_eing=1860):
    """
    This function returns the data needed from the corresponding participant.
    (In our case the in Nexus set Bodyweight and Bodyheight)

    args:   m_eing Bodyweight: int
            h_eing Bodyheight: int

    returns:
            m_eing Bodyweight: int
            h_eing Bodyheight: int
    """
    m_eing=m_eing     # Bodyweight
    h_eing=h_eing     # Bodyheight
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

def load_data(c3d_file, c3d_file_stat, first_foot, event_idx, m_eing=None, h_eing=None, trial="dyn01"):
    """
    Function to combine the read data functions and general varaible definitions and return all needed data dictionaries.

    args:   c3d_file: path to c3d dynamic file
            c3d_file_stat: path to c3d static file
            m_eing: Bodyweight of the participant (set in Nexus)
            h_eing: Bodyheight of the participant (set in Nexus)
    
    returns:    ang: dictionary with angles  
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
                m_eing: Bodyweight of the participant
                h_eing: Bodyheight of the participant
                R_num_gc_ges: number of gait cycles for the right foot
                L_num_gc_ges: number of gait cycles for the left foot
                gz_counter_R_ges: counter for the right foot
                gz_counter_L_ges: counter for the left foot
                SW_GRFz: threshold for the ground reaction force
                trial: trial name
                meta_data: meta data of the c3d file
                ff: first frame
    """

    # variables definition
    SW_GRFz = 1
    R_num_gc_ges = 0
    L_num_gc_ges = 0

    trial = "dyn01"

    # variables stored in the matlab file
    # where are these defined as 3? I thought they are defined as 0!?!?!?
    gz_counter_R_ges = 0
    gz_counter_L_ges = 0
    
    if m_eing and h_eing:
        m_eing, h_eing = participant_data(m_eing, h_eing)
    else:
        m_eing, h_eing = participant_data()
    
    ang, ev, force, force_sr, mark, mark_freq, ana, mom, power, ana_stat = create_data_dicts_kinetics(c3d_file, c3d_file_stat, event_idx=event_idx, first_foot= first_foot)
    V_analysis, meta_data, ff = create_data_dicts_btk(c3d_file)

    return ang, ev, force, force_sr, mark, mark_freq, V_analysis, ana, mom, power, ana_stat, m_eing, h_eing, R_num_gc_ges, L_num_gc_ges, gz_counter_R_ges, gz_counter_L_ges, SW_GRFz, trial, meta_data, ff


def read_force_data(c3d_file, c3d_file_stat, first_foot, event_idx, weight, height, trial="dyn01"):
    """
    This function reads the force data from the c3d file and processes it to be used in the further analysis.

    args:   c3d_file: path to c3d dynamic file
            c3d_file_stat: path to c3d static file

    returns:    force_data: Dictionary with the processed force data. Each key holds a pandas dataframe with the corresponding data.
    """

    ang, ev, force, _, _, mark_freq, _, ana, mom, power, ana_stat, m_eing, _, R_num_gc_ges, L_num_gc_ges, gz_counter_R_ges, gz_counter_L_ges, SW_GRFz, trial, _, ff = load_data(c3d_file, c3d_file_stat, first_foot, event_idx, weight, height, trial=trial)
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
    R_gc_akh_ges_MW = process_akh_data(R_gc_akh_ges, gz_counter_R_ges)
    L_gc_akh_ges_MW = process_akh_data(L_gc_akh_ges, gz_counter_L_ges)

    # Line 2950
    R_gc_pow_ges_MW = process_pow_data(R_gc_pow_ges, len(gz_counter_R))
    L_gc_pow_ges_MW = process_pow_data(L_gc_pow_ges, len(gz_counter_L))
    
    gc_count_force={'gz_counter_R_ges': gz_counter_R_ges,'gz_counter_L_ges': gz_counter_L_ges}

    force_data = {"R_gc_cop_ges_MW": R_gc_cop_ges_MW, "L_gc_cop_ges_MW": L_gc_cop_ges_MW, "R_gc_grf_ges_MW": R_gc_grf_ges_MW, "L_gc_grf_ges_MW": L_gc_grf_ges_MW, "R_gc_akh_ges_MW": R_gc_akh_ges_MW, "L_gc_akh_ges_MW": L_gc_akh_ges_MW, "norm_gc_para": norm_gc_para, "gc_count_force": gc_count_force, "R_gc_pow_ges_MW": R_gc_pow_ges_MW, "L_gc_pow_ges_MW": L_gc_pow_ges_MW}

    return force_data


def read_marker_data(c3d_file, c3d_file_stat, first_foot, event_idx, trial="dyn01"):
    """
    This function reads the marker data from the c3d files and processes it to be used in the further analysis.

    args:   c3d_file: path to c3d dynamic file
            c3d_file_stat: path to c3d static file

    returns:    
            force_data: Dictionary with the processed force data. Each key holds a pandas dataframe with the corresponding data.
    """
    ang, ev, force, force_sr, mark, mark_freq, _, ana, _, _, _, _, _, R_num_gc_ges, L_num_gc_ges, gz_counter_R_ges, gz_counter_L_ges, SW_GRFz, trial, _, ff = load_data(c3d_file, c3d_file_stat, first_foot, event_idx, trial=trial)

    R_Strike, R_ToeOff, L_Strike, L_ToeOff, fto_gc_R, fto_gc_L = get_strikes(ev, mark_freq, ff)
    
    # Define the number of gait cycles
    num_gc_R = len(R_Strike) - 1
    num_gc_L = len(L_Strike) - 1

    R_gc_xvec = create_xvec(trial, R_Strike, R_ToeOff, num_gc_R, fto_gc_R, R_num_gc_ges)
    L_gc_xvec = create_xvec(trial, L_Strike, L_ToeOff, num_gc_L, fto_gc_L, L_num_gc_ges)

    # Line 398
    R_gc_para = construct_gc_para(num_gc_R, R_Strike, R_ToeOff, trial, mark, fto_gc_R, L_Strike, L_ToeOff, mark_freq, side='R')
    L_gc_para = construct_gc_para(num_gc_L, L_Strike, L_ToeOff, trial, mark, fto_gc_L, R_Strike, R_ToeOff, mark_freq, side='L')

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
    EMG_R_roh = construct_emg_roh(trial, R_Strike, R_ToeOff, ana, num_gc_R, mark_freq)
    EMG_L_roh = construct_emg_roh(trial, L_Strike, L_ToeOff, ana, num_gc_L, mark_freq)

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