# general imports
import numpy as np
import matplotlib.pyplot as plt
import os
import read_file as rmd
from varaiables import PATH_TO_DYNAMIC_C3D_FILE, PATH_TO_STATIC_C3D_FILE, WEIGHT, HEIGHT, TRIAL_NAME, PLOT, FIRST_FOOT, LEFT_FOOT_STRIKE, LEFT_FOOT_OFF, RIGHT_FOOT_STRIKE, RIGHT_FOOT_OFF


def rms(values):
    return np.sqrt(np.mean(np.square(values)))

def plot_gind(gind_value, trial_name):
    y_max = max(8, round(gind_value) + 1)

    fig, ax = plt.subplots()

    ax.plot(trial_name, gind_value, 'ro') #, label=f'{trial_name}: {gind_value}')

    ax.set_ylim(0, y_max)

    ax.axhline(y=1, color='green', linestyle='-', linewidth=1, label='Healthy Movement')

    ax.set_ylabel('Value')
    ax.set_title('Gind Value')

    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Provided Variable checks.
    if not PATH_TO_DYNAMIC_C3D_FILE:
        raise ValueError("PATH_TO_DYNAMIC_C3D_FILE is an empty string. Please provide a valid file path.")
    if not PATH_TO_STATIC_C3D_FILE:
        raise ValueError("PATH_TO_STATIC_C3D_FILE is an empty string. Please provide a valid file path.")
    if not TRIAL_NAME:
        raise ValueError("TRIAL_NAME is an empty string. Please provide a valid trial name.")
    
    if not isinstance(WEIGHT, int):
        raise TypeError("WEIGHT must be an integer. Please provide a valid weight.")
    if not isinstance(HEIGHT, int):
        raise TypeError("HEIGHT must be an integer. Please provide a valid height.")

    if FIRST_FOOT not in ["right", "left", "Right", "Left", None]:
        raise ValueError("FIRST_FOOT must be 'right', 'left', or None. Please provide a valid value.")

    if FIRST_FOOT is None:
        if not isinstance(LEFT_FOOT_STRIKE, list) or not LEFT_FOOT_STRIKE:
            raise ValueError("LEFT_FOOT_STRIKE must be a non-empty list when FIRST_FOOT is None.")
        if not isinstance(LEFT_FOOT_OFF, list) or not LEFT_FOOT_OFF:
            raise ValueError("LEFT_FOOT_OFF must be a non-empty list when FIRST_FOOT is None.")
        if not isinstance(RIGHT_FOOT_STRIKE, list) or not RIGHT_FOOT_STRIKE:
            raise ValueError("RIGHT_FOOT_STRIKE must be a non-empty list when FIRST_FOOT is None.")
        if not isinstance(RIGHT_FOOT_OFF, list) or not RIGHT_FOOT_OFF:
            raise ValueError("RIGHT_FOOT_OFF must be a non-empty list when FIRST_FOOT is None.")



    c3d_file = PATH_TO_DYNAMIC_C3D_FILE
    c3d_file_stat = PATH_TO_STATIC_C3D_FILE

    # read marker and force data from c3d file
    if FIRST_FOOT == None:
        event_idx = {"Left_Foot_Strike": LEFT_FOOT_STRIKE, "Left_Foot_Off": LEFT_FOOT_OFF, "Right_Foot_Strike": RIGHT_FOOT_STRIKE, "Right_Foot_Off": RIGHT_FOOT_OFF}
    else:
        event_idx = {}

    FB_mark = rmd.read_marker_data(c3d_file, c3d_file_stat, FIRST_FOOT, event_idx, trial=TRIAL_NAME)
    FB_force = rmd.read_force_data(c3d_file, c3d_file_stat, FIRST_FOOT, event_idx, WEIGHT, HEIGHT, trial=TRIAL_NAME)

    # include health values for reference and calculations (already calculated from 3 healthy subjects)
    cop_healthy_m = np.load('utils/variables/cop_healthy_m.npy')
    g_healthy_m = np.load('utils/variables/g_healthy_m.npy')
    grf_healthy_m = np.load('utils/variables/grf_healthy_m.npy')
    GVS_healthy = np.load('utils/variables/GVS_healthy.npy')
    GVS_cop_healthy = np.load('utils/variables/GVS_cop_healthy.npy')
    GVS_grf_healthy = np.load('utils/variables/GVS_grf_healthy.npy')

    # Parameters for calculatio of gait symmetry
    para = np.array([12, 13, 14, 9, 10, 11, 6, 3, 17]) - 1

    # array inits for arrays in line 
    row_MW_R= FB_mark["gc_count_mark"]["R_num_gc_ges"] 
    row_MW_L= FB_mark["gc_count_mark"]["L_num_gc_ges"]
    angles_ges_mean = np.zeros(51)
    g_l = np.zeros(len(para)*angles_ges_mean.shape[0])
    g_r = np.zeros(len(para)*angles_ges_mean.shape[0])

    # loop for all the parameters that need to be collected
    for n in range(len(para)):
        # calculate g_l for left side
        angles_red_ges = np.zeros((51, row_MW_L), dtype=np.float64)
        for k in range(0, row_MW_L):
            angles_curr = FB_mark["L_gc_ang_ges_MW"].iloc[k, para[n]]
            angles_red = angles_curr[0:201:4]
            angles_red_ges[:, k] = angles_red
            angles_curr = []
            angles_red = []
        angles_ges_mean[:] = np.mean(angles_red_ges[:, 0:row_MW_L], axis=1)
        g_l[n*angles_ges_mean.shape[0]:(n+1)*angles_ges_mean.shape[0]] = angles_ges_mean
        
        # calculate right marks
        angles_red_ges = np.zeros((51, row_MW_R), dtype=np.float64)
        for k in range(0, row_MW_R):
            angles_curr = FB_mark["R_gc_ang_ges_MW"].iloc[k, para[n]]
            angles_red = angles_curr[0:201:4]
            angles_red_ges[:, k] = angles_red
        angles_ges_mean[:] = np.mean(angles_red_ges[:, 0:row_MW_R], axis=1)
        g_r[n*angles_ges_mean.shape[0]:(n+1)*angles_ges_mean.shape[0]] = angles_ges_mean

    p = 9 # number of parameters (len(para))
    GVS_l_healthy = np.array([np.sqrt(np.mean(np.square(g_l[i*51:(i+1)*51]-g_healthy_m[i*51:(i+1)*51]))) for i in range(p)])
    GVS_r_healthy = np.array([np.sqrt(np.mean(np.square(g_r[i*51:(i+1)*51]-g_healthy_m[i*51:(i+1)*51]))) for i in range(p)])

    # Final values for GIND calculation
    V_GVS_l_healthy = GVS_l_healthy / GVS_healthy
    V_GVS_r_healthy = GVS_r_healthy / GVS_healthy

    para = np.array([6, 7, 8]) - 1

    row_MW_R = FB_force["gc_count_force"]["gz_counter_R_ges"]
    row_MW_L = FB_force["gc_count_force"]["gz_counter_L_ges"]
    grf_ges_mean = np.zeros(51)
    grf_l_healthy = np.zeros(len(para)*grf_ges_mean.shape[0])
    grf_r_healthy = np.zeros(len(para)*grf_ges_mean.shape[0])

    for n in range(len(para)):
        grf_red_ges = np.zeros((51, row_MW_L), dtype=np.float64)
        for k in range(row_MW_L):
            grf_curr = FB_force["L_gc_grf_ges_MW"].iloc[k, para[n]]
            grf_red = grf_curr[0:201:4].T
            grf_red_ges[:, k] = grf_red
            grf_curr = []
            grf_red = []
        grf_ges_mean[:] = np.mean(grf_red_ges[:, 0:row_MW_L], axis=1)
        grf_l_healthy[n*grf_ges_mean.shape[0]:(n+1)*grf_ges_mean.shape[0]] = grf_ges_mean

        grf_ges_mean = np.zeros(51)
        grf_red_ges = np.zeros((51, row_MW_R), dtype=np.float64)
        for k in range(row_MW_R):
            grf_curr = FB_force["R_gc_grf_ges_MW"].iloc[k, para[n]]
            grf_red = grf_curr[0:201:4].T
            grf_red_ges[:, k] = grf_red
            grf_curr = []
            grf_red = []
        grf_ges_mean[:] = np.mean(grf_red_ges[:, 0:row_MW_R], axis=1)
        grf_r_healthy[n*grf_ges_mean.shape[0]:(n+1)*grf_ges_mean.shape[0]] = grf_ges_mean

    p = 3 # len(para)
    GVS_grf_l_healthy = np.array([np.sqrt(np.mean(np.square(grf_l_healthy[i*51:(i+1)*51]-grf_healthy_m[i*51:(i+1)*51]))) for i in range(p)])
    GVS_grf_r_healthy = np.array([np.sqrt(np.mean(np.square(grf_r_healthy[i*51:(i+1)*51]-grf_healthy_m[i*51:(i+1)*51]))) for i in range(p)])
    V_GVS_grf_l_healthy = GVS_grf_l_healthy / GVS_grf_healthy
    V_GVS_grf_r_healthy = GVS_grf_r_healthy / GVS_grf_healthy

    para = np.array([3, 4, 6, 7, 9, 10]) -1

    row_MW_R = FB_force["gc_count_force"]["gz_counter_R_ges"]
    row_MW_L = FB_force["gc_count_force"]["gz_counter_L_ges"]
    cop_ges_mean = np.zeros(51)
    cop_l_healthy = np.zeros(len(para)*cop_ges_mean.shape[0])
    cop_r_healthy = np.zeros(len(para)*cop_ges_mean.shape[0])

    for n in range(len(para)):
        cop_red_ges = np.zeros((51, row_MW_L), dtype=np.float64)
        for k in range(0, row_MW_L):
            cop_curr = FB_force["L_gc_cop_ges_MW"].iloc[k, para[n]]
            cop_red = cop_curr[0:201:4].T
            cop_red_ges[:, k] = cop_red
            cop_curr = []
            cop_red = []
        cop_ges_mean[:] = np.mean(cop_red_ges[:, 0:row_MW_L], axis=1)
        cop_l_healthy[n*cop_ges_mean.shape[0]:(n+1)*cop_ges_mean.shape[0]] = cop_ges_mean
        
        cop_red_ges = np.zeros((51, row_MW_R), dtype=np.float64)
        for k in range(0, row_MW_R):
            cop_curr = FB_force["R_gc_cop_ges_MW"].iloc[k, para[n]]
            cop_red = cop_curr[0:201:4].T
            cop_red_ges[:, k] = cop_red
            cop_curr = []
            cop_red = []
        cop_ges_mean[:] = np.mean(cop_red_ges[:, 0:row_MW_L], axis=1)
        cop_r_healthy[n*cop_ges_mean.shape[0]:(n+1)*cop_ges_mean.shape[0]] = cop_ges_mean

    p = 6 # (len(p))
    GVS_cop_l_healthy = np.array([np.sqrt(np.mean(np.square(cop_l_healthy[i*51:(i+1)*51]-cop_healthy_m[i*51:(i+1)*51]))) for i in range(p)])
    GVS_cop_r_healthy = np.array([np.sqrt(np.mean(np.square(cop_r_healthy[i*51:(i+1)*51]-cop_healthy_m[i*51:(i+1)*51]))) for i in range(p)])

    V_GVS_cop_l_healthy = GVS_cop_l_healthy / GVS_cop_healthy
    V_GVS_cop_r_healthy = GVS_cop_r_healthy / GVS_cop_healthy

    V_GPS_gesamt_l_healthy= rms([V_GVS_l_healthy[0], V_GVS_l_healthy[3], V_GVS_l_healthy[6], V_GVS_l_healthy[7], V_GVS_grf_l_healthy[2], V_GVS_cop_l_healthy[1]])
    V_GPS_gesamt_r_healthy= rms([V_GVS_r_healthy[0], V_GVS_r_healthy[3], V_GVS_r_healthy[6], V_GVS_r_healthy[7], V_GVS_grf_r_healthy[2], V_GVS_cop_r_healthy[1]])


    V_GPS_gesamt_o = rms([
        V_GVS_l_healthy[0],  # 1 in MATLAB
        V_GVS_l_healthy[3],  # 4 in MATLAB
        V_GVS_l_healthy[6],  # 7 in MATLAB
        V_GVS_l_healthy[7],  # 8 in MATLAB
        V_GVS_grf_l_healthy[2],  # 3 in MATLAB
        V_GVS_cop_l_healthy[1],  # 2 in MATLAB
        V_GVS_r_healthy[3],  # 4 in MATLAB
        V_GVS_r_healthy[6],  # 7 in MATLAB
        V_GVS_r_healthy[7],  # 8 in MATLAB
        V_GVS_grf_r_healthy[2],  # 3 in MATLAB
        V_GVS_cop_r_healthy[1]   # 2 in MATLAB
    ])

    print(V_GPS_gesamt_o)

    if PLOT:
        plot_gind(V_GPS_gesamt_o, TRIAL_NAME)