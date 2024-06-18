# general imports
import numpy as np
import os
import read_file as rmd


def rms(values):
    return np.sqrt(np.mean(np.square(values)))

if __name__ == "__main__":
    # Adapt path to dynamic and static c3d file
    pc =  r'C:\UNI\HIWI\TUM\HIWI_SPM\Messdaten\Masterarbeit Data\Data final\Christian\Gait FullBody'

    c3d_file = os.path.join(pc, 'dyn01.c3d')
    # path c3d file with static data
    c3d_file_stat = os.path.join(pc, 'stat01.c3d')

    # read marker and force data from c3d file
    FB_mark = rmd.read_marker_data(c3d_file, c3d_file_stat)
    FB_force = rmd.read_force_data(c3d_file, c3d_file_stat)

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
    g_l_healthy = np.zeros(len(para)*angles_ges_mean.shape[0])
    g_r_healthy = np.zeros(len(para)*angles_ges_mean.shape[0])

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
        g_l_healthy[n*angles_ges_mean.shape[0]:(n+1)*angles_ges_mean.shape[0]] = angles_ges_mean;
        
        # calculate right marks
        angles_red_ges = np.zeros((51, row_MW_R), dtype=np.float64)
        for k in range(0, row_MW_R):
            angles_curr = FB_mark["R_gc_ang_ges_MW"].iloc[k, para[n]]
            angles_red = angles_curr[0:201:4]
            angles_red_ges[:, k] = angles_red
        angles_ges_mean[:] = np.mean(angles_red_ges[:, 0:row_MW_R], axis=1)
        g_r_healthy[n*angles_ges_mean.shape[0]:(n+1)*angles_ges_mean.shape[0]] = angles_ges_mean

    p = 9 # number of parameters (len(para))
    GVS_l_healthy = np.array([np.sqrt(np.mean(np.square(g_l_healthy[i*51:(i+1)*51]-g_healthy_m[i*51:(i+1)*51]))) for i in range(p)])
    GVS_r_healthy = np.array([np.sqrt(np.mean(np.square(g_r_healthy[i*51:(i+1)*51]-g_healthy_m[i*51:(i+1)*51]))) for i in range(p)])

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

    V_GPS_gesamt_o_healthy_c = rms([
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

    print(V_GPS_gesamt_o_healthy_c)