"""
Please adapt this file with your variables for your test environment.
"""

# Trial is just a string which can be set differntly for intermediate data frames
TRIAL_NAME = "dyn01"


# Path to the dynamic c3d file, output by vicon
# For Windows it might look like: r'C:\Data\Vicon\Gait FullBody\dyn01.c3d'
PATH_TO_DYNAMIC_C3D_FILE = r'C:\UNI\HIWI\TUM\HIWI_SPM\Messdaten\Masterarbeit Data\Data final\Patient 02\Gait FullBody\Gait FullBody_dyn_03.c3d' 

#PATH_TO_DYNAMIC_C3D_FILE = r'C:\UNI\HIWI\TUM\HIWI_SPM\Messdaten\Masterarbeit Data\Data final\Christian\Gait FullBody\dyn01.c3d'


# Path to the static c3d file, output by vicon (same structure as for the dynamic file)
PATH_TO_STATIC_C3D_FILE = r'C:\UNI\HIWI\TUM\HIWI_SPM\Messdaten\Masterarbeit Data\Data final\Patient 02\Gait FullBody\Gait FullBody_static_04.c3d'

# PATH_TO_STATIC_C3D_FILE = r'C:\UNI\HIWI\TUM\HIWI_SPM\Messdaten\Masterarbeit Data\Data final\Christian\Gait FullBody\stat01.c3d'

# Please set both weight and height to the corresponding integers, set in Vicon for the participant. 
WEIGHT = 77 #88
HEIGHT = 1770 # 1860


# Please specify the indexing (order) of the Left and right food strikes and take off events
# Has to be a list of integers. The integers represent the index of the event starting from 0.
# The order of the events is important, as it will be used to calculate the gait parameters.

LEFT_FOOT_STRIKE = [2, 6] # 2, 6
LEFT_FOOT_OFF = [1, 5, 9] # 1, 5

RIGHT_FOOT_STRIKE = [0, 4, 8] # 0, 4
RIGHT_FOOT_OFF = [3, 7] # 3

#ev = {"Left_Foot_Strike": np.array([ev[2].time, ev[6].time]), "Left_Foot_Off": np.array([ev[1].time, ev[5].time]), "Right_Foot_Strike": np.array([ev[0].time, ev[4].time]), "Right_Foot_Off": np.array([ev[3].time])}