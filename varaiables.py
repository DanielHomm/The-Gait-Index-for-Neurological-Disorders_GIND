"""
Please adapt this file with your variables for your test environment.
"""

# Trial is just a string which can be set differntly for intermediate data frames
TRIAL_NAME = "dyn01"

# Set PLOT to True or False to choose if you want the GIND plotted in a simple graph (PLOT = True), or to just print the value of the GIND.
PLOT = True


# Path to the dynamic c3d file, output by vicon
# For Windows it might look like: r'C:\Data\Vicon\Gait FullBody\dyn01.c3d'
PATH_TO_DYNAMIC_C3D_FILE = r'Please enter here your path'

# Path to the static c3d file, output by vicon (same structure as for the dynamic file)
PATH_TO_STATIC_C3D_FILE = r'Please enter here your path'

# Please set both weight and height to the corresponding integers, set in Vicon for the participant. 
WEIGHT = 88 #77
HEIGHT = 1860 #1770

# Please select the foot for the first contact. 
# Can be either "right", "left" or None. If you set it to "right" or "left" the code for LEFT_FOOT_STRIKE, ... will not be used.
FIRST_FOOT = "right"

# If you want to be more specific please set FIRST_FOOT = None above. In this case the indeces below wil be automatically used.
# Please specify now the indexing (order) of the Left and right food strikes and take off events
# Has to be a list of integers. The integers represent the index of the event starting from 0.
# The order of the events is important, as it will be used to calculate the gait parameters.

LEFT_FOOT_STRIKE = [2, 6] # 2, 6
LEFT_FOOT_OFF = [1, 5, 9] # 1, 5

RIGHT_FOOT_STRIKE = [0, 4, 8] # 0, 4
RIGHT_FOOT_OFF = [3, 7] # 3,7 
