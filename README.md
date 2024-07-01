# The Gait Index for Neurological Disorders: GIND

This repository includes a script to calculate the Gait Index for Neurological Disorders: Gind.

If you are only interested in calculating the GIND for your trials, please adapt the variables according to your trial in
varaibles.py.

In requirements.txt you can find the necessary libraries for the needed python environment to run the final script and a short description on how to set up a conda environment with all libraries installed.

After adapting the varaibles.py file to your trial you can run the GIND.py file without any arguments:
open a terminal
navigate to the directory of GIND.py
run GIND.py with: python GIND.py

Currently you have to set the order of Foot_Strikes and Foot_Offs in the variables.py file as described in the file.
You can either find the order by observation or by using the btk library in matlab and extracting the events in matlab.
The indeces in the btk events in Python relate to the time stamps (index=0 => lowest timestamp, last index => highest timestamp)

The GIND.py file will print the final GIND after all calculations are done.

The documentation of the calculations from the GIND.py file can be found in the
calculate_GIND jupyter notebook.
