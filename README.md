# The Gait Index for Neurological Disorders: GIND

This repository includes a script to calculate the Gait Index for Neurological Disorders: Gind.

If you are only interested in calculating the GIND for your trials, please only adapt the variables according to your trial in
varaibles.py.

In requirements.txt you can find the necessary libraries for the needed python environment to run the final script and a short description on how to set up a conda environment with all libraries installed.

After adapting the varaibles.py file to your trial you can run the GIND.py file without any arguments:
open a terminal
navigate to the directory of GIND.py
run GIND.py with: python GIND.py

Currently you have to set either which foot striked first or the order of Foot_Strikes and Foot_Offs in the variables.py file as described in the file.
If you choose to use the exact order of strike and foot offs. You can find out the order by observation or by using the btk library in Matlab and extracting the events in Matlab. For this case, the indices in the btk events in Python relate to the time stamps (index=0 => lowest timestamp, last index => highest timestamp)

The GIND.py file will print the final GIND after all calculations are done.

The documentation of the calculations from the GIND.py file can be found in the
calculate_GIND jupyter notebook.



The c3d files for our trials included the following measurements. Recording, e.g., from EMG, is not used for the GIND calculation. However, a similar structure will ensure a correct result.:

**Angles**: 'LHipAngles', 'LKneeAngles', 'LAbsAnkleAngle', 'LAnkleAngles', 'RHipAngles', 'RKneeAngles', 'RAnkleAngles', 'RAbsAnkleAngle', 'LPelvisAngles', 'RPelvisAngles', 'LFootProgressAngles', 'RFootProgressAngles', 'RNeckAngles', 'LNeckAngles', 'RSpineAngles', 'LSpineAngles', 'LShoulderAngles', 'LElbowAngles', 'LWristAngles', 'RShoulderAngles', 'RElbowAngles', 'RWristAngles', 'RThoraxAngles', 'LThoraxAngles', 'RHeadAngles', 'LHeadAngles'



**Forces**: 'LGroundReactionForce', 'LNormalisedGRF', 'RGroundReactionForce', 'RNormalisedGRF', 'LAnkleForce', 'RAnkleForce', 'RKneeForce', 'LKneeForce', 'RHipForce', 'LHipForce', 'LWaistForce', 'RWaistForce', 'LNeckForce', 'RNeckForce', 'LShoulderForce', 'RShoulderForce', 'LElbowForce', 'RElbowForce', 'LWristForce', 'RWristForce'



**Moments**: 'LGroundReactionMoment', 'RGroundReactionMoment', 'LAnkleMoment', 'RAnkleMoment', 'RKneeMoment', 'LKneeMoment', 'RHipMoment', 'LHipMoment', 'LWaistMoment', 'RWaistMoment', 'LNeckMoment', 'RNeckMoment', 'LShoulderMoment', 'RShoulderMoment', 'LElbowMoment', 'RElbowMoment', 'LWristMoment', 'RWristMoment'



**Power**: 'LHipPower', 'LKneePower', 'LAnklePower', 'RHipPower', 'RKneePower', 'RAnklePower', 'LWaistPower', 'RWaistPower', 'LNeckPower', 'RNeckPower', 'LShoulderPower', 'RShoulderPower', 'LElbowPower', 'RElbowPower', 'LWristPower', 'RWristPower'



**Static Analysis**: 'Force.Fx1', 'Force.Fy1', 'Force.Fz1', 'Moment.Mx1', 'Moment.My1', 'Moment.Mz1', 'Force.Fx2', 'Force.Fy2', 'Force.Fz2', 'Moment.Mx2', 'Moment.My2', 'Moment.Mz2', 'EMG.GM_L', 'EMG.RF_R', 'EMG.Tensor_L', 'EMG.RF_L', 'EMG.BF_L', 'EMG.BF_R', 'EMG.GM_R', 'EMG.Tensor_R'



**Analysis from Vicon**: 'Cadence', 'Walking Speed', 'Stride Time', 'Step Time', 'Opposite Foot Off', 'Opposite Foot Contact', 'Foot Off', 'Single Support', 'Double Support', 'Stride Length', 'Step Length', 'Step Width', 'Limp Index'



