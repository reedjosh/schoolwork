#!/usr/bin/env python3
import numpy as np

if False:    
    DATADIR = "p2_data/"
    
    ID = np.array([[3/10, 7/10]])
    TM = np.array([[7/8, 1/8],
                   [2/8, 6/8]])
    
    # Pi sub 2 as given in the problem.
    PI2 = np.array([[2/3, 1/3]])
    
    print(ID@TM)
    print(PI2@TM)
    
    np.savetxt(DATADIR+'TM.csv', TM)
    np.savetxt(DATADIR+'ID.csv', ID)
    np.savetxt(DATADIR+'PI2.csv', PI2)


if False:
    DATADIR = "p2_data/"
    
    ID = np.array([[3/10, 7/10]])
    TM = np.array([[7/8, 1/8],
                   [2/8, 6/8]])
    
    # Pi sub 2 as given in the problem.
    PI2 = np.array([[2/3, 1/3]])
    
    print(ID@TM)
    print(PI2@TM)
    
    np.savetxt(DATADIR+'TM.csv', TM)
    np.savetxt(DATADIR+'ID.csv', ID)
    np.savetxt(DATADIR+'PI2.csv', PI2)
   
if False:
    ID = np.array([[6/10, 4/10]])
    TM = np.array([[0.37, 0.63],
                   [0.18, 0.82]])
    
    print(ID@TM)
    

if True:
    DATADIR = "p5_data/"
    ID = np.array([[1, 0, 0]])
    TM = np.array([[2/3, 1/3, 0],
                   [1/3, 1/3, 1/3],
                   [0, 1/3, 2/3]])
    TM3 = TM@TM@TM 
    OD = ID@TM3
    
    print(TM3)
    print(OD)
    print(OD[0][2])
    print(1-OD[0][2])
    np.savetxt(DATADIR+'TM3.csv', TM3)
    np.savetxt(DATADIR+'OD.csv', OD)



