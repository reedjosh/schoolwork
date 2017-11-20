#!/usr/bin/env python3
import numpy as np

DATADIR = "p1_data/"

ID = np.array([[0,0,1,0,0]])
TM = np.array([[  1,   0,   0,   0,   0],
               [1/2,   0, 1/2,   0,   0],
               [  0, 1/2,   0, 1/2,   0],
               [  0,   0, 1/2,   0, 1/2],
               [  0,   0,   0,   0,   1]])

TM2 = TM@TM
OD = ID@TM2
print(TM2)
print(OD)

np.savetxt(DATADIR+'TM.csv', TM)
np.savetxt(DATADIR+'TM2.csv', TM2)
np.savetxt(DATADIR+'ID.csv', ID)
np.savetxt(DATADIR+'OD.csv', OD)
