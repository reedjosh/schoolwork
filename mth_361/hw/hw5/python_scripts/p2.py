#!/usr/bin/env python3
import numpy as np

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
