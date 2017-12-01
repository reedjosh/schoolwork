#!/usr/bin/env python3

#import numpy as np
from scipy.integrate import quad as integrate

print(integrate(lambda x: 3/4*(1-x**2),-1,1/2)[0])
print(3/4*((1/2-(1/2)**3/3+2/3)))


