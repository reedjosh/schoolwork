"""
doc
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
from scipy.stats import norm

SIGMA = 1.0
def integral(k):
    """Integrates the area under a gaussian curve k sigma from the mean."""
    result = lambda k: integrate.quad(lambda x: 1/(math.sqrt( \
        2*scipy.pi*SIGMA**2))*math.exp((-1/2)*((x-0)/SIGMA)**2), -100, -1)
    return result(k)[0]  # Scipy returns a tuple. Only return element 0.

def to_probability(x_val):
    """Converts from area under curve at a distance of k*sigma to a probability
    of actually being that distance from the curve."""
    return 1-2*x_val

print(integral(2))
print(1./2.-integral(2))
#X_AXIS = np.arange(8)
#print(X_AXIS)
#Y_AXIS = np.array([to_probability(integral(num)) for num in X_AXIS])
#print(Y_AXIS)
#
#plt.plot(X_AXIS, Y_AXIS)
#plt.ylabel('Probability')
#plt.xlabel(r'$K\sigma$ distance from the mean.')
#plt.title(r'Gaussian Distribution Probability $k\sigma$ from the mean')
##plt.savefig('img/plot_gaussian_ksigma.png')
#plt.show()





X_AXIS = np.arange(-10, 10, 0.01)
plt.plot(X_AXIS, norm.pdf(X_AXIS, 1, 4))
plt.show()
