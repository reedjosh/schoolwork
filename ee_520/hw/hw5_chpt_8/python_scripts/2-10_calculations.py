import scipy
import scipy.integrate as integrate
import numpy as np
import math
import matplotlib.pyplot as plt




a = -1.0/(2.0*(scipy.exp(-4)-scipy.exp(-1)))
print("A is: {}".format(a))

integral = integrate.quad(lambda x: a*scipy.exp(-x),1,4)[0]
print("The area under Ae^-1 for 1 to 4 is: {}".format(integral))

integral = integrate.quad(lambda x: a*scipy.exp(-x),2,3)[0]
print("The area under Ae^-1 for 2 to 3 is: {}".format(integral))

integral = integrate.quad(lambda x: a*scipy.exp(-x),1,3)[0]
print("The area under Ae^-1 for 1 to 3 is: {}".format(integral))



x = np.arange(1,4, 0.1)
print(x)
func = lambda x: integrate.quad(lambda x:a*scipy.exp(-x),1,x)[0]
y = np.array([func(num) for num in x])
print(y)

plt.plot(x,y)    
plt.title("$Ae^{-x}")
#plt.savefig('img/plot_2-10_aex.png')
plt.show()
