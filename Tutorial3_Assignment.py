# Import useful packages
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt, ceil

#Question 1

# Define constants s and M
s = 1
M = 1
n_bins = 2

# Initiate an empty array eps and fill in the values using a loop
eps = np.empty((M))
for i in range(M):
  X = ((np.random.default_rng().uniform(size=M) > 0.5)*2-1)*(s/sqrt(M)) 
  eps[i] = np.sum(X)

# Define X and Y for plotting
x_space = np.linspace(np.amin(eps), np.amax(eps), M)
y_normal = norm.pdf(x_space, loc=0, scale=1)
#y_normal = (np.amax(eps)/np.amax(y_normal))*y_normal

# Plot the histogram of the N Zi's that we have generated
plt.hist(eps, bins=n_bins, density=True)
plt.plot(x_space, y_normal, linewidth=2, color='r')
plt.title('Histogram of Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Number of Mesurements')
plt.savefig('Q1Hist.png', format='png', bbox_inches='tight')
plt.show()
quit()

#Question 2
print('Question 2\n')

# Define Number of mesurements
N = 10000 

# Define function to generate error and define variables in a clean way
def genError(var0, var_sig, N):
    var = np.random.default_rng().normal(var0, var_sig, N)
    return var0, var_sig, var

# Define variables using function above
L0, L_sig, L = genError(0.7, 0.014, N)
F0, F_sig, F = genError(80, 0.8, N)
B0, B_sig, B = genError(0.04, 0.0004, N)
H0, H_sig, H = genError(0.03, 0.003, N)
Del0, Del_sig, Del = genError(0.008, 0.00008, N)

# Next, compute an array of N 
E = (4*F*(L**3))/(3*Del*B*(H**3)) 
E0 = (4*F0*(L0**3))/(3*Del0*B0*(H0**3)) 

# Compute standeard diviation and mean of E normalized by the E best
E_sdt_rel = np.std(E)/E0
mean_E = np.mean(E)/E0
print("E STD Relative: ", E_sdt_rel)
print("Youngs Modulus: ", E0)
# Plot all measured variables in one histogram
plt.hist(L/L0, density=True)
plt.hist(F/F0, density=True)
plt.hist(B/B0, density=True)
plt.hist(H/H0, density=True)
plt.hist(Del/Del0, density=True)
plt.title('Histogram of All Measured Variables')
plt.xlabel('Ratio of Mesurement to Nominal Value')
plt.ylabel('Number of Mesurements')
plt.legend(['L/L0', 'F/F0', 'B/B0', 'H/H0', 'Del/Del0'], loc=1)
plt.savefig('Q2HistOverly.png', format='png', bbox_inches='tight')      
plt.show()

# Plot the histogram of E/E0
plt.hist(E/E0, density=True)
plt.title('Histogram of Youngs Modulus')
plt.xlabel('E/E0')
plt.ylabel('Number of Mesurements')
plt.savefig('Q2HistEnE0.png', format='png', bbox_inches='tight')
plt.show()