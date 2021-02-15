import math
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

# This function is defined here for use later.
def normpdf(x, mu, sigma):
    temp = (x-mu)/sigma
    p = (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-temp**2/2)
    return p

# Function to print out probabilities of each nominal mesurement
def printThetas(thetas, P_thetas):
    np.set_printoptions(precision=3)
    sz = int(np.size(thetas))
    for i in range(sz):
        print('Probability of ', thetas[i], ': ', P_thetas[i])
    return

def Bayes(xs, thetas):
    # Define empty matrix of a given size to fill later
    P_x_theta_all = np.zeros((np.size(xs), np.size(thetas)))
    '''
    This matrix is used to keep track of
    | P(x=5.1|theta=5) P(x=5.1|theta=6) ...|
    | P(x=4.9|theta=5) P(x=4.9|theta=6) ...|
    |       ...              ...        ...|
    '''
    # Calculate probabilities of getting a given nominal 
    # mesurement taking into acount some mesurement information
    for i in range(int(np.size(xs))):
        for j in range(int(np.size(thetas))):
            P_x_theta_all[i, j] = normpdf(xs[i], thetas[j], sigma)
            print('Probability of getting X = ', xs[i], 'given that theta = ', thetas[j], 'is: ', P_x_theta_all[i,j])            
    return P_x_theta_all


# Priors given
'''
Bolt Diameter (mm): 3   4   5   6   7   8   10
Number in Stock:   200  10  6  200  15 100  200
'''

''' --- DEFINE CONSTANTS --- '''
# Mesurements
xs = np.array([5.5, 5., 4.9, 5.6])  
# Nominal dimensions
thetas = np.array([3, 4, 5, 6, 7, 8, 10])
# Number of bolts in each category
num_theta = np.array([200, 10, 6, 200, 15, 100, 200])
# Assign Sigma for Assignment Question
sigma = 10

''' --- PERFORM CALCULATIONS --- '''
# Probability of getting a given nominal bolt mesurement
P_thetas = num_theta / np.sum(num_theta)
# P_thetas = np.full((np.size(thetas)), 1/np.size(xs))
# Print probabilities of getting a given nominal bolt mesurement
printThetas(thetas, P_thetas)
P_x_theta_all = Bayes(xs, thetas)
P_x1x2_theta = np.prod(P_x_theta_all, axis=0)

'''
"0" means we do product of elements on the 1st axis (by row)
So we end up with the following vector:
[P([x1=5.1,x2=4.9]|theta=5), P([x1=5.1,x2=4.9]|theta=6), ...]
'''
# Do matrix multiplication here
P_xs = np.dot(P_thetas, P_x1x2_theta)   
# Do element-wise multiplication here
P_thetas_xs = P_thetas*P_x1x2_theta / P_xs   
''' 
We end up with the following vector as our final result:
[P([x1,x2]|theta=5)*P(theta=5)/P([x1,x2]), P([x1,x2]|theta=6)*P(theta=6)/P([x1,x2])]
which is =
[P(theta=5|[x1, x2]), P(theta=6|[x1, x2])]
'''
plt.plot(thetas, P_thetas, 'b*--',  linewidth=2, markersize=10)
plt.plot(thetas, P_thetas_xs, 'ro--',  linewidth=2, markersize=10)
ax = plt.gca()
ax.legend(('Prior Probability','Posterior Probability'))
ax.set_xlabel('Bolt Diameter / mm')
ax.set_ylabel('Probability')
ax.set_title('Two Measurements (Two Bolt Categories)')
plt.savefig('PriorNPost_10.png', format='png', bbox_inches='tight')
plt.show()