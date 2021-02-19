"""
--------------------------------
MECH-305
--------------------------------
NAME: THOMAS BEMENT
SN: 24099822
TUTORIAL 5 HYPOTHESIS TESTING
--------------------------------
"""
"""
IMPORTS
"""
import math # Math functions
import scipy.stats as spstats # Statistics library
import numpy as np # Array and matrix opperations
import matplotlib.pyplot as plt # Plotting functions
import matplotlib.patches as pat # Plotting functions
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)

"""
DEFINE CONSTANTS
"""
filName1 = 'CompressiveStengthTestData.csv' # First data set
filName2 = 'Grades.csv' # Seccond data set
STD_Range = 4 # For plot range
StepSZ = 0.01 # For plot divisions
alpha1 = 0.05 # Assume 5% for Question 1
alpha2 = 0.05 # Assume 5% for Question 2
mu1 = 500 # Mean for strength
sigma1 = 2 # STD for strength
mu2 = 500 # Mean for grades
sigma2 = 2 # STD for grades
"""
DEFINE FUNCTIONS
"""
# Z score formula (not used for testing if data is normaly distributed)
def Z_Score(arr, mu, sigma):
    x_mean = np.mean(arr)
    print("X Mean: ", x_mean)
    n = np.size(arr)
    return (x_mean-mu)/(sigma/np.sqrt(n))

# Simple hypothesis test function
def hypothesisTest(Z, critZ):
    if (-critZ  < Z and Z < critZ):
        print('Fail to reject H0')
        rej = 0
    else:
        print('Reject H0')
        rej = 1
    return rej

# This function is defined here for use later.
def normpdf(x, mu, sigma):
    temp = (x-mu)/sigma
    p = (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-temp**2/2)
    return p

"""
READ IN DATA
"""
# Read ".CSV" files
Dat1 = pd.read_csv(filName1)
Dat2 = pd.read_csv(filName2)
# Split into arrays to perform data analysis on
Samples = np.array(Dat1.Samples)
Strength = np.array(Dat1.Strength)
Grades = np.array(Dat2)

"""
PERFORM ANALYSIS ON DATA
"""
# Use Scipy function to test distribution
statistics1, p1 = spstats.normaltest(Strength)
statistics2, p2 = spstats.normaltest(Grades)
# Calulate Z_Scores for the hypothesis that 
Z_Score1 = math.sqrt(statistics1/2)
Z_Score2 = math.sqrt(statistics2/2)
# Calculate critical Z_Score value
critZ1 = spstats.norm.ppf(1-alpha1/2)
critZ2 = spstats.norm.ppf(1-alpha2/2)   
# Perform hypothesis testing
print('Question 1:')
_ = hypothesisTest(Z_Score1, critZ1)
print('Question 2:')
_ = hypothesisTest(Z_Score2, critZ2)
print('CIRCLES QUESTION 1:')
print('r: ', math.sqrt(statistics1), 'R: ', math.sqrt(-2*math.log(alpha1)))
print('CIRCLES QUESTION 2:')
print('r: ', math.sqrt(statistics2), 'R: ', math.sqrt(-2*math.log(alpha2)))
print('Question 2:')
print('Mean: ', np.mean(Strength), 'STD: ', np.std(Strength))
"""
FILL ARRAYS
"""
# Question 1
x1 = np.arange(-STD_Range, STD_Range, StepSZ)
x_left1 = np.arange(-STD_Range, -critZ1, StepSZ)
x_right1 = np.arange(critZ1, STD_Range, StepSZ)
# Question 2
x2 = np.arange(-STD_Range, STD_Range, StepSZ)
x2_hist = np.arange(np.amin(Strength), np.amax(Strength), StepSZ)
x_left2 = np.arange(-STD_Range, -critZ1, StepSZ)
x_right2 = np.arange(critZ1, STD_Range, StepSZ)

PDF2 = np.zeros(len(x2_hist))
k=1100
for i in range(len(x2_hist)):
    PDF2[i] = k*normpdf(x2_hist[i], np.mean(Strength), np.std(Strength))

"""
ALL PLOTS
"""
# Question 1 circle plot
circle0 = plt.Circle((0,0), 6, hatch='|', color = '#FF6961')
circle1 = plt.Circle((0,0), math.sqrt(-2*math.log(alpha1)), hatch='/', color='#90EE90')
circle2 = plt.Circle((0,0), math.sqrt(statistics1), color='#6A0DAD', fill=False, linewidth=2.0)
circle3 = plt.Circle((0,0), 1, color='#1D2951', fill=False, linestyle=':')
circle4 = plt.Circle((0,0), 2, color='#1D2951', fill=False, linestyle=':')
circle5 = plt.Circle((0,0), 3, color='#1D2951', fill=False, linestyle=':')
circle6 = plt.Circle((0,0), 4, color='#1D2951', fill=False, linestyle=':')
circle7 = plt.Circle((0,0), 5, color='#1D2951', fill=False, linestyle=':')
fig, ax = plt.subplots() 
ax.set_aspect('equal', 'box')
ax.set_xlim((-1.5*math.sqrt(-2*math.log(alpha1)),1.5*math.sqrt(-2*math.log(alpha1))))
ax.set_ylim((-1.5*math.sqrt(-2*math.log(alpha1)),1.5*math.sqrt(-2*math.log(alpha1))))
ax.add_patch(circle0)
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)
ax.add_patch(circle6)
ax.add_patch(circle7)
plt.title('2D Z-Score Visualization of Strength Distribution')
ax.set_xlabel('Z-Score (Kurtosis)')
ax.set_ylabel('Z-Score (Skew)')
plt.legend(['Reject', 'Fail to Reject','Z Score','Divisions of Z = 1'], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('2D_Z_Score_Strength.png', format='png', bbox_inches='tight')
plt.show()
# Question 1
plt.plot(x1, spstats.norm.pdf(x1))
plt.fill_between(x_left1,spstats.norm.pdf(x_left1))
plt.fill_between(x_right1,spstats.norm.pdf(x_right1))
plt.plot([Z_Score1, Z_Score1], [0, spstats.norm.pdf(Z_Score1)], '-bo')
ax = plt.gca()
plt.title('Standard Normal Distribution for Z-Score')
ax.set_xlabel('Z-Score')
ax.set_ylabel('Probability Density')
plt.savefig('Z_Score_Strength.png', format='png', bbox_inches='tight')
plt.show()
# Question 1 histogram
plt.hist(Strength, bins=15)
plt.plot(x2_hist, PDF2, color='r')
plt.title('Histogram of Specimen Compressive Strength')
plt.xlabel('Compressive Strength (ksi)')
plt.ylabel('Number of Measurements')
plt.legend(['Normal Distribution from Mean and STD','Histogram'], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('StrengthHist.png', format='png', bbox_inches='tight')
plt.show()
quit()
# Question 2 circle plot
circle0 = plt.Circle((0,0), 12, hatch='|', color = '#FF6961')
circle1 = plt.Circle((0,0), math.sqrt(-2*math.log(alpha2)), hatch='/', color='#90EE90')
circle2 = plt.Circle((0,0), math.sqrt(statistics2), color='#6A0DAD', fill=False, linewidth=2.0)
circle3 = plt.Circle((0,0), 1, color='#1D2951', fill=False, linestyle=':')
circle4 = plt.Circle((0,0), 2, color='#1D2951', fill=False, linestyle=':')
circle5 = plt.Circle((0,0), 3, color='#1D2951', fill=False, linestyle=':')
circle6 = plt.Circle((0,0), 4, color='#1D2951', fill=False, linestyle=':')
circle7 = plt.Circle((0,0), 5, color='#1D2951', fill=False, linestyle=':')
circle8 = plt.Circle((0,0), 6, color='#1D2951', fill=False, linestyle=':')
circle9 = plt.Circle((0,0), 7, color='#1D2951', fill=False, linestyle=':')
circle10 = plt.Circle((0,0), 8, color='#1D2951', fill=False, linestyle=':')
circle11 = plt.Circle((0,0), 9, color='#1D2951', fill=False, linestyle=':')
circle12 = plt.Circle((0,0), 10, color='#1D2951', fill=False, linestyle=':')
fig, ax = plt.subplots() 
ax.set_aspect('equal', 'box')
ax.set_xlim((-1.5*math.sqrt(statistics2),1.5*math.sqrt(statistics2)))
ax.set_ylim((-1.5*math.sqrt(statistics2),1.5*math.sqrt(statistics2)))
ax.add_patch(circle0)
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)
ax.add_patch(circle6)
ax.add_patch(circle7)
ax.add_patch(circle8)
ax.add_patch(circle9)
ax.add_patch(circle10)
ax.add_patch(circle11)
ax.add_patch(circle12)
plt.title('2D Z-Score Visualization of Grades Distribution')
ax.set_xlabel('Z-Score (Kurtosis)')
ax.set_ylabel('Z-Score (Skew)')
plt.legend(['Reject', 'Fail to Reject','Z Score','Divisions of Z = 1'], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('2D_Z_Score_Grades.png', format='png', bbox_inches='tight')
plt.show()
# Question 2 Z-Score plot
plt.plot(x2, spstats.norm.pdf(x2))
plt.fill_between(x_left2,spstats.norm.pdf(x_left2))
plt.fill_between(x_right2,spstats.norm.pdf(x_right2))
plt.plot([Z_Score2, Z_Score2], [0, spstats.norm.pdf(Z_Score2)], '-bo')
ax = plt.gca()
plt.title('Standard Normal Distribution for Z-Score')
ax.set_xlabel('Z-Score')
ax.set_ylabel('Probability Density')
plt.savefig('Z_Score_Grades.png', format='png', bbox_inches='tight')
plt.show()
# Question 2 histogram
plt.hist(Grades, bins=12)
plt.title('Histogram of Student Grades')
plt.xlabel('Grades (%)')
plt.ylabel('Number of Measurements')
plt.savefig('GradesHist.png', format='png', bbox_inches='tight')
plt.show()
