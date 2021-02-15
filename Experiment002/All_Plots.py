import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For plotting results
import numpy as np # For analysis of data and matrix/vector opperations
import math # For complex curve fitting functions and mass flux
from scipy.optimize import curve_fit, fmin # Curve fitting function to try and match a function to a data set
from scipy import odr  
"""
DEFINE CONSTANTS
"""
CD = 0.5
epsilon = 1
density = 1.2041
pipe_diam = 48
errD = 0.05
errP = 10


"""
DEFINE FUNCTIONS
"""
'''
# Define function to try and fit data with
def objective(x, a, b, c): #, f, g, h
	return a + (b * x) + (c * x**2) #+ (d * x**3) + (e * x**4) + (f * x**5) + (g * x**6) + (h * x**7)
'''

def objective(p, x): 
    a, b, c = p
    return a + (b * x) + (c * x**2)

# Define function for calulating mass flux (m^3/min)
def flowRate(CD, Rho, orfD, pipeD, eps, dP):
    return 35.3147*60*(CD/(Rho*math.sqrt(1-(((orfD/1000)/(pipeD/1000))**4))))*((eps*math.pi)/4)*((orfD/1000)**2)*math.sqrt(2*Rho*dP)

# Define function for error propagation
def errProp(CD, Rho, orfD, pipeD, eps, dP, sigd, sigdP):
    return math.sqrt(((2000*sigd*(CD/(Rho*math.sqrt(1-(((orfD/1000)/(pipeD/1000))**4))))*((eps*math.pi)/4)*(orfD/1000)*math.sqrt(2*Rho*dP))**2)+
    (1000*sigdP*(CD/(Rho*math.sqrt(1-(((orfD/1000)/(pipeD/1000))**4))))*((eps*math.pi)/4)*((orfD/1000)**2)*(1/2*math.sqrt(2*Rho*dP)))**2)

def errPropBetter(fn, val_list, err_list):
    ans = 0.
    for i in range(len(err_list)):
        args = [val for val in val_list] # Copy val_list to args with no connection
        fn_a = fn(*args) # Expand list to pass to fn
        args[i] += err_list[i] # Add error to args
        fn_b = fn(*args) # New result of fn with error added to args
        ans += (fn_a-fn_b)**2 # Use finite difference to estimate derivative
    return np.sqrt(ans)

    

"""
DATA PROSCESSING 
"""

# Define file to import
filName1 = 'Exp2_Dat_TB.csv'
filName2 = 'Exp2_Dat_AM.csv'
filName3 = 'Exp2_Dat_WC.csv'
filName4 = 'Exp2_Dat_RL.csv'
filName5 = 'Exp2_Dat_XL.csv'

def fillPoints(arr, ID, errD, errP):
    sz = int(np.size(arr)/2)
    Flow = np.zeros(sz)
    FlowErr = np.zeros(sz)
    FanHead = np.zeros(sz)
    HeadErr = np.zeros(sz)
    step = 0
    for i in range(sz):
        Flow[i] = flowRate(CD, density, ID, pipe_diam, epsilon, arr[step])
        if Flow[i] <= 2:
            FlowErr[i] = errPropBetter(flowRate, [CD, density, ID, pipe_diam, epsilon, arr[step]], [0, 0, errD, 0, 0, 2*errP])
            HeadErr[i] = 0.10197162129779283*errP
            FanHead[i] = 0.10197162129779283*arr[step+1]
            step = step+2 
        else:
            FlowErr[i] = errPropBetter(flowRate, [CD, density, ID, pipe_diam, epsilon, arr[step]], [0, 0, errD, 0, 0, Flow[i]*errP])
            HeadErr[i] = 0.10197162129779283*errP
            FanHead[i] = 0.10197162129779283*arr[step+1]
            step = step+2    
    return [Flow, FanHead, FlowErr, HeadErr]

# Join 4 arrays into 2 
def joinAll(arr1, arr2, arr3, arr4):
    Flow_Dat = np.concatenate((arr1, arr2), axis=0)
    Head_Dat = np.concatenate((arr3, arr4), axis=0)
    return Flow_Dat, Head_Dat

# Subtracts 4 arrays into 2
def subAll(arr1, arr2, arr3, arr4):
    # Arr1 and arr2 can have zero flow, in this case the error will be zero
    sz = int(np.size(arr1))
    sub1 = np.zeros(sz)
    sub2 = np.zeros(sz)
    for i in range(sz):
        if (arr1[i]==0) or (arr2[i]==0):
            sub1[i]=0
            sub2[i] = abs((arr3[i]-arr4[i])/((arr3[i]+arr4[i])/2))
        else:
           sub1[i] = abs((arr1[i]-arr2[i])/((arr1[i]+arr2[i])/2))
           sub2[i] = abs((arr3[i]-arr4[i])/((arr3[i]+arr4[i])/2))
    return sub1, sub2

# Import and split csv file into arrays
def readDAT(filName):
    # Deading data and splitting columns
    AllDatFil001 = pd.read_csv(filName) 
    Orrifice1 = np.array(AllDatFil001.Orrifice1)
    Orrifice2 = np.array(AllDatFil001.Orrifice2)
    Orrifice3 = np.array(AllDatFil001.Orrifice3)
    # Constant diameters
    ID_Orf1 = Orrifice1[0]
    ID_Orf2 = Orrifice2[0]
    ID_Orf3 = Orrifice3[0]
    # Slices
    Orrifice1_Day1 = Orrifice1[1:9]
    Orrifice2_Day1 = Orrifice2[1:9]
    Orrifice3_Day1 = Orrifice3[1:9]
    Orrifice1_Day2 = Orrifice1[9:17]
    Orrifice2_Day2 = Orrifice2[9:17]
    Orrifice3_Day2 = Orrifice3[9:17]
    # Day 1
    [Orrifice1_Day1_Flow, Orrifice1_Day1_FanHead, Orrifice1_Day1_FlowErr, Orrifice1_Day1_HeadErr] = fillPoints(Orrifice1_Day1, ID_Orf1, errD, errP)
    [Orrifice2_Day1_Flow, Orrifice2_Day1_FanHead, Orrifice2_Day1_FlowErr, Orrifice2_Day1_HeadErr] = fillPoints(Orrifice2_Day1, ID_Orf2, errD, errP)
    [Orrifice3_Day1_Flow, Orrifice3_Day1_FanHead, Orrifice3_Day1_FlowErr, Orrifice3_Day1_HeadErr] = fillPoints(Orrifice3_Day1, ID_Orf3, errD, errP)
    # Day 2
    [Orrifice1_Day2_Flow, Orrifice1_Day2_FanHead, Orrifice1_Day2_FlowErr, Orrifice1_Day2_HeadErr] = fillPoints(Orrifice1_Day2, ID_Orf1, errD, errP)
    [Orrifice2_Day2_Flow, Orrifice2_Day2_FanHead, Orrifice2_Day2_FlowErr, Orrifice2_Day2_HeadErr] = fillPoints(Orrifice2_Day2, ID_Orf2, errD, errP)
    [Orrifice3_Day2_Flow, Orrifice3_Day2_FanHead, Orrifice3_Day2_FlowErr, Orrifice3_Day2_HeadErr] = fillPoints(Orrifice3_Day2, ID_Orf3, errD, errP)
    # Condensing
    Day1_Flow_Dat = np.concatenate((Orrifice1_Day1_Flow,Orrifice2_Day1_Flow,Orrifice3_Day1_Flow), axis=0)
    Day1_Head_Dat = np.concatenate((Orrifice1_Day1_FanHead,Orrifice2_Day1_FanHead,Orrifice3_Day1_FanHead), axis=0)
    Day1_FlowErr_Dat = np.concatenate((Orrifice1_Day1_FlowErr,Orrifice2_Day1_FlowErr,Orrifice3_Day1_FlowErr), axis=0)
    Day1_HeadErr_Dat = np.concatenate((Orrifice1_Day1_HeadErr,Orrifice2_Day1_HeadErr,Orrifice3_Day1_HeadErr), axis=0)

    Day2_Flow_Dat = np.concatenate((Orrifice1_Day2_Flow,Orrifice2_Day2_Flow,Orrifice3_Day2_Flow), axis=0)
    Day2_Head_Dat = np.concatenate((Orrifice1_Day2_FanHead,Orrifice2_Day2_FanHead,Orrifice3_Day2_FanHead), axis=0)
    Day2_FlowErr_Dat = np.concatenate((Orrifice1_Day2_FlowErr,Orrifice2_Day2_FlowErr,Orrifice3_Day2_FlowErr), axis=0)
    Day2_HeadErr_Dat = np.concatenate((Orrifice1_Day2_HeadErr,Orrifice2_Day2_HeadErr,Orrifice3_Day2_HeadErr), axis=0)
    return [Day1_Flow_Dat, Day1_Head_Dat, Day2_Flow_Dat, Day2_Head_Dat, Day1_FlowErr_Dat, Day2_FlowErr_Dat, Day1_HeadErr_Dat, Day2_HeadErr_Dat]

# Use function to read data
[Day1_Flow_Dat1, Day1_Head_Dat1, Day2_Flow_Dat1, Day2_Head_Dat1, Day1_FlowErr_Dat1, Day2_FlowErr_Dat1, Day1_HeadErr_Dat1, Day2_HeadErr_Dat1] = readDAT(filName1)
[Day1_Flow_Dat2, Day1_Head_Dat2, Day2_Flow_Dat2, Day2_Head_Dat2, Day1_FlowErr_Dat2, Day2_FlowErr_Dat2, Day1_HeadErr_Dat2, Day2_HeadErr_Dat2] = readDAT(filName2)
[Day1_Flow_Dat3, Day1_Head_Dat3, Day2_Flow_Dat3, Day2_Head_Dat3, Day1_FlowErr_Dat3, Day2_FlowErr_Dat3, Day1_HeadErr_Dat3, Day2_HeadErr_Dat3] = readDAT(filName3)
[Day1_Flow_Dat4, Day1_Head_Dat4, Day2_Flow_Dat4, Day2_Head_Dat4, Day1_FlowErr_Dat4, Day2_FlowErr_Dat4, Day1_HeadErr_Dat4, Day2_HeadErr_Dat4] = readDAT(filName4)
[Day1_Flow_Dat5, Day1_Head_Dat5, Day2_Flow_Dat5, Day2_Head_Dat5, Day1_FlowErr_Dat5, Day2_FlowErr_Dat5, Day1_HeadErr_Dat5, Day2_HeadErr_Dat5] = readDAT(filName5)

# Day 1 data set
Day1_Flow_Dat = np.concatenate((Day1_Flow_Dat1, Day1_Flow_Dat2, Day1_Flow_Dat3, Day1_Flow_Dat4, Day1_Flow_Dat5), axis=0)
Day1_Head_Dat = np.concatenate((Day1_Head_Dat1, Day1_Head_Dat2, Day1_Head_Dat3, Day1_Head_Dat4, Day1_Head_Dat5), axis=0)
Day1_FlowErr_Dat = np.concatenate((Day1_FlowErr_Dat1, Day1_FlowErr_Dat2, Day1_FlowErr_Dat3, Day1_FlowErr_Dat4, Day1_FlowErr_Dat5), axis=0)
Day1_HeadErr_Dat = np.concatenate((Day1_HeadErr_Dat1, Day1_HeadErr_Dat2, Day1_HeadErr_Dat3, Day1_HeadErr_Dat4, Day1_HeadErr_Dat5), axis=0)
# Day 2 data set
Day2_Flow_Dat = np.concatenate((Day2_Flow_Dat1, Day2_Flow_Dat2, Day2_Flow_Dat3, Day2_Flow_Dat4, Day2_Flow_Dat5), axis=0)
Day2_Head_Dat = np.concatenate((Day2_Head_Dat1, Day2_Head_Dat2, Day2_Head_Dat3, Day2_Head_Dat4, Day2_Head_Dat5), axis=0)
Day2_FlowErr_Dat = np.concatenate((Day2_FlowErr_Dat1, Day2_FlowErr_Dat2, Day2_FlowErr_Dat3, Day2_FlowErr_Dat4, Day2_FlowErr_Dat5), axis=0)
Day2_HeadErr_Dat = np.concatenate((Day2_HeadErr_Dat1, Day2_HeadErr_Dat2, Day2_HeadErr_Dat3, Day2_HeadErr_Dat4, Day2_HeadErr_Dat5), axis=0)
# Difference data arrays
[Flow_Diff_Dat1, Head_Diff_Dat1] = subAll(Day1_Flow_Dat1, Day2_Flow_Dat1, Day1_Head_Dat1, Day2_Head_Dat1)
[Flow_Diff_Dat2, Head_Diff_Dat2] = subAll(Day1_Flow_Dat2, Day2_Flow_Dat2, Day1_Head_Dat2, Day2_Head_Dat2)
[Flow_Diff_Dat3, Head_Diff_Dat3] = subAll(Day1_Flow_Dat3, Day2_Flow_Dat3, Day1_Head_Dat3, Day2_Head_Dat3)
[Flow_Diff_Dat4, Head_Diff_Dat4] = subAll(Day1_Flow_Dat4, Day2_Flow_Dat4, Day1_Head_Dat4, Day2_Head_Dat4)
[Flow_Diff_Dat5, Head_Diff_Dat5] = subAll(Day1_Flow_Dat5, Day2_Flow_Dat5, Day1_Head_Dat5, Day2_Head_Dat5)
Flow_Diff = np.concatenate((Flow_Diff_Dat1, Flow_Diff_Dat2, Flow_Diff_Dat3, Flow_Diff_Dat4, Flow_Diff_Dat5), axis=0)
Head_Diff = np.concatenate((Head_Diff_Dat1, Head_Diff_Dat2, Head_Diff_Dat3, Head_Diff_Dat4, Head_Diff_Dat5), axis=0)
# Total data set for plotting
[Flow_Dat1, Head_Dat1] = joinAll(Day1_Flow_Dat1, Day2_Flow_Dat1, Day1_Head_Dat1, Day2_Head_Dat1)
[Flow_Dat2, Head_Dat2] = joinAll(Day1_Flow_Dat2, Day2_Flow_Dat2, Day1_Head_Dat2, Day2_Head_Dat2)
[Flow_Dat3, Head_Dat3] = joinAll(Day1_Flow_Dat3, Day2_Flow_Dat3, Day1_Head_Dat3, Day2_Head_Dat3)
[Flow_Dat4, Head_Dat4] = joinAll(Day1_Flow_Dat4, Day2_Flow_Dat4, Day1_Head_Dat4, Day2_Head_Dat4)
[Flow_Dat5, Head_Dat5] = joinAll(Day1_Flow_Dat5, Day2_Flow_Dat5, Day1_Head_Dat5, Day2_Head_Dat5)

[FlowErr_Dat1, HeadErr_Dat1] = joinAll(Day1_FlowErr_Dat1, Day2_FlowErr_Dat1, Day1_HeadErr_Dat1, Day2_HeadErr_Dat1)
[FlowErr_Dat2, HeadErr_Dat2] = joinAll(Day1_FlowErr_Dat2, Day2_FlowErr_Dat2, Day1_HeadErr_Dat2, Day2_HeadErr_Dat2)
[FlowErr_Dat3, HeadErr_Dat3] = joinAll(Day1_FlowErr_Dat3, Day2_FlowErr_Dat3, Day1_HeadErr_Dat3, Day2_HeadErr_Dat3)
[FlowErr_Dat4, HeadErr_Dat4] = joinAll(Day1_FlowErr_Dat4, Day2_FlowErr_Dat4, Day1_HeadErr_Dat4, Day2_HeadErr_Dat4)
[FlowErr_Dat5, HeadErr_Dat5] = joinAll(Day1_FlowErr_Dat5, Day2_FlowErr_Dat5, Day1_HeadErr_Dat5, Day2_HeadErr_Dat5)

# Total data set for optimizing
Flow_Dat = np.concatenate((Day1_Flow_Dat, Day2_Flow_Dat), axis=0)
Head_Dat = np.concatenate((Day1_Head_Dat, Day2_Head_Dat), axis=0)
FlowErr_Dat = np.concatenate((Day1_FlowErr_Dat, Day2_FlowErr_Dat), axis=0)
HeadErr_Dat = np.concatenate((Day1_HeadErr_Dat, Day2_HeadErr_Dat), axis=0)



# Curve Fit

# Model object
quad_model = odr.Model(objective)

# Create a RealData object for all data sets
data_day1 = odr.RealData(Day1_Flow_Dat, Day1_Head_Dat, sx=Day1_FlowErr_Dat, sy=Day1_HeadErr_Dat)
data_day2 = odr.RealData(Day2_Flow_Dat, Day2_Head_Dat, sx=Day2_FlowErr_Dat, sy=Day2_HeadErr_Dat)
data_all = odr.RealData(Day1_Flow_Dat, Day1_Head_Dat, sx=FlowErr_Dat, sy=HeadErr_Dat)

# Set up ODR with the model and data.
odr1 = odr.ODR(data_day1, quad_model, beta0=[0., 1., 1.])
odr2 = odr.ODR(data_day2, quad_model, beta0=[0., 1., 1.])
odr3 = odr.ODR(data_all, quad_model, beta0=[0., 1., 1.])

# Run the regression.
out1 = odr1.run()
a1, b1, c1 = out1.beta
out2 = odr2.run()
a2, b2, c2 = out2.beta
out3 = odr3.run()
a3, b3, c3 = out3.beta
print("Coefficents:")
print("A: ",a3,"B: ",b3,"C: ",c3)


'''
curve1, _ = curve_fit(objective, Day1_Flow_Dat, Day1_Head_Dat, p0=[0, 0, -1e6])
a1, b1, c1 = curve1 #, d1, e1, f1, g1, h1 

curve2, _ = curve_fit(objective, Day2_Flow_Dat, Day2_Head_Dat)
a2, b2, c2 = curve2 #, d2, e2, f2, g2, h2

curve3, _ = curve_fit(objective, Flow_Dat, Head_Dat)
a3, b3, c3 = curve3 #, d2, e2, f2, g2, h2
'''

# define a sequence of inputs between the smallest and largest known inputs
x_line1 = np.linspace(min(Day1_Flow_Dat), max(Day1_Flow_Dat)+0.0005, num=256)
x_line2 = np.linspace(min(Day2_Flow_Dat), max(Day2_Flow_Dat)+0.0005, num=256)
x_line3 = np.linspace(min(Flow_Dat), max(Flow_Dat)+0.0005, num=256)
# calculate the output for the range , d1, e1, f1, g1, h1
y_line1 = objective(out1.beta, x_line1)
y_line2 = objective(out2.beta, x_line2)
y_line3 = objective(out3.beta, x_line3)

# Get max product and index arg max
prod1 = x_line1*y_line1
prod2 = x_line2*y_line2
prod = x_line3*y_line3
index1 = np.argmax(prod1)
index2 = np.argmax(prod2)
index3 = np.argmax(prod)
print("Max Power Points:")
print("Day 1,\n","Flow CFM: ", x_line1[index1],"Head mmH2O: ", y_line1[index1])
print("Day 2,\n","Flow CFM: ", x_line2[index2],"Head mmH2O: ", y_line2[index2])
print("Both Days,\n","Flow CFM: ", x_line3[index3],"Head mmH2O: ", y_line3[index3])
"""
PLOTS
"""

# Plot difference scatter plot

plt.errorbar(Flow_Diff_Dat1, Head_Diff_Dat1, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Diff_Dat2, Head_Diff_Dat2, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Diff_Dat3, Head_Diff_Dat3, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Diff_Dat4, Head_Diff_Dat4, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Diff_Dat5, Head_Diff_Dat5, xerr=None, yerr=None, fmt='o')
plt.axhline(y=np.mean(Head_Diff), color='r', linestyle='--')
plt.axvline(x=np.mean(Flow_Diff), color='r', linestyle='--')
plt.title('Normalized Absolute Difference in Flow and Head Between Day 1 and 2')
plt.xlabel('Normalized Absolute Difference of Air Flow')
plt.ylabel('Normalized Absolute Difference of Fan Head')
plt.legend(['Mean Head Abs. Diff.', 'Mean Flow Abs. Diff.','TB', 'AM', 'WC', 'RL', 'XL'], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('DiffPltNorm.png', format='png', bbox_inches='tight')
plt.show()

# Plot scatter plot for day 1
'''
plt.errorbar(Day1_Flow_Dat1, Day1_Head_Dat1, xerr=Day1_FlowErr_Dat1, yerr=Day1_HeadErr_Dat1, fmt='o')
plt.errorbar(Day1_Flow_Dat2, Day1_Head_Dat2, xerr=Day1_FlowErr_Dat2, yerr=Day1_HeadErr_Dat2, fmt='o')
plt.errorbar(Day1_Flow_Dat3, Day1_Head_Dat3, xerr=Day1_FlowErr_Dat3, yerr=Day1_HeadErr_Dat3, fmt='o')
plt.errorbar(Day1_Flow_Dat4, Day1_Head_Dat4, xerr=Day1_FlowErr_Dat4, yerr=Day1_HeadErr_Dat4, fmt='o')
plt.errorbar(Day1_Flow_Dat5, Day1_Head_Dat5, xerr=Day1_FlowErr_Dat5, yerr=Day1_HeadErr_Dat5, fmt='o')
'''
print()
plt.errorbar(Day1_Flow_Dat1, Day1_Head_Dat1, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day1_Flow_Dat2, Day1_Head_Dat2, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day1_Flow_Dat3, Day1_Head_Dat3, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day1_Flow_Dat4, Day1_Head_Dat4, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day1_Flow_Dat5, Day1_Head_Dat5, xerr=None, yerr=None, fmt='o')
plt.plot(x_line1[index1], y_line1[index1], marker='^', markersize=10, color="b")
plt.plot(x_line1, y_line1, '--', color='r')
plt.title('Experimental Fan Flow Curve Day 1')
plt.xlabel('Air Flow (CFM)')
plt.ylabel('Fan Head (mm H20)')
plt.legend(['Max Power Opp.', 'Best Fit Curve', 'TB', 'AM', 'WC', 'RL', 'XL'], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('FanFlowCurveErrTB.png', format='png', bbox_inches='tight')
plt.show()

quit()

# Plot scatter plot for day 2
'''
plt.errorbar(Day2_Flow_Dat1, Day2_Head_Dat1, xerr=Day2_FlowErr_Dat1, yerr=Day2_HeadErr_Dat1, fmt='o')
plt.errorbar(Day2_Flow_Dat2, Day2_Head_Dat2, xerr=Day2_FlowErr_Dat2, yerr=Day2_HeadErr_Dat2, fmt='o')
plt.errorbar(Day2_Flow_Dat3, Day2_Head_Dat3, xerr=Day2_FlowErr_Dat3, yerr=Day2_HeadErr_Dat3, fmt='o')
plt.errorbar(Day2_Flow_Dat4, Day2_Head_Dat4, xerr=Day2_FlowErr_Dat4, yerr=Day2_HeadErr_Dat4, fmt='o')
plt.errorbar(Day2_Flow_Dat5, Day2_Head_Dat5, xerr=Day2_FlowErr_Dat5, yerr=Day2_HeadErr_Dat5, fmt='o')
'''
plt.errorbar(Day2_Flow_Dat1, Day2_Head_Dat1, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day2_Flow_Dat2, Day2_Head_Dat2, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day2_Flow_Dat3, Day2_Head_Dat3, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day2_Flow_Dat4, Day2_Head_Dat4, xerr=None, yerr=None, fmt='o')
plt.errorbar(Day2_Flow_Dat5, Day2_Head_Dat5, xerr=None, yerr=None, fmt='o')
plt.plot(x_line2[index2], y_line2[index2], marker='^', markersize=10, color="b")
plt.plot(x_line2, y_line2, '--', color='r')
plt.title('Experimental Fan Flow Curve Day 2')
plt.xlabel('Air Flow (CFM)')
plt.ylabel('Fan Head (mm H20)')
plt.legend(['Max Power Opp.', 'Best Fit Curve', 'TB', 'AM', 'WC', 'RL', 'XL'], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('FanFlowCurveDay2.png', format='png', bbox_inches='tight', orientation='landscape')
plt.show()

# Plot scatter plot for all data
'''
plt.errorbar(Flow_Dat1, Head_Dat1, xerr=FlowErr_Dat1, yerr=HeadErr_Dat1, fmt='o')
plt.errorbar(Flow_Dat2, Head_Dat2, xerr=FlowErr_Dat2, yerr=HeadErr_Dat2, fmt='o')
plt.errorbar(Flow_Dat3, Head_Dat3, xerr=FlowErr_Dat3, yerr=HeadErr_Dat3, fmt='o')
plt.errorbar(Flow_Dat4, Head_Dat4, xerr=FlowErr_Dat4, yerr=HeadErr_Dat4, fmt='o')
plt.errorbar(Flow_Dat5, Head_Dat5, xerr=FlowErr_Dat5, yerr=HeadErr_Dat5, fmt='o')
'''
plt.errorbar(Flow_Dat1, Head_Dat1, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Dat2, Head_Dat2, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Dat3, Head_Dat3, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Dat4, Head_Dat4, xerr=None, yerr=None, fmt='o')
plt.errorbar(Flow_Dat5, Head_Dat5, xerr=None, yerr=None, fmt='o')
plt.plot(x_line3[index3], y_line3[index3], marker='^', markersize=10, color="b")
plt.plot(x_line3, y_line3, '--', color='r')
plt.title('Experimental Fan Flow Curve All Data')
plt.xlabel('Air Flow (CFM)')
plt.ylabel('Fan Head (mm H20)')
plt.legend(['Max Power Opp.', 'Best Fit Curve', 'TB', 'AM', 'WC', 'RL', 'XL'], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('FanFlowCurveAllDat.png', format='png', bbox_inches='tight', orientation='landscape')
plt.show()

# Plot scatter plot for all data
plt.plot(x_line1[index1], y_line1[index1], marker='^', markersize=10, color="orange")
plt.plot(x_line1, y_line1, '--', color='g')
plt.plot(x_line2[index2], y_line2[index2], marker='^', markersize=10, color="cyan")
plt.plot(x_line2, y_line2, '--', color='b')
plt.plot(x_line3[index3], y_line3[index3], marker='^', markersize=10, color="purple")
plt.plot(x_line3, y_line3, '--', color='r')
plt.title('Experimental Fan Flow Curves for Day 1 & 2 Overlayed on Fan Curve for All Data')
plt.xlabel('Air Flow (CFM)')
plt.ylabel('Fan Head (mm H20)')
plt.legend(['Max Power Opp. Day 1', 'Best Fit Curve Day 1', 'Max Power Opp. Day 2', 'Best Fit Curve Day 2', 'Max Power Opp. All Data', 'Best Fit Curve All Data',], bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.savefig('FanFlowCurveOverlays.png', format='png', bbox_inches='tight', orientation='landscape')
plt.show()
quit()

### Plot Histograms ####

# summarize the parameter values
a, b, c, d, e, f, g, h = curvemass

# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(mass), max(mass), 1/100)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e, f, g, h)

# Plot scatter plot
#plt.axhline(y = Freq3_mean, color='r', linestyle='-')
plt.hist(impactA/impact_mean, bins=5, density=True)
#plt.hist(ForceA/force_mean, bins=5, density=True)
#plt.scatter(massB, impactB)
#plt.scatter(massC, impactC)
#plt.plot(x_line, y_line, '--', color='green')
plt.title('Histogram of Impact')
plt.xlabel('Impact/Impact Mean')
plt.ylabel('Number of Mesurements')
#plt.legend(['Impact/Impact Mean', 'Force/Force Mean'], loc=1)
plt.savefig('ImpactHist.png', format='png', bbox_inches='tight')
plt.show()
