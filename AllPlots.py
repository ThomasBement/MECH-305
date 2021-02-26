"""
MECH 305/6 DATA ANALYSIS
----------------------------------------
WRITTEN BY: THOMAS BEMENT
DATE: 2/25/2021

Takes a while to run but eventualy chuggs through it
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

"""
DEFINE FUNCTIONS
"""
# For curve fitting tempature decay
def objective1(x, a, b, c): 
    return (a*np.exp(-b*x)) + c

# For curve fitting heat transfer coefficents
def objective2(x, a, b, c): 
    return a + b*(x) + c*(x**2)

# Power from voltage (3.9 is resistor voltage is mesured over, 3.3 is the voltage regulator output)
def v2Power(arr):
    return (arr/3.9)*(3.3-arr)

# Get flow rate from given parameters, only applies for our orfice setup
def flowRate(CD, Rho, orfD, pipeD, eps, dP):
    return 35.3147*60*(CD/(Rho*math.sqrt(1-(((orfD/1000)/(pipeD/1000))**4))))*((eps*math.pi)/4)*((orfD/1000)**2)*math.sqrt(2*Rho*dP)

"""
DATA PROSCESSING 
"""
# Import and split csv file into arrays
def readDAT(filName, window, order):
    # Reading data while smoothing and splitting columns
    AllDat = pd.read_csv('%s.csv' %filName) 
    Time = np.array(AllDat.time)
    Pressure = np.array(AllDat.P)
    Tempature = savgol_filter(np.array(AllDat.Temp), window, order)
    Voltage = np.array(AllDat.Voltage)
    return [Time, Pressure, Tempature, Voltage]

# Add arrays to list
def appendArrays(lis):
    [Time, Pressure, Tempature, Voltage] = readDAT(lis[0], 201, 3)
    lis.append(Time)
    lis.append(Pressure)
    lis.append(Tempature)
    lis.append(Voltage)
    return

# Get start and stop arrays for a given num
def getSS4Num(lis, arr1, arr2, arr3):
    if (len(arr1) != len(arr2)):
        print('Unequal number of start and stops')
        return
    else:
        start, stop= [], []
        for i in range(len(arr1)):
            if (arr3[i]==lis[1][0]):
                start.append(arr1[i])
                stop.append(arr2[i])
        if (len(start)==0):
            print('Invalid num identifier')
            return       
        lis.append(np.array(start))
        lis.append(np.array(stop))
        return

# Trim tempatures from the chosen start stops
def Trim(lis):
    start = lis[6]
    stop = lis[7]
    tempature = np.array(lis[4], copy=True)
    time = np.array(lis[2], copy=True)
    pres = np.array(lis[3], copy=True)
    V = np.array(lis[5], copy=True)
    # Empty lists for the three pressure ranges
    temp1, temp2, temp3, time1, time2, time3, pres1, pres2, pres3, V1, V2, V3= [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(3):
        for j in range(len(time)):
            if (time[j] >= start[i]) and (time[j] <= stop[i]):
                if i==0:
                    temp1.append(tempature[j])
                    time1.append(time[j])
                    pres1.append(pres[j])
                    V1.append(V[j])
                elif i==1:
                    temp2.append(tempature[j])
                    time2.append(time[j])
                    pres2.append(pres[j])
                    V2.append(V[j])
                else:
                    temp3.append(tempature[j])
                    time3.append(time[j])
                    pres3.append(pres[j])
                    V3.append(V[j])
            elif(time[j] >= stop[i]):
                break
    lis.append(temp1)
    lis.append(temp2)
    lis.append(temp3)
    lis.append(time1)
    lis.append(time2)
    lis.append(time3)
    lis.append(pres1)
    lis.append(pres2)
    lis.append(pres3)
    lis.append(V1)
    lis.append(V2)
    lis.append(V3)
    return

# To save line space for rezeroing data
def reZero(arr):
    return (arr-arr[0])

# Curve Fitting for all 3 tempature sections
def curveFit1(lis):
    defaults = [0, 0.001, 20]
    coeff1, _ = curve_fit(objective1, reZero(lis[11]), lis[8], p0=defaults)
    coeff2, _ = curve_fit(objective1, reZero(lis[12]), lis[9], p0=defaults)
    coeff3, _ = curve_fit(objective1, reZero(lis[13]), lis[10], p0=defaults)
    # Assign coefficents
    a1, b1, c1 = coeff1
    a2, b2, c2 = coeff2
    a3, b3, c3 = coeff3
    """
    print("Coefficents for ", lis[0], ':')
    print("A1: ",a1,"B1: ",b1,"C1: ",c1)
    print("A2: ",a2,"B2: ",b2,"C2: ",c2)
    print("A3: ",a3,"B3: ",b3,"C3: ",c3)
    """
    # calculate the output for the range , d1, e1, f1, g1, h1
    lis.append(objective1(np.array(reZero(lis[11])), a1, b1, c1))
    lis.append(objective1(np.array(reZero(lis[12])), a2, b2, c2))
    lis.append(objective1(np.array(reZero(lis[13])), a3, b3, c3))
    lis.append(np.array(coeff1))
    lis.append(np.array(coeff2))
    lis.append(np.array(coeff3))
    return

# Uses curve fit to calculate aproximate derivative and find styeady state for some specified threshold
def findSS(lis, thres):
    time1 = np.array(reZero(lis[11]))
    time2 = np.array(reZero(lis[12]))
    time3 = np.array(reZero(lis[13])) 
    ss1 = time1[0]
    ss2 = time2[0]
    ss3 = time3[0]
    for i in range(len(time1)):
        if (abs(objective1(time1[i], *lis[23])-objective1(time1[i+1], *lis[23]))<=thres):
            #print(lis[0], ' Index 1: ',i)
            ss1 = time1[i]
            index1 = i
            break
    for i in range(len(time2)):
        if (abs(objective1(time1[i], *lis[24])-objective1(time1[i+1], *lis[24]))<=thres):
            #print(lis[0], ' Index 2: ',i)
            ss2 = time2[i]
            index2 = i
            break
    for i in range(len(time3)):
        if (abs(objective1(time1[i], *lis[25])-objective1(time1[i+1], *lis[25]))<=thres):
            #print(lis[0], ' Index 3: ',i)
            ss3 = time3[i]
            index3 = i
            break
    critPnt1 = [time1[0], ss1, time1[len(time1)-1], int(0), int(index1), int(len(time1)-1)]
    critPnt2 = [time2[0], ss2, time2[len(time2)-1], int(0), int(index2), int(len(time1)-1)]
    critPnt3 = [time3[0], ss3, time3[len(time3)-1], int(0), int(index3), int(len(time1)-1)]
    lis.append(critPnt1)
    lis.append(critPnt2)
    lis.append(critPnt3)
    return

# Calculate averages for: pressure, tempature, power
# Calculate heat transfer coefficents and flow rates
def avgNTC(lis, filname):
    T0 = []
    for i in range(len(lis[2])):
            if (lis[2][i] <= 20):
                T0.append(lis[4][i])
            elif (lis[2][i] >= 20):
                break
    T0_Ave = np.average(np.array(T0))
    radius = lis[1][2]
    length = lis[1][1]
    SA = 2*math.pi*((radius*length)+(radius**2))
    aveT1 = np.average(lis[8][lis[26][4]:lis[26][5]])
    aveT2 = np.average(lis[9][lis[27][4]:lis[27][5]])
    aveT3 = np.average(lis[10][lis[28][4]:lis[28][5]])
    aveP1 = np.average(lis[14][lis[26][4]:lis[26][5]])
    aveP2 = np.average(lis[15][lis[27][4]:lis[27][5]])
    aveP3 = np.average(lis[16][lis[28][4]:lis[28][5]])
    avePow1 = np.average(v2Power(np.array(lis[17][lis[26][4]:lis[26][5]])))
    avePow2 = np.average(v2Power(np.array(lis[18][lis[27][4]:lis[27][5]])))
    avePow3 = np.average(v2Power(np.array(lis[19][lis[28][4]:lis[28][5]])))
    H1 = (avePow1/SA)/(aveT1 - T0_Ave)
    H2 = (avePow2/SA)/(aveT2 - T0_Ave)
    H3 = (avePow3/SA)/(aveT3 - T0_Ave)
    # Open with "a" to append data
    f = open(filname, "a")
    f.write('%s,%.4f,%.2f,%.2f,%.3f,%.2f,%.4f,%.4f,%.4f,%.4f\n' % (lis[0], aveT1, aveP1, avePow1, (lis[26][2]-lis[26][1]), (1/lis[23][1]), lis[23][0], lis[23][1], lis[23][2], H1))
    f.write('%s,%.4f,%.2f,%.2f,%.3f,%.2f,%.4f,%.4f,%.4f,%.4f\n' % (lis[0], aveT2, aveP2, avePow2, (lis[27][2]-lis[27][1]), (1/lis[24][1]), lis[24][0], lis[24][1], lis[24][2], H2))
    f.write('%s,%.4f,%.2f,%.2f,%.3f,%.2f,%.4f,%.4f,%.4f,%.4f\n' % (lis[0], aveT3, aveP3, avePow3, (lis[28][2]-lis[28][1]), (1/lis[25][1]), lis[25][0], lis[25][1], lis[25][2], H3))
    f.close()
    Flow1 = 1e9*flowRate(0.5, 1.2041, lis[1][3], 48/1000, 1, aveP1)
    Flow2 = 1e9*flowRate(0.5, 1.2041, lis[1][3], 48/1000, 1, aveP2)
    Flow3 = 1e9*flowRate(0.5, 1.2041, lis[1][3], 48/1000, 1, aveP3)
    lis.append([Flow1,Flow2,Flow3])
    lis.append([H1,H2,H3])
    return

# Curve Fitting for heat transfer coefficents
def curveFit2(lis):
    defaults = [0, 0.001, 20]
    # Define empty lists for HTC and flow data
    Flow, HTC = lis[0][29], lis[0][30]
    for i in range(1,len(lis)):
        for x in lis[i][29]:
            Flow.append(x)
        for y in lis[i][30]:
            HTC.append(y)
    
    coeff, _ = curve_fit(objective2, Flow, HTC, p0=defaults)
    # Assign coefficents
    a, b, c = coeff
    """
    print("Coefficents for ", Add a variable in function if coefficents are needed, ':')
    print("A1: ",a1,"B1: ",b1,"C1: ",c1)
    """
    # Calculate range for output
    xdist = np.arange(min(Flow), max(Flow), (max(Flow)-min(Flow))/256)
    lis.append([xdist, objective2(xdist, a, b, c)])
    return

"""
PLOTS
"""
# For debugging curve fit issues
def fitDebug(lis):
    defaults = [0, 0.001, 20]
    fig, axs = plt.subplots(3)
    fig.suptitle('Debug Fit')
    fig.tight_layout(pad=3.0)
    # Tempature vs. Time No Block
    axs[0].plot(reZero(lis[11]), lis[8], color="#ff94ff")
    axs[0].plot(reZero(lis[11]), objective1(np.array(reZero(lis[11])), *defaults), '--', color='r')
    axs[0].set_title('Tempature vs. Time No Block')
    axs[0].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time Medium Block
    axs[1].plot(reZero(lis[12]), lis[9], color="#ff94ff")
    axs[1].plot(reZero(lis[12]), objective1(np.array(reZero(lis[12])), *defaults), '--', color='r')
    axs[1].set_title('Tempature vs. Time Medium Block')
    axs[1].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time Large Block
    axs[2].plot(reZero(lis[13]), lis[10], color="#ff94ff")
    axs[2].plot(reZero(lis[13]), objective1(np.array(reZero(lis[13])), *defaults), '--', color='r')
    axs[2].set_title('Tempature vs. Time Large Block')
    axs[2].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Save Plots
    plt.show()

# Plot time serries data
def timeSerriesPlot(lis):
    # Pressure
    fig, axs = plt.subplots(3)
    fig.suptitle('Time series plot: %s' %lis[0])
    fig.tight_layout(pad=3.0)
    # Pressure vs. Time
    axs[0].plot(lis[2], lis[3], color="#9b94ff")
    axs[0].set_title('Pressure vs. Time')
    axs[0].set(xlabel='Time (s)', ylabel='Pressure (Pa)')
    # Tempature vs. Time No Block
    axs[1].plot(lis[2], lis[4], color="#ff94ff")
    axs[1].set_title('Tempature vs. Time')
    axs[1].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time
    axs[2].plot(lis[2], lis[5], color="#ff8969")
    axs[2].set_title('Voltage vs. Time')
    axs[2].set(xlabel='Time (s)', ylabel='Voltage (V)')
    # Save Plots
    plt.savefig('PythonFigures/TimeSeries/%s_TS.png' %lis[0], format='png', bbox_inches='tight', orientation='landscape')
    plt.close()
    return

# Tempature plots for pull pressure, 1/2 pressure and 1/4 pressure
def tempPlot(lis):
    fig, axs = plt.subplots(3)
    fig.suptitle('Tempature plot for: %s' %lis[0])
    fig.tight_layout(pad=3.0)
    # Tempature vs. Time No Block
    axs[0].plot(reZero(lis[11]), lis[8], color="#ff94ff")
    axs[0].plot(reZero(lis[11]), lis[20], '--', color='#ff8969')
    axs[0].scatter(lis[26][0], objective1(lis[26][0], *lis[23]), color="#9b94ff")
    axs[0].scatter(lis[26][1], objective1(lis[26][1], *lis[23]), marker = 'X', color="#9b94ff")
    axs[0].scatter(lis[26][2], objective1(lis[26][2], *lis[23]), marker = '^', color="#9b94ff")
    axs[0].set_title('Tempature vs. Time No Block')
    axs[0].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time Medium Block
    axs[1].plot(reZero(lis[12]), lis[9], color="#ff94ff")
    axs[1].plot(reZero(lis[12]), lis[21], '--', color='#ff8969')
    axs[1].scatter(lis[27][0], objective1(lis[27][0], *lis[24]), color="#9b94ff")
    axs[1].scatter(lis[27][1], objective1(lis[27][1], *lis[24]), marker = 'X', color="#9b94ff")
    axs[1].scatter(lis[27][2], objective1(lis[27][2], *lis[24]), marker = '^', color="#9b94ff")
    axs[1].set_title('Tempature vs. Time Medium Block')
    axs[1].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time Large Block
    axs[2].plot(reZero(lis[13]), lis[10], color="#ff94ff")
    axs[2].plot(reZero(lis[13]), lis[22], '--', color='#ff8969')
    axs[2].scatter(lis[28][0], objective1(lis[28][0], *lis[25]), color="#9b94ff")
    axs[2].scatter(lis[28][1], objective1(lis[28][1], *lis[25]), marker = 'X', color="#9b94ff")
    axs[2].scatter(lis[28][2], objective1(lis[28][2], *lis[25]), marker = '^', color="#9b94ff")
    axs[2].set_title('Tempature vs. Time Large Block')
    axs[2].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Save Plots
    plt.savefig('PythonFigures/TempaturePlots/%s_TMP.png' %lis[0], format='png', bbox_inches='tight', orientation='landscape')
    plt.close()
    return

"""
LARGE LIST FUNCTIONS
"""
# Run function to call all functions above for all data
def fullRun(lis, start, stop, num, thresh):
    Day1_Dat = [lis[0], lis[1], lis[4], lis[5]]
    Day2_Dat = [lis[2], lis[3], lis[6], lis[7]]
    for i in range(len(lis)):
        print('Running %s...' %lis[i][0])
        appendArrays(lis[i])
        getSS4Num(lis[i], start, stop, num)
        Trim(lis[i])
        curveFit1(lis[i])
        findSS(lis[i], thresh)
        avgNTC(lis[i], 'Results.csv')
        timeSerriesPlot(lis[i])
        tempPlot(lis[i])
    curveFit2(Day1_Dat)
    curveFit2(Day2_Dat)
    curveFit2(lis)
    HeatTransPlt(Day1_Dat, 'Day 1 Heat transfer coefficents', 'Day1_HTC.png')
    HeatTransPlt(Day2_Dat, 'Day 2 Heat transfer coefficents', 'Day2_HTC.png')
    HeatTransPlt(AllDat, 'Both days heat transfer coefficents', 'BothDays_HTC.png')
    return

# Plot for heat transfer coefficents
def HeatTransPlt(lis, title, saveas):
    legendLis = []
    xRange = lis[(len(lis)-1)][0]
    curveFitDat = lis[(len(lis)-1)][1]
    plt.title(title)
    legendLis.append('Curve Fit')
    plt.xlabel('Flow Rate (mm^3/s)')
    plt.ylabel('Heat Transfer Coefficent')
    for i in range(len(lis)-1):
        plt.scatter(lis[i][29], lis[i][30])
        # Generate list for legend
        legendLis.append(lis[i][0])
    plt.plot(xRange, curveFitDat, '--', color='#ff8969')
    plt.legend(legendLis, bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.savefig('PythonFigures/TempaturePlots/%s' %saveas, format='png', bbox_inches='tight', orientation='landscape')    
    plt.show()
    return


"""
RUN CODE
"""
# Define cut offs
CutOffs = pd.read_csv('CutOffs.csv')
Start = np.array(CutOffs.start)
Stop = np.array(CutOffs.stop)
Num = np.array(CutOffs.num)

# Define file to import
# List format: ['name',[num], time, pressure, tempature, voltage, start, stop, temp1, temp2, temp3, time1, time2, time3, pres1, pres2, pres3, V1, V2, V3, y_line1, y_line2, y_line3, coeff1, coeff2, coeff3, critPnt1, critPnt2, critPnt3, [Flow], [Heat Coeff] ]
#              [   0  ,  1  ,  2  ,   3     ,    4     ,    5   ,  6   ,  7  ,  8   ,   9  ,  10  ,  11  ,   12 ,  13  ,   14 ,   15 ,  16  ,  17, 18, 19,  20    ,   21   ,   22   ,   23 ,   24  ,   25  ,   26    ,    27   ,    28   ,   29  ,     30       ]

"""
DEFINE INDIVIDUAL DATA LISTS
"""
# Thomas's Data
TBlen = 21.1/1000
TBraid = 5.6/2000
TBid30 = 31.7/2000
TBid18 = 19.1/2000
TB_Day3_30mm = ['TB_Day3_30mm', [30.5, TBlen, TBraid, TBid30]]
TB_Day3_18mm = ['TB_Day3_18mm', [18.5, TBlen, TBraid, TBid18]]
TB_Day4_30mm = ['TB_Day4_30mm', [30.6, TBlen, TBraid, TBid30]]
TB_Day4_18mm = ['TB_Day4_18mm', [18.6, TBlen, TBraid, TBid18]]
# Ryan's Data
RLlen = 21.1/1000
RLraid = 5.6/2000
RLid30 = 31.45/2000
RLid18 = 18.93/2000
RL_Day1_30mm = ['RL_Day1_30mm', [30.3, RLlen, RLraid, RLid30]]
RL_Day1_18mm = ['RL_Day1_18mm', [18.3, RLlen, RLraid, RLid18]]
RL_Day2_30mm = ['RL_Day2_30mm', [30.4, RLlen, RLraid, RLid30]]
RL_Day2_18mm = ['RL_Day2_18mm', [18.4, RLlen, RLraid, RLid18]]

"""
DIFINE LARGE LISTS
"""
# Define lists for day 1, day 2 and both days
AllDat = [TB_Day3_30mm, TB_Day3_18mm, TB_Day4_30mm, TB_Day4_18mm, RL_Day1_30mm, RL_Day1_18mm, RL_Day2_30mm, RL_Day2_18mm]

# Open output CSV file and add in headers
f = open("Results.csv","w+")
f.write('Run Description,Steady State Tempature (C),Average Pressure (Pa),Average Power (W),Average Range (s),Time Constant (s),Coefficent A,Coefficent B,Coefficent C,Heat Transfer Coefficent\n')
f.close()
# Run all data analysis functions (can take a while)
fullRun(AllDat, Start, Stop, Num, 0.0005)