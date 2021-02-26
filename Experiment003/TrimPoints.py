import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For plotting results
import numpy as np # For analysis of data and matrix/vector opperations
import math # For complex curve fitting functions and mass flux
from scipy.optimize import curve_fit, fmin # Curve fitting function to try and match a function to a data set
from scipy.signal import savgol_filter # For smoothing data
from scipy import odr 

"""
FUNCTIONS
"""
# Import and split csv file into arrays
def readDAT(filName):
    # Deading data and splitting columns
    AllDat = pd.read_csv('%s.csv' %filName) 
    Time = np.array(AllDat.time)
    Pressure = np.array(AllDat.P)
    Tempature = np.array(AllDat.Temp)
    Voltage = np.array(AllDat.Voltage)
    return [Time, Pressure, Tempature, Voltage]

def diffDat(arrDat, arrTime, width):
    DatDiff = abs(arrDat[width:] - arrDat[:-width])
    TimeDiff = abs(arrTime[(width//2):-(width//2)])  
    return DatDiff, TimeDiff

def transitionPoint(arrDat, arrTime, width, numAbove, consec):
    [DatDiff, TimeDiff] = diffDat(arrDat, arrTime, width)
    transitionPoints = []
    countAbove = 0
    for i in range(len(DatDiff)):
        if (DatDiff[i] >= numAbove):
            countAbove += 1
        else:
            if (countAbove >= consec):
                if len(transitionPoints)==0:
                    transitionPoints.append(i - countAbove//2)
                else:
                    if abs((i - countAbove//2)-transitionPoints[len(transitionPoints)-1])<1000: 
                        transitionPoints[len(transitionPoints)-1] = (i - countAbove//2)
                    else:
                        transitionPoints.append(i - countAbove//2)
            countAbove = 0
    return transitionPoints    

# Function to sclice data and smooth it
def sliceNsmooth(arrDat, arrTime, arrIndex, window, order):
    DatSclice1 = np.array(arrDat[:arrIndex[0]], copy=True)
    TimeSclice1 = np.array(arrTime[:arrIndex[0]], copy=True)
    DatSclice2 = savgol_filter(np.array(arrDat[arrIndex[0]:arrIndex[1]], copy=True), window, order)
    TimeSclice2 = np.array(arrTime[arrIndex[0]:arrIndex[1]], copy=True)
    DatSclice3 = savgol_filter(np.array(arrDat[arrIndex[1]:arrIndex[2]], copy=True), window, order)
    TimeSclice3 = np.array(arrTime[arrIndex[1]:arrIndex[2]], copy=True)
    DatSclice4 = savgol_filter(np.array(arrDat[arrIndex[2]:], copy=True), window, order)
    TimeSclice4 = np.array(arrTime[arrIndex[2]:], copy=True)
    return DatSclice1, TimeSclice1, DatSclice2, TimeSclice2, DatSclice3, TimeSclice3, DatSclice4, TimeSclice4

"""
PLOTS
"""

def initialPlot(title, Time, Pressure, Tempature, Voltage):
    # Pressure
    fig, axs = plt.subplots(3)
    fig.suptitle(title)
    fig.tight_layout(pad=3.0)
    # Pressure vs. Time
    axs[0].plot(Time, Pressure, color="#9b94ff")
    axs[0].set_title('Pressure vs. Time')
    axs[0].set(xlabel='Time (s)', ylabel='Pressure (Pa)')
    # Tempature vs. Time No Block
    axs[1].scatter(Time, savgol_filter(Tempature, 201, 3), color="#ff94ff")
    axs[1].set_title('Tempature vs. Time')
    axs[1].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time
    axs[2].plot(Time, Voltage, color="#ff8969")
    axs[2].set_title('Voltage vs. Time')
    axs[2].set(xlabel='Time (s)', ylabel='Voltage (V)')
    plt.savefig('PythonFigures/TrimmingRuns/%s' %title, format='png', bbox_inches='tight', orientation='landscape')
    plt.show()
    return

def tempPlotScat(temp1, temp2, temp3, time1, time2, time3, title, saveas):
    fig, axs = plt.subplots(3)
    fig.suptitle(title)
    fig.tight_layout(pad=3.0)
    # Tempature vs. Time No Block
    axs[0].scatter(time1, temp1, color="#ff94ff")
    axs[0].set_title('Tempature vs. Time No Block')
    axs[0].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time Medium Block
    axs[1].scatter(time2, temp2, color="#ff94ff")
    axs[1].set_title('Tempature vs. Time Medium Block')
    axs[1].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Tempature vs. Time Large Block
    axs[2].scatter(time3, temp3, color="#ff94ff")
    axs[2].set_title('Tempature vs. Time Large Block')
    axs[2].set(xlabel='Time (s)', ylabel='Tempature (C)')
    # Save Plots
    plt.savefig('PythonFigures/TrimmingRuns/%s' %saveas, format='png', bbox_inches='tight', orientation='landscape')
    plt.show()
    return

"""
RUN FUNCTIONS
"""

# Define file to import
filName1 = 'TB_Day4_18mm'

[Time1, Pressure1, Tempature1, Voltage1] = readDAT(filName1)
initialPlot('TimeSerriesPlot', Time1, Pressure1, Tempature1, Voltage1)

# Assign box width
width = 8      
tStep = Time1[1] # May be unused

#[PresIndex1, PresAvg1] = steadyState(Pressure1, 10E-3, 20, tStep)
PressureTP = np.array(transitionPoint(Pressure1, Time1, width, 7, 4))
[PressureDiff, TimeDiff1] = diffDat(Pressure1, Time1, width)

# Run Plots
[TempNBHO, TimeNBHO, TempNB, TimeNB, TempMB, TimeMB, TempLB, TimeLB] = sliceNsmooth(Tempature1, Time1, PressureTP, 201, 3)
tempPlotScat(TempNB, TempMB, TempLB, TimeNB, TimeMB, TimeLB, 'Day-2 Small Orifice Run', '%s_ScatPlt.png' %filName1)