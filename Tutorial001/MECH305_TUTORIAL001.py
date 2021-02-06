# Import libraries
import numpy as np # linear algebra
import math as m # Math ( for sqrt() )
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Read in data
data = pd.read_csv("real_raw_data.csv") # Read in data file from same diretory as .py program

# Remove any mm in data to repair data points
for col in data.columns:
    data[col] = data[col].map(lambda x: x if type(x)==float else float(x.replace('mm', '')))
data.info()

# Impose filters and count removed outliers
all_data = [np.array(data[colnm]) for colnm in ["ID measurement 1", "ID measurement 2", "ID measurement 3", "ID measurement 4", "Thickness measurement 1", "Thickness measurement 2"]] 
all_data_new, total_removed = [], 0
for col in all_data:
    keep_indices = ((0.3<col)&(col<0.7))|((30.5<col)&(col<32.5)) # filter extreme values
    total_removed += np.sum(1 - keep_indices)
    all_data_new.append(col[keep_indices])

# Create new set with student data to compare student mean values for measurements
student_data = data
i = 0
for i in range(len(student_data.index)):
    okay = True
    for col in student_data:
        val = student_data[col][i]
        okay &= ((0.3<val)&(val<0.7))|((30.5<val)&(val<32.5))
    if not okay:
        student_data = student_data.drop(index=i, axis=0)
            

# Create arrays for all measurements for further analysis
inner_diameter = np.concatenate(all_data_new[slice(4)])
thickness = np.concatenate(all_data_new[slice(4,6)])
student_data = np.array(student_data)
stu_id = np.stack(student_data[:, :4])
stu_t = np.stack(student_data[:, 4:])


# Take the mean of the student data for diameters to get average value arrays
dim = np.shape(stu_id)
stu_id_mean = np.empty(dim[0], dtype=float)

for i in range(dim[0]):
    temp = 0
    for j in range(dim[1]):
        temp += stu_id[i][j]
    stu_id_mean[i] = (temp/dim[1])

# Take the mean of the student data for diameters to get average value arrays
dim = np.shape(stu_t) 
stu_t_mean = np.empty(dim[0], dtype=float)

for i in range(dim[0]):
    temp = 0
    for j in range(dim[1]):
        temp += stu_t[i][j]
    stu_t_mean[i] = (temp/dim[1])


# Print mean, min/max and standard diviation values for measurements
print("ID Mean: ", np.mean(inner_diameter))
print("ID Standard Diviation ", np.std(inner_diameter))
print("ID Min: ", np.amin(inner_diameter))
print("ID Max: ", np.amax(inner_diameter))
print("T Mean: ", np.mean(thickness))
print("T Standard Diviation ", np.std(thickness))
print("T Min: ", np.amin(thickness))
print("T Max: ", np.amax(thickness))

# Print number of outliers removed
print("Number of Removed Outliers: ", total_removed)

# Plot all data after clean up
for col in all_data_new:
    plt.plot(col, 'o')
plt.title('All Measurements')
plt.xlabel('Measurement #')
plt.ylabel("Length / mm")
plt.show()

# Plot histogram of ID
plt.hist(inner_diameter, bins=18)
plt.title('Histogram of Inner Diameter')
plt.xlabel('Inner Diameter/ mm')
plt.ylabel('Number of Measurements')
plt.show()

# Plot histogram of Thickness
plt.hist(thickness, bins=18)
plt.title('Histogram of Thickness')
plt.xlabel('Thickness/ mm')
plt.ylabel('Number of Measurements')
plt.show()

# Plot historgram of student means
plt.hist(stu_id_mean, bins=5)
plt.title('Histogram of Diameter Means Per Student')
plt.xlabel('Mean Diameter / mm')
plt.ylabel('Number of Students')
plt.show()

plt.hist(stu_t_mean, bins=5)
plt.title('Histogram of Thickness Means Per Student')
plt.xlabel('Mean Thickness / mm')
plt.ylabel('Number of Students')
plt.show()
