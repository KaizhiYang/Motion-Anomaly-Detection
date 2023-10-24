from math import sqrt
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

PATH = "CrowdTracking/Kaizhi's/data.txt"

# Create KDE (Kernel Density Estimate) plot
def KDEgraph(total_mag, total_ang):

    # Create a 2D KDE plot
    sns.kdeplot(x=total_mag, y=total_ang, cmap="bwr", fill=True, levels=8, cbar=True)

    # Limit x axis from 0 to 20
    plt.xlim(0,20)

    # Add labels and a title
    plt.xlabel("Magnitude")
    plt.ylabel("Angle")
    plt.title("2D KDE Plot")
    plt.show()

def dataGetter(file_path):
    data = []
    min_max = (-1,-1)
    
    # Open the file in read mode ('r')
    with open(file_path, 'r') as file:
        # Read each line of the file
        for line in file:
            # Split the line into two values (assuming tab-separated values)
            values = line.strip().split('\t')
            if len(values) == 3:
                # Convert the values to float and append to respective lists
                value1 = float(values[0])
                value2 = float(values[1])
                value3 = float(values[2])
                data.append([value1, value2, value3])
            elif len(values) == 2:
                value1 = float(values[0])
                value2 = float(values[1])
                min_max = [value1, value2]
    print(f"Data has been saved to local variables")
    print("Min and max density: ", min_max)
    print("\n")
    return data, min_max


m = []
a = []
kde_value = []
magnitude_angle, min_max = dataGetter(PATH)
for data in magnitude_angle:
    m.append(data[0])
    a.append(data[1])
    if len(data) == 3:
        kde_value.append(data[2])
# Define the range of your x and y values
m_min, m_max = min(m), max(m)
a_min, a_max = min(a), max(a)
a.pop()
m.pop()
kde_value.pop()
# Define the number of points for the grid
num_points = 100

# Create a grid of points
m_grid, a_grid = np.meshgrid(np.linspace(m_min, m_max, 185), np.linspace(a_min, a_max, 50))

kde_value = np.array(kde_value).reshape(m_grid.shape)


plt.figure(figsize=(8, 6))
plt.contourf(m_grid, a_grid, kde_value, levels=8, cmap='bwr')
plt.colorbar()
plt.xlabel('Magnitude')
plt.ylabel('Angle')
plt.title('2D KDE Plot')
plt.show()

