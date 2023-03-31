import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

"""
Create a heatmap of the Wi-Fi signal strength in a room from a dataset file
The heatmap is multi-dimensional; each dimension represents a different Access Point (AP)
Authors: Arthur L.
"""
DATASET_FILENAME = r'\Users\nikithaveera\Library\CloudStorage\OneDrive-Umich\Desktop\pythonProject1\HW\dataset.txt'
#'C:\Users\Nikitha M V\OneDrive - Umich\Desktop\pythonProject1\HW\dataset.txt'


def read_dataset_file(filename):
    # Read the dataset file
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    # Parse the dataset file
    data_points = {}
    #dictionary whose key is the bssid and whose and value is another dictionary
    #that has the x,y coordinates as the key and the signal strength as the value
    for line in lines:
        # Skip empty lines
        if line == '\n':
            continue


        # Parse the line
        line = line.strip()
        line = line.split(',')

        if(line[0][0] == '('):
            x, y = float(line[0][1:]), float(line[1][:-1])
            timestamp = line[2]
        elif(line[0][0].isalnum()):
            networks = line[0:]
            # Parse the networks
            for network in networks:
                network = network.split()
                bssid = network[0]
                signal_strength = float(network[1])
                ssid = ' '.join(network[2:])

                try:
                    # if dictionary is empty, create one  
                    # otherwise append to the dictionary
                    data_points[bssid]
                except KeyError:
                    data_points[bssid]={}
                try:
                    data_points[bssid][(x,y)]
                except KeyError:
                    data_points[bssid][(x,y)] = []
                data_points[bssid][(x,y)].append(signal_strength)

    return data_points





def create_heatmap(data_points):
    # Convert the data points into a grid
    x, y, signal_strength = zip(*data_points)
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    zi = griddata((x, y), signal_strength, (xi[None, :], yi[:, None]), method='cubic')

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, zi, levels=15, cmap='inferno')
    plt.colorbar(label='Signal Strength (dBm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Wi-Fi Signal Strength Heatmap')

    plt.show()

if __name__ == "__main__":
    data_points = read_dataset_file(DATASET_FILENAME)
    print(data_points)
   # print(ssid)
    create_heatmap(data_points)