import numpy as np
import matplotlib.pyplot as plt

"""
Create a heatmap of the Wi-Fi signal strength in a room from a dataset file
The heatmap is multi-dimensional; each dimension represents a different Access Point (AP)
Authors: Arthur L.
"""

DATASET_FILENAME = r'C:\Users\arthu\OneDrive\Documents\Classwork\NA568_Mobile_Robotics\project\dataset.txt'

def read_dataset_file(filename):
    # Read the dataset file
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    # Parse the dataset file
    data_points = []
    for line in lines:
        # Skip empty lines
        if line == '\n':
            continue

        # Parse the line
        line = line.strip() #remove whitespace
        line = line.split(',') #split on commas to create a list
        x= float(line[0][1:])
        y= float(line[1][0:-1])
        timestamp = line[2]
        print(x)
        print(y)
        print(timestamp)
        networks = line[3:]
        print(networks)

        # Parse the networks
        for network in networks:
            network = network.split()
            bssid = network[0]
            signal_strength = float(network[1])
            ssid = ' '.join(network[2:])

            # Add the data point to the list
            data_points.append((x, y, signal_strength))

    return data_points

def create_heatmap(data_points):
    # Convert the data points into a grid
    x, y, signal_strength = zip(*data_points)
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    zi = plt.griddata((x, y), signal_strength, (xi[None, :], yi[:, None]), method='cubic')

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
    # create_heatmap(data_points)

