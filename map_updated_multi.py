import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

"""
Create a heatmap of the Wi-Fi signal strength in a room from a dataset file
The heatmap is multi-dimensional; each dimension represents a different Access Point (AP)
Authors: Arthur L.
"""
DATASET_FILENAME = r'\Users\prmanav\Downloads\dataset_3_4.txt'
#'C:\Users\Nikitha M V\OneDrive - Umich\Desktop\pythonProject1\HW\dataset.txt'


def read_dataset_file(filename):
    # Read the dataset file
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    data_points = {}
    #dictionary whose key is the bssid and whose and value is another dictionary
    #that has the x,y coordinates as the key and the signal strength as the value
    
    # Parse the dataset file
    data_points = {}
    for line in lines:
        # Skip empty lines
        if line == '\n':
            continue


        # Parse the line
        line = line.strip()
        line = line.split(',')

        if(line[0][0] == '('): # if header line
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


#create a heatmap of average signal strength for a given bssid
def avg_rssi_heatmap(data_points, bssid):
    pos_rssi_dict = data_points[bssid] #dictionary of x,y coordinates and signal strength for this bssid

    x, y = zip(*pos_rssi_dict.keys())
    signal_strength_lists = pos_rssi_dict.values()
    avg_signal_strengths = []
    for signal_strength_list in signal_strength_lists:
        avg_signal_strengths.append(sum(signal_strength_list)/len(signal_strength_list))

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    zi = griddata((x, y), avg_signal_strengths, (xi[None, :], yi[:, None]), method='cubic')

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, zi, levels=15, cmap='inferno')
    plt.colorbar(label='Signal Strength (dBm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Wi-Fi Signal Strength Heatmap')


if __name__ == "__main__":
    data_points = read_dataset_file(DATASET_FILENAME)
    # print(data_points)
    i=0
    for key in data_points.keys():
        i+=1
        avg_rssi_heatmap(data_points, key) 
        if i>5:
            break
    
    #avg_rssi_heatmap(data_points, "cc:88:c7:41:b1:22:")
    #avg_rssi_heatmap(data_points, "cc:88:c7:42:9f:73:") 
    #avg_rssi_heatmap(data_points, "cc:88:c7:42:9f:73:") 
    plt.show()
     #