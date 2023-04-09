import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import wifi_dataset_collection as wdc
import pywifi
"""
Create a heatmap of the Wi-Fi signal strength in a room from a dataset file
The heatmap is multi-dimensional; each dimension represents a different Access Point (AP)
Authors: Arthur L.
"""
DATASET_FILENAME_3 = r'.\data\dataset_3_4.txt'

def read_dataset_file(filename, method='bssid'):
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

                if method == 'bssid':
                    if bssid in data_points:
                        if (x,y) in data_points[bssid]:
                            # TODO: change to numpy array
                            data_points[bssid][(x,y)].append(signal_strength)
                        else:
                            data_points[bssid][(x,y)] = [signal_strength]
                    else:
                        data_points[bssid] = {(x,y):[signal_strength]}

                elif method == 'location':
                    if (x,y) in data_points:
                        if bssid in data_points[(x,y)]:
                            data_points[(x,y)][bssid].append(signal_strength)
                        else:
                            data_points[(x,y)][bssid] = [signal_strength]
                    else:
                        data_points[(x,y)] = {bssid:[signal_strength]}
    return data_points


def avg_rssi_heatmap(data_points, bssid):
    """
    create a heatmap of average signal strength for a given bssid
    args:
        data_points: dictionary whose keys are bssid and whose and values are another dictionary
            that has the x,y coordinates as the key and the list of signal strengths as the value
        bssid: string, the bssid of the Access Point (AP) to plot
    """
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
    plt.title(f'Signal Strength at {bssid}')


class NaiveBayesClassifier:
    pass

def location_estimate_naive_bayes(wifi_reading, dataset_filename):
    """
    args:
        wifi_reading: list of WifiNetwork objects, 
            where each one contains its bssid, ssid, and signal_strength
        dataset_filename: string, the name of the dataset file
    returns:
        location_estimate: dictionary whose keys are the (x,y) tuples from data_points, 
            and whose values are the probability that that location is the true location 
            given the wifi_reading

    This function calculates the probability distribution of 
        P(this location is the true location | wifi reading)
    It does this by fitting a log-normal distribution to the signal strength values at each location
    Then, it calculates the probability of the wifi_reading at each location, and normalizes the values
    """
    data_points = read_dataset_file(dataset_filename, method='location')
    location_estimate = {}
    for location, bssid_rssi_dict in data_points.items():
        pass
    
    # # Normalize the values
    # total = sum(location_estimate.values())
    # for location in location_estimate:
    #     location_estimate[location] /= total
    return location_estimate

def location_estimate_avg(wifi_reading, dataset_filename):
    """
    args:
        wifi_reading: list of WifiNetwork objects, 
            where each one contains its bssid, ssid, and signal_strength
        dataset_filename: string, the name of the dataset file
    returns:
        location_estimate: dictionary whose keys are the (x,y) tuples from data_points, 
            and whose values are the probability that that location is the true location 
            given the wifi_reading

    This function calculates the probability distribution of 
        P(this location is the true location | wifi reading)
    It does this by calculating the MSE between the wifi reading (which can be though of as a vector 
    with as many dimensions as there are bssids) and the vector of average RSSI values at the given location
    These values are then normalized across the entire space so they sum to 1.
    """
    data_points = read_dataset_file(dataset_filename, method='location')
    location_estimate = {}
    for location, bssid_rssi_dict in data_points.items():
        diff_vector = [] 
        for network in wifi_reading:
            if network.bssid in bssid_rssi_dict:
                l = len(bssid_rssi_dict[network.bssid])
                if l != 0:
                    avg_dataset_rssi = sum(bssid_rssi_dict[network.bssid])/l
                    diff = avg_dataset_rssi - network.signal_strength
                    diff_vector.append(diff)
        sum_squares = sum([x**2 for x in diff_vector])
        location_estimate[location] = sum_squares

    # normalize the values so they sum to 1
    total = sum(location_estimate.values())
    for location in location_estimate:
        location_estimate[location] /= total

    return location_estimate

def plot_wifi_scan_pdf(wifi_scan, dataset_filename):
    """
    Plot the probability that the wifi_scan was taken at each location present in the dataset
    args:
        wifi_scan: WifiScan object, containing a list of WifiNetwork objects (in wifi_scan.networks)
        dataset_filename: string, the name of the dataset file
    """
    location_estimate = location_estimate_avg(wifi_scan.networks, dataset_filename)
    # location_estimate = location_estimate_naive_bayes(wifi_scan.networks, dataset_filename)
    x, y = zip(*location_estimate.keys())
    probabilities = np.fromiter(location_estimate.values(), dtype=float)#location_estimate.values()

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    zi = griddata((x, y), probabilities, (xi[None, :], yi[:, None]), method='cubic') #(100,100)

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, zi, levels=15, cmap='GnBu')
    plt.colorbar(label='Probability')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Probability of Location given Wifi Scan')

    # return the most likely location
    ind = np.unravel_index(np.argmax(zi, axis=None), zi.shape)
    most_likely_location = (xi[ind[1]], yi[ind[0]])
    print(f'Most likely location: {most_likely_location}')
    return most_likely_location
    
def location_estimate_pdf(dataset_filename,method='MSE'):
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0] # the Wi-Fi interface which we use to perform Wi-Fi operations (e.g. scan, connect, disconnect, ..
    scan = wdc.collect_wifi_scan(iface)

    if method == 'MSE':
        plot_wifi_scan_pdf(scan, dataset_filename)


if __name__ == "__main__":
    # Plot the probability that the wifi_scan was taken at each location present in the dataset
    location_estimate_pdf(DATASET_FILENAME_3,method='MSE')
    
    # data_points_3 = read_dataset_file(DATASET_FILENAME_3)
    # data_points_4 = read_dataset_file(DATASET_FILENAME_4)

    #plt.plot(data_points_3["cc:88:c7:41:b1:22:"][(0,-1)])
    
    # Find unique MAC Addresses in different instances of dataset collection.
    #print(len([k for k in data_points_3 if k in data_points_4]))
    #print(len(data_points_3))
    #print(len(data_points_4))
    # print(data_points)

    # # Make n plots of wifi visualizations
    # n_plots = 5
    # i=0
    # for key in data_points_3.keys():
    #     i+=1
    #     # plt.plot(data_points_3[key][(3,0)],label = key)
    #     avg_rssi_heatmap(data_points_3, key)
    #     if i>n_plots:
    #         break
    # plt.legend()

    # Generating RSSI heatmaps for the particular MAC Address
    #avg_rssi_heatmap(data_points, "cc:88:c7:41:b1:22:")
    #avg_rssi_heatmap(data_points, "cc:88:c7:42:9f:73:") 
    plt.show()