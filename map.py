import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import wifi_dataset_collection as wdc
import pywifi
from preprocessing import FilteredData

"""
Create a heatmap of the Wi-Fi signal strength in a room from a dataset file
The heatmap is multi-dimensional; each dimension represents a different Access Point (AP)
Authors: Arthur L.
"""

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
    def __init__(self):
        self.prior_distribution = {}
    
    def craete_prior_distribution(self, data_points):
        """
        args:
            data_points: dictionary whose keys are locations and whose and values are another dictionary
                that has the bssid as the key and the list of signal strengths as the value
        Fits a log-normal distribution to the recieved signal strength values at each location, for each bssid
        The final prior distribution is a dictionary whose keys are locations in space, and whose values are
        a parametrization of an N-dimensional log-normal distribution that is fit to the data_points, where N is the number of bssids
        """
        for location, bssid_rssi_dict in data_points.items():
            pass

    def location_estimate(self, wifi_reading, datapoints_by_location):
        """
        args:
            wifi_reading: list of WifiNetwork objects, 
                where each one contains its bssid, ssid, and signal_strength
        returns:
            location_estimate: dictionary whose keys are the (x,y) tuples from data_points, 
                and whose values are the probability that that location is the true location 
                given the wifi_reading

        This function calculates the probability distribution of 
            P(this location is the true location | wifi reading)
        It does this by caluculating the probability of the wifi_reading given the stored RSSI log-normal distribution at each location
        Then, it normalizes the values
        """
        location_estimate = {}
        for location, bssid_rssi_dict in datapoints_by_location.items():
            pass
        
        # # Normalize the values
        # total = sum(location_estimate.values())
        # for location in location_estimate:
        #     location_estimate[location] /= total
        return location_estimate

def location_estimate_avg(wifi_reading, datapoints_by_location):
    """
    args:
        wifi_reading: list of WifiNetwork objects, 
            where each one contains its bssid, ssid, and signal_strength
        datapoints_by_location: dictionary {location: {bssid: [rssi values]} }
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
    data_points = datapoints_by_location
    location_estimate = {}
    for location, bssid_rssi_dict in data_points.items():
        diff_vector = [] # contains difference between the average and measured RSSI values for each bssid at this location
        for network in wifi_reading:
            if network.bssid in bssid_rssi_dict:
                l = len(bssid_rssi_dict[network.bssid])
                if l != 0:
                    avg_dataset_rssi = sum(bssid_rssi_dict[network.bssid])/l
                    diff = avg_dataset_rssi - network.signal_strength
                    diff_vector.append(diff)
        # square the distance to get positive values and 
        # take the inverse to weight closer points higher
        for i in range(len(diff_vector)):
            diff_vector[i] = diff_vector[i]**2 if diff_vector[i] > 0 else 0
        max_diff = max(diff_vector)
        for i in range(len(diff_vector)):
            if diff_vector[i] ==0:
                diff_vector[i] = max_diff
        inv_sum_squares = sum(diff_vector) #sum([x**-2 for x in diff_vector])
        location_estimate[location] = inv_sum_squares

    # normalize the values so they sum to 1
    total = sum(location_estimate.values())
    for location in location_estimate:
        location_estimate[location] /= total

    return location_estimate

def plot_wifi_scan_pdf(wifi_scan, datapoints_by_location):
    """
    Plot the probability that the wifi_scan was taken at each location present in the dataset
    args:
        wifi_scan: WifiScan object, containing a list of WifiNetwork objects (in wifi_scan.networks)
        dataset_filename: string, the name of the dataset file
    """
    location_estimate = location_estimate_avg(wifi_scan.networks, datapoints_by_location)
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
    
def location_estimate_pdf(datapoints_by_location,method='MSE'):
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0] # the Wi-Fi interface which we use to perform Wi-Fi operations (e.g. scan, connect, disconnect, ..
    scan = wdc.collect_wifi_scan(iface)

    if method == 'MSE':
        plot_wifi_scan_pdf(scan, datapoints_by_location)
    elif method == 'NB':
        pass


if __name__ == "__main__":
    dataset_files = [
        './data/dataset_3_4.txt',
        './data/dataset_4_4.txt',
        './data/dataset_5_4.txt'
    ]
    fd = FilteredData(dataset_files)
    fd.filter_data()

    # Plot the probability that the wifi_scan was taken at each location present in the dataset
    location_estimate_pdf(fd.data_by_location,method='MSE')


    # Make n plots of wifi visualizations
    n_plots = 5
    i=0
    for key in fd.data_by_bssid.keys():
        i+=1
        # plt.plot(fd.data_by_bssid[key][(3,0)],label = key)
        avg_rssi_heatmap(fd.data_by_bssid, key)
        if i>n_plots:
            break

    plt.show()