import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import stats
import wifi_dataset_collection as wdc
import pywifi
from preprocessing import FilteredData
import pickle
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


class NaiveBayesEstimator:
    def __init__(self):
        self.prior_distribution = {} # dictionary whose keys are locations and whose values are a parametrization of an N-dimensional log-normal distribution

    def fit_lognormal(self, datapoints_by_location=None):
        """
        args:
            data_points: dictionary whose keys are locations and whose and values are another dictionary
                that has the bssid as the key and the list of signal strengths as the value
        Fits a log-normal distribution to the recieved signal strength values at each location, for each bssid
        The final prior distribution is a dictionary whose keys are locations in space, and whose values are
        a parametrization of an N-dimensional log-normal distribution that is fit to the data_points, where N is the number of bssids
        """
        # self.means = np.zeros(self.n_locations, self.n_bssids)  
        # self.variances = np.zeros(self.n_locations, self.n_bssids, self.n_bssids)  

        if datapoints_by_location is None:
            self.load('./data/prior_distribution.pkl')
        else:
            for location, bssid_rssi_dict in datapoints_by_location.items():
                self.prior_distribution[location] = {}
                for bssid, rssi_list in bssid_rssi_dict.items():
                    # (mu, sigma) = stats.lognorm.fit(rssi_list)
                    if len(rssi_list) < 3:
                        mean, var = rssi_list[0], 0.00000000001
                    else:
                        _ , mean, var = stats.lognorm.fit(rssi_list)
                    print(f"loc: {location}, mean: {mean}, var: {var}")
                    # print(mean, var)
                    self.prior_distribution[location][bssid] = (mean, var) 
            
            self.save('./data/prior_distribution.pkl')

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.prior_distribution, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.prior_distribution = pickle.load(f)

    def location_estimate(self, wifi_reading):
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
        n_locations_processed = 0

        for location, bssid_fit_dict in self.prior_distribution.items():
            # get the probability that the wifi_reading is from this location
            log_odds = 0
            for network in wifi_reading:
                # print(f'Processing {network.bssid}')
                if network.bssid in bssid_fit_dict:
                    mu, sigma = bssid_fit_dict[network.bssid]
                    mu = abs(mu)
                    # z_score = (np.log(abs(network.bssid)) - mu) / (sigma * np.sqrt(2))
                    # # find log probability using logarithmic cumulative distribution function
                    # log_odds += stats.lognorm.logcdf(x, s, loc=loc, scale=scale)
                    l = stats.lognorm.logpdf(abs(network.signal_strength), mu, sigma)
                    if l != float('-inf') and l != float('-inf'): log_odds -= l 
                    # print(f'Processing {network.bssid}, mu: {mu}, sigma: {sigma}, log_odds={log_odds}')
                
            location_estimate[location] = log_odds
            n_locations_processed += 1
        # print(f'Processed {n_locations_processed} locations')

        # Normalize the values
        total = sum(location_estimate.values())
        for location in location_estimate:
            location_estimate[location] /= total

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

def plot_wifi_scan_pdf(wifi_scan, location_estimate):
    """
    Plot the probability that the wifi_scan was taken at each location present in the dataset
    args:
        wifi_scan: WifiScan object, containing a list of WifiNetwork objects (in wifi_scan.networks)
        dataset_filename: string, the name of the dataset file
    """
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
    plt.show()
    # return the most likely location
    ind = np.unravel_index(np.argmax(zi, axis=None), zi.shape)
    most_likely_location = (xi[ind[1]], yi[ind[0]])
    print(f'Most likely location: {most_likely_location}')
    return most_likely_location
    
def location_estimate_pdf(datapoints_by_location,method='MSE', plot=True):
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0] # the Wi-Fi interface which we use to perform Wi-Fi operations (e.g. scan, connect, disconnect, ..
    scan = wdc.collect_wifi_scan(iface)

    location_estimate = {}
    if method == 'MSE':
        location_estimate = location_estimate_avg(scan.networks, datapoints_by_location)
    elif method == 'NB':
        nb = NaiveBayesEstimator()
        try:
            nb.fit_lognormal() # if you've already fit the data, you can just load it from the pickle file
        except:
            nb.fit_lognormal(datapoints_by_location)

        location_estimate = nb.location_estimate(scan.networks)

    if plot: plot_wifi_scan_pdf(scan, location_estimate)

    return location_estimate





if __name__ == "__main__":
    dataset_files = [
        './data/dataset_3_4.txt',
        './data/dataset_4_4.txt',
        './data/dataset_5_4.txt'
    ]
    fd = FilteredData(dataset_files)
    fd.filter_data() # if you haven't saved the filtered data before
    # fd.load_data('./data/filtered_data.pkl')

    # Plot the probability that the wifi_scan was taken at each location present in the dataset
    # location_estimate_pdf(fd.data_by_location,method='MSE')
    location_estimate_pdf(fd.data_by_location,method='NB')

    # # Make n plots of wifi visualizations
    # n_plots = 5
    # i=0
    # for key in fd.data_by_bssid.keys():
    #     i+=1
    #     # plt.plot(fd.data_by_bssid[key][(3,0)],label = key)
    #     avg_rssi_heatmap(fd.data_by_bssid, key)
    #     if i>n_plots:
    #         break

    plt.show()