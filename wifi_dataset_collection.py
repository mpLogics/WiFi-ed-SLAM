import pywifi #https://github.com/awkman/pywifi/blob/master/DOC.md
# from pywifi import const
import time
"""
Collect a dataset of Wi-Fi signal strength in a room
Procedure:
    1. Move to pre-measured global position
        Enter the X coordinate of the scan
        Enter the Y coordinate of the scan
    2. Once you enter the Y coordinate, the program will perform a Wi-Fi scan
    3. Repeat several times at each location
Authors: Arthur L.
"""

DATASET_FILENAME = r'.\data\dataset_4_5.txt'
N_DATAPOINTS_PER_LOCATION = 5

class WifiNetwork:
    """
    A WifiNetwork object represents a single Wi-Fi network. It contains:
        BSSID: MAC address of the access point (unique hardware identifier) (eg. "00:1A:2B:3C:4D:5E")
        SSID: name of the network (eg. "MWireless")
        signal strength: signal strength (RSSI) of the access point
    """
    def __init__(self, bssid, ssid, signal_strength):
        self.bssid = bssid
        self.ssid = ssid
        self.signal = signal_strength

    def __str__(self):
        return f"{self.bssid}   {self.signal}   {self.ssid}"
    
    def __eq__(self, other):
        return self.bssid == other.bssid and self.ssid == other.ssid and self.signal == other.signal
    


class WifiScan:
    """
    A WifiScan object represents a single Wi-Fi scan of all networks. It contains: 
        location: tuple (x,y) [meters]    
        timestamp: float representing time [seconds] since the epoch (output of time.time())
        networks: list of WifiNetwork objects
    """
    def __init__(self, timestamp, location):
        self.networks = []
        self.timestamp = timestamp
        self.location = location

    def add_networks(self, scan_results):
        for network in scan_results:
            self.networks.append(WifiNetwork(network.bssid, network.ssid, network.signal))

    # Append dataset string to a dataset text file
    def save_to_file(self, filename):
        with open(filename, 'a') as fp:
            fp.write(str(self))
        print(f'Datapoint appended to file: {filename}')

    def __str__(self):
        b = time.localtime(self.timestamp)
        time_str = f"{b.tm_year:04d}:{b.tm_mon:02d}:{b.tm_mday:02d}:{b.tm_hour:02d}:{b.tm_min:02d}:{b.tm_sec:02d}"
        # TODO: FIgure out good way to format this string
        output = f"{self.location}, {time_str}\n"
        for network in self.networks:
            output += f"{str(network)}\n"
        return(f"{output}\n")
    
    # def __sub__(self, other):


# Create a heatmap of the Wi-Fi signal strength in a room
# The heatmap is multi-dimensional; each dimension represents a different network
def map_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0] # the Wi-Fi interface which we use to perform Wi-Fi operations (e.g. scan, connect, disconnect, ..

    while True:
        collect_wifi_scan(iface, n_scans=N_DATAPOINTS_PER_LOCATION)


def collect_wifi_scan(iface, n_scans=5, wait_seconds=4):
    # Input the location of the scan
    x = float(input("Enter X coordinate: "))
    y = float(input("Enter Y coordinate: "))
    location = (x, y)
    
    for i in range(n_scans):
        # Create a WifiScan object
        datapoint = WifiScan(time.time(), location)

        # Perform scan as soon as Y coordinate is entered
        iface.scan() #Trigger the interface to scan APs.
        time.sleep(wait_seconds) #scan time for each Wi-Fi interface is variant. Safer to wait 2 ~ 8 sec
        networks = iface.scan_results()

        # Add the scan results to the WifiScan object
        datapoint.add_networks(networks)

        # Save the WifiScan object to a file
        datapoint.save_to_file(DATASET_FILENAME)


def collect_slam_data():
    pass

def test_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(4)
    networks = iface.scan_results()
    # pywifi.profile.Profile object attributes:
    # 'akm', 'auth', 'bssid', 'cipher', 'freq', 'id', 'key', 'process_akm', 'signal', 'ssid'
    for profile in networks:
        print(f"{profile.bssid}   {profile.signal}   {profile.ssid}")


if __name__ == "__main__":
    # test_wifi()
    map_wifi()
