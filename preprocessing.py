import numpy as np
from map import read_dataset_file

class Data():
    def __init__(self):
        pass
    
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


class FilterData():
    def __init__(self):
        self.consistent_mac_addresses = set()
        self.unique_mac_addresses = set()
        self.temporal_data = {}
        self.spatial_data = {}

    def filter_data(self,filename):
        dataset_points = read_dataset_file(filename)
        mac_addresses = set(dataset_points.keys())
        self.unique_mac_addresses = self.unique_mac_addresses.union(mac_addresses)
        self.consistent_mac_addresses = self.consistent_mac_addresses.intersect(mac_addresses)
        self.structure_space_time(dataset_points=dataset_points,mac_addresses=mac_addresses)

    def structure_space_time(self,dataset_points,mac_addresses):
        coordinates = dataset_points[mac_addresses[0]].keys()

        for i in self.consistent_mac_addresses:
            for j in coordinates:
                try:
                    self.temporal_data[j]
                except KeyError:
                    self.temporal_data[j] = []
                self.temporal_data[j].append((i,dataset_points[i][j]))
            self.spatial_data = dataset_points[i]
    
    
    def make_mac_sets(self,filenames):
        for file_ in filenames:
            self.filter_data(filename=file_)