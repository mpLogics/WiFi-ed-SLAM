import numpy as np
from map import read_dataset_file
import torch
NUM_READINGS_PER_BSSID=10


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
    '''
    Class for filtering the data files, extracting temporal and spatial data
    for modelling WiFi readings
    '''
    def __init__(self):
        self.consistent_mac_addresses = set()
        self.unique_mac_addresses = set()
        self.temporal_data = {}
        self.bssid_data = {}
        self.iter_= 0
        self.coordinates = None

    def make_mac_sets(self,filename):
        '''
        @args: filename: The complete path of a single dataset (.txt) file
        for generating unique and constant mac addresses throughout the 
        entire data.
        '''
        dataset_points = read_dataset_file(filename)
        mac_addresses = dataset_points.keys()
        set_mac_addresses = set(mac_addresses)
        self.unique_mac_addresses = self.unique_mac_addresses.union(set_mac_addresses)
        if self.coordinates is None:
            self.coordinates = list(dataset_points[list(mac_addresses)[0]].keys())
        if len(self.consistent_mac_addresses)==0:
            self.consistent_mac_addresses = set_mac_addresses
        else:
            self.consistent_mac_addresses.intersection_update(set_mac_addresses)
        
    
    def structure_space_time(self,filename):
        '''
        Given that we have created consistent_mac addresses across all datafiles, create the temporal and spatial dictionaries:
        self.temporal_data is a dictionary whose key is the bssid and whose and value is another dictionary
        that has the x,y coordinates as the key and the signal strength as the value

        self.spatial_data is a dictionary whose key is the location and whose and value is another dictionary
        that has the bssid as the key and the signal strength as the value
        '''
        dataset_points = read_dataset_file(filename,method='bssid')    
        
        for bssid in self.consistent_mac_addresses:
            for j in self.coordinates:
                
                try:
                    self.bssid_data[bssid]
                except KeyError:
                    self.bssid_data[bssid] = {}
                try:
                    self.bssid_data[bssid][j]
                except KeyError:
                    self.bssid_data[bssid][j]=[]
                try:
                    dataset_points[bssid][j]
                except KeyError:
                    dataset_points[bssid][j]= [-200]# for i in range(NUM_READINGS_PER_BSSID)]
                if type(dataset_points[bssid][j])!=list:
                    dataset_points[bssid][j]=[dataset_points[bssid][j]]
                self.bssid_data[bssid][j].append(dataset_points[bssid][j])
            
            print(self.bssid_data[bssid][j],j,bssid)
            self.bssid_data[bssid][j] = [item for sublist in self.bssid_data[bssid][j] for item in sublist] #convert nested lists to normal list

        for j in self.coordinates:
            for bssid in self.consistent_mac_addresses:
                try:
                    self.temporal_data[j]
                except KeyError:
                    self.temporal_data[j] = {}
                try:
                    self.temporal_data[j][bssid]
                except KeyError:
                    self.temporal_data[j][bssid]=[]
                
                try:
                    dataset_points[bssid][j]
                    self.temporal_data[j][bssid].append(dataset_points[bssid][j])
                except KeyError:
                    self.temporal_data[j][bssid].append([-200])# for i in range(NUM_READINGS_PER_BSSID)]
            
            try:
                self.temporal_data[j][bssid] = [item for sublist in self.temporal_data[j][bssid] for item in sublist] #convert nested lists to normal list    
            except:
                print(self.temporal_data[j][bssid])
                print(dataset_points[bssid])
                

    
    def structure_space_time2(self,dataset_points,mac_addresses):
        '''
        Given that we have created consistent_mac addresses across all datafiles, create the temporal and spatial dictionaries:
            self.temporal_data is a dictionary whose key is the bssid and whose and value is another dictionary
                that has the x,y coordinates as the key and the signal strength as the value

            self.spatial_data is a dictionary whose key is the location and whose and value is another dictionary
                that has the bssid as the key and the signal strength as the value

        Generate temporal and spatial key-value pairs dictionary for
        querying based on model type
        @args 
        dataset_points - rssi values for file in loop
        mac_addresses - mac addresses for the file in loop

        '''
        coordinates = dataset_points[mac_addresses[0]].keys()

        for i in self.consistent_mac_addresses:
            for j in coordinates:
                try:
                    self.temporal_data[j]
                except KeyError:
                    self.temporal_data[j] = []
                self.temporal_data[j].append((i,dataset_points[i][j]))
            self.spatial_data[i] = dataset_points[i]
    
    
    def filter_data(self,filenames):
        '''
        Driver function for making sets of mac-addresses and 
        unique mac addresses and structuring data with time-valued
        and space-valued keys
        @args 
        filenames - list of path values for all files in the 
        dataset
        '''
        for file_ in filenames:
            self.make_mac_sets(filename=file_)
        for file_ in filenames:
            self.structure_space_time(filename=file_)
    
    def get_consistent_mac_addresses(self):
        return self.consistent_mac_addresses
    
    def get_unique_mac_addresses(self):
        return self.unique_mac_addresses