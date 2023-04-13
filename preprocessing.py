
NUM_READINGS_PER_BSSID=5

class FilteredData():
    '''
    Class for filtering the data files, extracting temporal and spatial data
    for modelling WiFi readings
    '''
    def __init__(self,filenames):
        self.consistent_mac_addresses = set()
        self.unique_mac_addresses = set()
        self.data_by_location = {} # key is the location and value is another dictionary
                                   # that has the bssid as the key and the signal strengths list as the value
        self.data_by_bssid = {} # key is the bssid and value is another dictionary
                                # that has the x,y coordinates as the key and the signal strengths list as the value
        self.coordinates = []
        self.filenames = filenames

    def make_mac_sets(self,filename):
        '''
        Update the set of unique mac addresses and the set of consistent mac addresses
        @args: filename: The complete path of a single dataset (.txt) file
        for generating unique and constant mac addresses throughout the 
        entire data.
        '''
        datapoints = self.read_dataset_file(filename, method='bssid') 
        mac_addresses = datapoints.keys() # all bssids in the file
        set_mac_addresses = set(mac_addresses)

        # take union to get unique_mac_addresses and intersection to get consistent_mac_addresses
        self.unique_mac_addresses = self.unique_mac_addresses.union(set_mac_addresses)
        if self.consistent_mac_addresses == set():#len(self.consistent_mac_addresses)==0:
            self.consistent_mac_addresses = set_mac_addresses
        else:
            self.consistent_mac_addresses.intersection_update(set_mac_addresses)
        
        # Fill in the coordinates on which all the data is collected (make sure first datafile loaded has all coordinates)
        if not self.coordinates:
            self.coordinates = list(datapoints[list(mac_addresses)[0]].keys())

    
    def structure_space_time(self,filename):
        '''
        Given that we have created consistent_mac addresses across all datafiles, 
        create the dictionaries :
        '''
        datapoints = self.read_dataset_file(filename,method='bssid')    
        
        # go through consistent mac addresses and coordinates to fill in bssid_data and temporal_data
        for bssid in self.consistent_mac_addresses:
            for loc in self.coordinates:
                # fill in self.data_by_bssid
                if bssid in datapoints: # true by construction
                    if loc in datapoints[bssid]:
                        self.add_to_dict(self.data_by_bssid,bssid,loc,datapoints[bssid][loc])
                        self.add_to_dict(self.data_by_location,loc,bssid,datapoints[bssid][loc])
                
    
    def filter_data(self):
        '''
        Driver function that fills in all FilteredData member variables 
        from a list of dataset filenames 
        
        @args 
        filenames - list of path values for all files in the dataset
        '''
        # go throught all files once to get consistent mac addresses
        for file_ in self.filenames:
            self.make_mac_sets(filename=file_)
        # go through again to fill in data structures
        for file_ in self.filenames:
            self.structure_space_time(filename=file_)
    
    def get_consistent_mac_addresses(self):
        return self.consistent_mac_addresses
    
    def get_unique_mac_addresses(self):
        return self.unique_mac_addresses
    
    def read_dataset_file(self, filename, method='bssid'):
        # Read the dataset file
        with open(filename, 'r') as fp:
            lines = fp.readlines()

        datapoints = {}
        #dictionary whose key is the bssid and whose and value is another dictionary
        #that has the x,y coordinates as the key and the signal strength as the value
        
        # Parse the dataset file
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
                    signal_strength = [float(network[1])]
                    ssid = ' '.join(network[2:])

                    if method == 'bssid':
                        datapoints = self.add_to_dict(datapoints, bssid, (x,y), signal_strength)
 
                    elif method == 'location':
                        self.add_to_dict(datapoints, (x,y), bssid, signal_strength)

        return datapoints
    
    def add_to_dict(self, dict_to_update, key, value_dict, value):
        """"
        Helper function to add to a dictionary of dictionaries
        @args
        value = list (of RSSI values)
        key = bssid or location
        value_dict = location or bssid
        dict_to_update = {key: {value_dict: value}}
        """
        if key in dict_to_update:
            if value_dict in dict_to_update[key]:
                dict_to_update[key][value_dict] = dict_to_update[key][value_dict] + value
            else:
                dict_to_update[key][value_dict] = value
        else:
            dict_to_update[key] = {value_dict:value}

        return dict_to_update
        

if __name__== "__main__":
    dataset_files = [
        './data/dataset_3_4.txt',
        './data/dataset_4_4.txt',
        './data/dataset_5_4.txt'
    ]
    fd = FilteredData(dataset_files)
    fd.filter_data()
    print(len(fd.get_consistent_mac_addresses()))
    print(fd.data_by_bssid["cc:88:c7:41:b1:22:"][(0,-1)])

