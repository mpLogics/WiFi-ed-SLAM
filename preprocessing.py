import pickle
import torch

class Data():
    def __init__(self,consistent_mac_address,data_by_bssid,data_by_location):
        self.consistent_mac_address = consistent_mac_address
        self.data_by_bssid = data_by_bssid
        self.data_by_location = data_by_location

    def prep_infer_data(self,networks,input_signal):
        # pywifi.profile.Profile object attributes:
        # 'akm', 'auth', 'bssid', 'cipher', 'freq', 'id', 'key', 'process_akm', 'signal', 'ssid'

        for profile in networks:
            if profile.bssid in self.consistent_mac_address:
                input_signal[0,self.consistent_mac_address.index(profile.bssid)] = profile.signal

        # Padding with -200 values 
        for i in range(len(self.consistent_mac_address)):
            if input_signal[0,i]==0:
                input_signal[0,i] = -200
        return input_signal
    
    def make_data(self,coordinates,mode='train'):

        NUM_POINTS_IN_SPACE = len(coordinates)
        max_reading_len = len(self.data_by_location[(0,0)][list(self.data_by_bssid.keys())[0]])
        coord_encodings = torch.Tensor([i for i in range(NUM_POINTS_IN_SPACE)])
        # Mapping Integer Encodings to Coordinate Space
        i=0
        coords_to_encodings={}
        encodings_to_coords={}
        
        for coord in coordinates:
            coords_to_encodings[coord] = torch.Tensor([coord_encodings[i]])
            encodings_to_coords[(int)(coord_encodings[i])] = coord
            i+=1

        total_mac_address = len(self.consistent_mac_address)

        if mode=='train':
            train_data = torch.empty(0,total_mac_address + 1)
            for coord in coordinates:
                x_temp = torch.empty((0,max_reading_len))
                for mac_address in self.consistent_mac_address:
                    reading = torch.Tensor(self.data_by_location[coord][mac_address]).reshape(1,-1)
                    x_temp = torch.cat((x_temp,reading),dim=0)
                
                for i in range(max_reading_len):
                    x_temp_reshaped = torch.cat((x_temp[:,i],coords_to_encodings[coord]))
                    x_temp_reshaped = torch.unsqueeze(x_temp_reshaped,0)
                    train_data = torch.cat((train_data,x_temp_reshaped),dim=0)
            return train_data, coords_to_encodings, encodings_to_coords
        return coords_to_encodings, encodings_to_coords

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
        self.n_bssids = 0 
        self.coordinates = []
        self.filenames = filenames
        

    def save_data(self, filename):
        '''
        Save the data in a pickle file
        '''
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)

    def load_data(self, filename):
        '''
        Load the data from a pickle file
        '''
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
        return data
    
    def update_mac_sets(self,filename):
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

    def update_data_dicts_with_padding(self,filename):
        '''
        Given that we have created consistent_mac addresses across all datafiles, 
        update the dictionaries.
        @args: filename: The complete path of a single dataset (.txt) file
        '''
        datapoints = self.read_dataset_file(filename,method='bssid')    
        
        # go through consistent mac addresses and coordinates to fill in bssid_data and temporal_data
        max_len_bssid = -1
        max_len_loc = -1
        for bssid in self.consistent_mac_addresses:
            for loc in self.coordinates:
                # fill in self.data_by_bssid
                if bssid in datapoints: # true by construction
                    if loc in datapoints[bssid]:
                        self.add_to_dict(self.data_by_bssid,bssid,loc,datapoints[bssid][loc])
                        self.add_to_dict(self.data_by_location,loc,bssid,datapoints[bssid][loc])
                    else:
                        self.add_to_dict(self.data_by_bssid,bssid,loc,[])
                        self.add_to_dict(self.data_by_location,loc,bssid,[])
                
                if max_len_bssid < len(self.data_by_bssid[bssid][loc]):
                    max_len_bssid = len(self.data_by_bssid[bssid][loc])
                
                if max_len_loc < len(self.data_by_location[loc][bssid]):
                    max_len_loc = len(self.data_by_location[loc][bssid])

        for bssid in self.consistent_mac_addresses:
            for loc in self.coordinates:
                for i in range(len(self.data_by_bssid[bssid][loc]),(max_len_bssid)):
                    self.data_by_bssid[bssid][loc].append(-200)
                for i in range(len(self.data_by_location[loc][bssid]),(max_len_loc)):
                    self.data_by_location[loc][bssid].append(-200)
        

    def update_data_dicts(self,filename):
        '''
        Given that we have created consistent_mac addresses across all datafiles, 
        update the dictionaries.
        @args: filename: The complete path of a single dataset (.txt) file
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
                
    def filter_data_padding(self):
        '''
        Driver function that fills in all FilteredData member variables 
        from a list of dataset filenames 
        
        @args 
        filenames - list of path values for all files in the dataset
        '''
        # go throught all files once to get consistent mac addresses
        for file_ in self.filenames:
            self.update_mac_sets(filename=file_)
        self.n_bssids = len(self.consistent_mac_addresses)

        # go through again to fill in data structures
        for file_ in self.filenames:
            self.update_data_dicts_with_padding(filename=file_)

    def filter_data(self):
        '''
        Driver function that fills in all FilteredData member variables 
        from a list of dataset filenames 
        
        @args 
        filenames - list of path values for all files in the dataset
        '''
        # go throught all files once to get consistent mac addresses
        for file_ in self.filenames:
            self.update_mac_sets(filename=file_)
        self.n_bssids = len(self.consistent_mac_addresses)

        # go through again to fill in data structures
        for file_ in self.filenames:
            self.update_data_dicts(filename=file_)
    
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
                        datapoints = self.add_to_dict(datapoints, (x,y), bssid, signal_strength)

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
    fd.save_data('./data/filtered_data.pkl')
    # print(len(fd.get_consistent_mac_addresses()))
    # print(fd.data_by_bssid["cc:88:c7:41:b1:22:"][(0,-1)])

