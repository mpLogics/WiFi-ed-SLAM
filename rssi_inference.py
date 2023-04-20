from rssi_space_prediction import Encoder,Classifier
from wifi_dataset_collection import WifiNetwork,WifiScan,test_wifi
import pywifi
import time
import torch
from preprocessing import FilteredData, Data
from map import plot_wifi_scan_pdf

filenames = [r'.\data\dataset_3_4.txt', r'.\data\dataset_4_4.txt',r'.\data\dataset_5_4.txt']

def get_wifi_scan(consistent_mac_address,d):
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(4)
    networks = iface.scan_results()
    # pywifi.profile.Profile object attributes:
    # 'akm', 'auth', 'bssid', 'cipher', 'freq', 'id', 'key', 'process_akm', 'signal', 'ssid'
    input_signal = torch.zeros((1,len(consistent_mac_address)))
    
    return d.prep_infer_data(
        networks=networks,
        input_signal=input_signal)

def make_data(data_by_location,coordinates,mode='train'):

    NUM_POINTS_IN_SPACE = len(coordinates)
    max_reading_len = len(data_by_location[(0,0)][list(data_by_bssid.keys())[0]])
    coord_encodings = torch.Tensor([i for i in range(NUM_POINTS_IN_SPACE)])
    # Mapping One Hot Encodings to Coordinate Space
    i=0
    coords_to_encodings={}
    encodings_to_coords={}
    
    for coord in coordinates:
        coords_to_encodings[coord] = torch.Tensor([coord_encodings[i]])
        encodings_to_coords[(int)(coord_encodings[i])] = coord
        i+=1

    total_mac_address = len(consistent_mac_address)

    if mode=='train':
        train_data = torch.empty(0,total_mac_address + 1)
        for coord in coordinates:
            x_temp = torch.empty((0,max_reading_len))
            for mac_address in consistent_mac_address:
                reading = torch.Tensor(data_by_location[coord][mac_address]).reshape(1,-1)
                x_temp = torch.cat((x_temp,reading),dim=0)
            
            for i in range(max_reading_len):
                x_temp_reshaped = torch.cat((x_temp[:,i],coords_to_encodings[coord]))
                x_temp_reshaped = torch.unsqueeze(x_temp_reshaped,0)
                train_data = torch.cat((train_data,x_temp_reshaped),dim=0)
        return train_data, coords_to_encodings, encodings_to_coords
    return coords_to_encodings, encodings_to_coords



if __name__=='__main__':
    fd = FilteredData(filenames=filenames)
    fd.filter_data_padding()
    consistent_mac_address = sorted(list(fd.get_consistent_mac_addresses()))
    unique_mac_addresses = sorted(list(fd.get_unique_mac_addresses()))

    data_by_location = fd.data_by_location
    data_by_bssid = fd.data_by_bssid
    coordinates = fd.coordinates
    d = Data(consistent_mac_address,data_by_bssid,data_by_location)
    coords_to_encodings, encodings_to_coords = d.make_data(coordinates,mode='infer')
    data = get_wifi_scan(consistent_mac_address,d)
    encoder = Encoder()
    classifier = Classifier()
    print(torch.load(r'encoder.pth').keys())
    encoder.load_state_dict(torch.load(r'encoder.pth'))
    classifier.load_state_dict(torch.load(r'classifier.pth'))
    encoder.eval()
    classifier.eval()
    encoded_data = encoder(data)
    prob_dist = classifier(encoded_data)
    prob_estimate={}
    for i in range(24):
        prob_estimate[encodings_to_coords[i]] = prob_dist[0,i].detach().numpy()
    
    plot_wifi_scan_pdf(None,location_estimate = prob_estimate)
    
    #pred_val = torch.argmax(prob_dist).long()
    #print(encodings_to_coords[(int)(pred_val)])
