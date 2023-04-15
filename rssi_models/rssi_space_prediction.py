from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from map import read_dataset_file
from torch.utils.data import DataLoader

NUM_POINTS_IN_SPACE = 24
NUM_POINTS_IN_TIME = 10



filenames = [r'.\data\dataset_3_4.txt', r'.\data\dataset_4_4.txt',r'.\data\dataset_5_4.txt']
dataset_points=[]
macs_coords = []

for file in filenames:
    dataset_points.append(read_dataset_file(file))

from preprocessing import FilterData
fd = FilterData()
fd.filter_data(filenames)
consistent_mac_address = fd.get_consistent_mac_addresses()
bssid_data = fd.bssid_data


'''
Get all consistent Bssids. Use this Bssid's to get
a 
data_train <- Batch_size_train x len(consistent_mac_addresses) x 1
data_test  <- Batch_size_test x len(consistent_mac_addresses) x 1
inference  <- len(consistent_mac_addresses) x 1

Fill value -200 for all empty readings in the train, test or inference sets

'''

#unique_mac_address = fd.get_unique_mac_addresses()
#print("Consistent Mac Addresses",consistent_mac_address)


macs = list(dataset_points[0].keys())
macs_coords = dataset_points[0][macs[0]].keys()

train_data = torch.empty((NUM_POINTS_IN_SPACE,0))

for data in dataset_points:
    train_coords = torch.empty((0,NUM_POINTS_IN_TIME))
    appending_coords = False
    for coord in macs_coords:
        train_coord = torch.Tensor(data[macs[0]][coord]).unsqueeze(0)
        train_coords = torch.cat((train_coords,train_coord),dim=0)
    train_data = torch.cat((train_data,train_coords),dim=1)