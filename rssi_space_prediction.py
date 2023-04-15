from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader
from preprocessing import FilteredData
from sortedcollections import OrderedSet

import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions



class Classifier(nn.Module):

    def __init__(self,hidden_dims=512,in_features=343,num_classes=24):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features=self.in_features,out_features=self.hidden_dims)
        self.fc2 = nn.Linear(in_features=self.hidden_dims,out_features=self.hidden_dims)
        self.fc3 = nn.Linear(in_features=self.hidden_dims,out_features=num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3,
            self.softmax)
    
    def forward(self,x):
        out = self.model(x)
        return out


filenames = [r'.\data\dataset_3_4.txt', r'.\data\dataset_4_4.txt',r'.\data\dataset_5_4.txt']


fd = FilteredData(filenames=filenames)
fd.filter_data_padding()

consistent_mac_address = OrderedSet(fd.get_consistent_mac_addresses())
unique_mac_addresses = OrderedSet(fd.get_unique_mac_addresses())
data_by_location = fd.data_by_location
data_by_bssid = fd.data_by_bssid
coordinates = fd.coordinates
NUM_POINTS_IN_SPACE = len(coordinates)

i=0
max_reading_len = len(data_by_location[(0,0)][list(data_by_bssid.keys())[0]])
coord_encodings = torch.Tensor([i for i in range(NUM_POINTS_IN_SPACE)])

# Mapping One Hot Encodings to Coordinate Space
i=0
coords_to_encodings={}
encodings_to_coords={}
for coord in coordinates:
    coords_to_encodings[coord] = torch.Tensor([coord_encodings[i]])
    encodings_to_coords[coord_encodings[i]] = coord
    i+=1

total_mac_address = len(consistent_mac_address)

train_data = torch.empty(0,total_mac_address + 1)
# x = torch.empty(0,total_mac_address)
# y = torch.empty((0,NUM_POINTS_IN_SPACE))
for coord in coordinates:
    x_temp = torch.empty((0,max_reading_len))
    for mac_address in consistent_mac_address:
        reading = torch.Tensor(data_by_location[coord][mac_address]).reshape(1,-1)
        x_temp = torch.cat((x_temp,reading),dim=0)
    
    for i in range(max_reading_len):
        x_temp_reshaped = torch.cat((x_temp[:,i],coords_to_encodings[coord]))
        x_temp_reshaped = torch.unsqueeze(x_temp_reshaped,0)
        train_data = torch.cat((train_data,x_temp_reshaped),dim=0)

train_data[:,:total_mac_address] = nn.functional.normalize(train_data[:,:total_mac_address],dim=1)

def train_step(model,train_loader,optimizer,loss_class):
    train_loss = 0.

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x = data[:,:343]
        labels = data[:,343:].squeeze(1).long()
        predicted_coords = model(x)
        loss = loss_class(input=predicted_coords,target=labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def train_classifier_model(model,train_loader,lr=0.1,num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch_i in pbar:
      train_loss_i = None
      optimizer.zero_grad()
      train_loss_i = train_step(model,train_loader,optimizer,loss_class=loss_fn)
      pbar.set_description(f'Train Loss: {train_loss_i:.4f} ')
      train_losses.append(train_loss_i)
    return train_losses


def fit_distribution(train_data):
    '''
    Function for fitting different distributions on data
    '''
    sampled_data = train_data[:32,:343].detach().numpy()
    f = Fitter(sampled_data,
            distributions=['gamma',
                            'lognorm',
                            "beta",
                            "burr",
                            "norm"])
    f.fit()
    print(f.summary())
    print(f.get_best(method = 'sumsquare_error'))

# Dataloader for data for classification
train_classification_loader = DataLoader(train_data,batch_size=24,shuffle=False)

# Training the model for classification
model = Classifier()
train_losses =train_classifier_model(
    model = model,
    train_loader=train_classification_loader)