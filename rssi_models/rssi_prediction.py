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

class Classifier(nn.Module):

    def __init__(self,hidden_dims=100,in_features=100,num_classes=24):
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

class Encoder(nn.Module):
    
    def __init__(self,hidden_dims=100,in_features=10):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features=self.in_features,out_features=self.hidden_dims)
        self.fc2 = nn.Linear(in_features=self.hidden_dims,out_features=self.hidden_dims)
        self.fc3 = nn.Linear(in_features=self.hidden_dims,out_features=self.hidden_dims)
        self.fc4 = nn.Linear(in_features=self.hidden_dims,out_features=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.classifier = Classifier()
        self.model = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc4)
        
    
    def forward(self,x,x_lat):
        x = self.model(x)
        x = torch.cat((x,x_lat),dim=1)
        out = self.classifier(x)
        return out

def train_step(model,train_loader,optimizer,loss_class):
    train_loss = 0.

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        x = data['x']
        predicted_coords = model(x)
        target_coords = data['target']
        loss = loss_class(input=predicted_coords,target=target_coords)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def train_classifier_model(model,train_loader,lr=0.01,num_epochs=1000):
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

filenames = [r'.\data\dataset_3_4.txt', r'.\data\dataset_4_4.txt',r'.\data\dataset_5_4.txt']
dataset_points=[]
macs_coords = []

for file in filenames:
    dataset_points.append(read_dataset_file(file))

macs = list(dataset_points[0].keys())
macs_coords = dataset_points[0][macs[0]].keys()

train_data = torch.empty((NUM_POINTS_IN_SPACE,0))

# Transforming Data for aggregating rssi values in latent space
for data in dataset_points:
    train_coords = torch.empty((0,NUM_POINTS_IN_TIME))
    appending_coords = False
    for coord in macs_coords:
        train_coord = torch.Tensor(data[macs[0]][coord]).unsqueeze(0)
        train_coords = torch.cat((train_coords,train_coord),dim=0)
    train_data = torch.cat((train_data,train_coords),dim=1)

encodings = torch.Tensor([i for i in range(NUM_POINTS_IN_SPACE)])
encodings = nn.functional.one_hot(encodings.long())
train = torch.cat((train_data,encodings),dim=1)

model = Encoder(in_features=len(dataset_points)*NUM_POINTS_IN_TIME)

train_x = train[:,:NUM_POINTS_IN_TIME*len(dataset_points)]
train_target = train[:,NUM_POINTS_IN_TIME*len(dataset_points):]


data_dict={}
data_dict['x'] = train[:,:NUM_POINTS_IN_TIME*len(dataset_points)]
data_dict['target'] = train[:,NUM_POINTS_IN_TIME*len(dataset_points):]
train_classification_loader = DataLoader(data,batch_size=len(macs_coords),shuffle=False)

train_losses =train_classifier_model(
    model = model,
    train_loader=train_classification_loader)
