from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader
from preprocessing import FilteredData

import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss,self).__init__()
    
    def forward(self,y_val,num_spatial_points):    
        y_val = y_val.repeat(1,num_spatial_points)
        mean_squared = torch.nn.MSELoss()
        loss = -mean_squared(y_val,y_val.t())
        return loss
    
class Classifier(nn.Module):
    def __init__(self,in_features=64,num_classes=24,hidden_dims=1):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_features = in_features
        self.fc = nn.Linear(in_features=self.in_features,out_features=num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(
            self.fc,
            self.softmax,)
    
    def forward(self,x):
        out = self.model(x)
        return out

class Encoder(nn.Module):

    def __init__(self,hidden_dims=512,in_features=343,num_classes=64):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features=self.in_features,out_features=200)
        self.fc2 = nn.Linear(in_features=200,out_features=100)
        self.fc3 = nn.Linear(in_features=100,out_features=50)
        self.fc4 = nn.Linear(in_features=50,out_features=100)
        self.fc5 = nn.Linear(in_features=100,out_features=150)
        self.fc6 = nn.Linear(in_features=150,out_features=200)
        self.fc_final = nn.Linear(in_features=200,out_features=48)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3,
            self.relu,
            self.fc4,
            self.relu,
            self.fc5,
            self.relu,
            self.fc6,
            self.relu,
            self.fc_final,)
            #self.softmax)
    
    def forward(self,x):
        out = self.model(x)
        return out


filenames = [r'.\data\dataset_3_4.txt', r'.\data\dataset_4_4.txt',r'.\data\dataset_5_4.txt']


fd = FilteredData(filenames=filenames)
fd.filter_data_padding()
consistent_mac_address = list(fd.get_consistent_mac_addresses())

consistent_mac_address = sorted(list(fd.get_consistent_mac_addresses()))
unique_mac_addresses = sorted(list(fd.get_unique_mac_addresses()))
data_by_location = fd.data_by_location
data_by_bssid = fd.data_by_bssid
coordinates = fd.coordinates
NUM_POINTS_IN_SPACE = len(coordinates)

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

#train_data[:,:total_mac_address] = nn.functional.normalize(train_data[:,:total_mac_address],dim=1)

def get_label_collection(batched_labels):
    label_args = {}
    
    for i in torch.unique(batched_labels):
        label_args[i] = torch.where(batched_labels==i)
    return label_args
                
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss,self).__init__()
    
    def forward(self,batched_labels,batched_predicted_coords):
        loss = torch.tensor([0.0], requires_grad=True)
        m = 500
        tot_samples = batched_predicted_coords.shape[0]
        for i in range(tot_samples):
            labels = batched_labels[i]-batched_labels
            labels = torch.where(labels==0,0,1)
            dist = torch.linalg.norm(batched_predicted_coords[i].reshape(1,-1) - batched_predicted_coords,dim=1)**2
            loss = (1-labels)*(dist) + labels*torch.max(torch.zeros_like(dist),m - dist)
            loss = loss.sum()

        return loss
    
def batched_training(model,train_loader,optimizer,loss_class,batch_size):
    train_loss = 0.
    batched_predicted_coords = torch.empty((0,48))
    batched_labels = torch.empty(48)
    
    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        x = data[:,:343]
        labels = data[:,343:].long().reshape(-1)
        predicted_coords = model(x)
        batched_predicted_coords = torch.cat((batched_predicted_coords,predicted_coords),dim=0)
        batched_labels = torch.cat((batched_labels,labels))
        loss = loss_class(labels,predicted_coords)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()    
    return train_loss
    

def train_step(encoder,classifier,train_loader,optimizer,loss_class):
    train_loss = 0.

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x = data[:,:343]
        labels = data[:,343:].squeeze(1).long()
        latent_coords = encoder(x)
        predicted_coords = classifier(latent_coords)
        loss = loss_class(input=predicted_coords,target=labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss

def train_model(encoder,train_loader,lr=0.0001,num_epochs=1000,batch_size=24,training='encoder',classifier=None):
    if training == 'classifier':
        # Freezing the encoder layers
        trainable_params = classifier.parameters
    else:
        trainable_params = encoder.parameters
    
    optimizer = torch.optim.Adam(trainable_params(), lr=lr)    
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    #
    if training=='encoder':
        loss_fn = CustomLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    for epoch_i in pbar:
      train_loss_i = None
      optimizer.zero_grad()
      
      if training=='encoder':
        train_loss_i = batched_training(encoder,train_loader,optimizer,loss_fn,batch_size)
      else:
        train_loss_i = train_step(encoder,classifier,train_loader,optimizer,loss_fn)  
        
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


#'''
if __name__=='__main__':
    train_classification_loader = DataLoader(train_data,batch_size=24*8,shuffle=True)
    encoding_model = Encoder()
    classifying_model = Classifier()
    out = encoding_model(train_data[:,:343]).detach().numpy()
    
    #fig,ax = plt.subplots(nrows=6,ncols=4)
    #plt.tight_layout()
    j = 0
    plt.figure()
    plt.imshow(out[(train_data[:,343:]==1).reshape(-1)])
    #plt.show()
    '''
    for row in range(6):
        for col in range(4):
            ax[row][col].set_title(j)
            ax[row][col].axis('off')
            ax[row][col].imshow(out[(train_data[:,343:]==j).reshape(-1)])

            j+=1
    plt.show()
    '''
    train_losses_encoder = train_model(
        encoder = encoding_model,
        num_epochs=1500,
        train_loader=train_classification_loader,training='encoder')
    
    plt.figure()
    plt.imshow(out[(train_data[:,343:]==1).reshape(-1)])
    plt.show()
    # Training the model for classification

    #train_losses_classifier = train_model(
    #    encoder = encoding_model,
    #    classifier = classifying_model,
    #    lr = 0.01,
    #    num_epochs=3000,
    #    train_loader=train_classification_loader,training='classifier')

    '''
    out = encoding_model(train_data[:,:343]).detach().numpy()
    fig,ax = plt.subplots(nrows=6,ncols=4)
    plt.tight_layout(pad=1.2)
    j = 0
    for row in range(6):
        for col in range(4):
            ax[row][col].set_title(j)
            ax[row][col].imshow(out[(train_data[:,343:]==j).reshape(-1)])
            j+=1
    plt.show()
    '''
        
    #plt.figure()
    #plt.imshow()
        
    
    #plt.figure()
    #plt.imshow(out.T)

    '''

    plt.figure()
    plt.plot(train_losses_encoder)
    plt.title('Contrastive Loss for the Encoder')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss value')
    
    plt.figure()
    plt.plot(train_losses_classifier)
    plt.title('Cross Entropy Loss for classification')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss Value')
    '''
    
    

    torch.save(encoding_model.state_dict(), r'encoder.pth')
    torch.save(classifying_model.state_dict(), r'classifier.pth')

