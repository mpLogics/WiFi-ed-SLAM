from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.interpolate import griddata
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from map import read_dataset_file
from gpytorch.kernels import RBFKernel, CosineKernel, LinearKernel, PolynomialKernel, MaternKernel, ScaleKernel
from torch.utils.data import DataLoader

NUM_POINTS_IN_SPACE = 24
NUM_POINTS_IN_TIME = 10

class EncoderEnsemble(nn.Module):
    def __init__(self,num_ensembles):
        super().__init__()
        self.models = nn.ModuleList([Encoder() for _ in range(num_ensembles)])
    
    def forward(self,x):
        full_encoding=-1
        for model in self.models:
            batch_encoding = model(x)
            if full_encoding==-1:
                full_encoding = batch_encoding
            else:
                full_encoding = torch.cat((full_encoding,batch_encoding),dim=1)
        return full_encoding

class Encoder(nn.Module):
    
    def __init__(self,hidden_dims=100,in_features=10):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features=self.in_features,out_features=self.hidden_dims)
        self.fc2 = nn.Linear(in_features=self.hidden_dims,out_features=self.hidden_dims)
        #self.fc3 = nn.Linear(in_features=self.hidden_dims,out_features=self.hidden_dims)
        self.fc4 = nn.Linear(in_features=self.hidden_dims,out_features=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.model = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc4)
    
    def forward(self,x):
        out = self.model(x)
        return out

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss,self).__init__()
    def forward(self,predicted,target):
        pass

class MultitaskGPModel(gpytorch.models.ExactGP):
    """
        Multi-task GP model for dynamics x_{t+1} = f(x_t, u_t)
        Each output dimension of x_{t+1} is represented with a seperate GP
    """

    def __init__(self, train_x, train_y, likelihood,kernel="RBF"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        # --- Your code here
        self.mean_module = gpytorch.means.ZeroMean(size=train_x.size)
        
        if kernel=="RBF":
            self.covar_module = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=5))
        elif kernel=="Polynomial":
            self.covar_module = ScaleKernel(base_kernel=PolynomialKernel(power=5))
        elif kernel=="LinearCosine":
            self.covar_module = ScaleKernel(base_kernel = LinearKernel() + CosineKernel())

    def forward(self, x):
        """

        Args:
            x: torch.tensor of shape (B, dx + du) concatenated state and action

        Returns: gpytorch.distributions.MultitaskMultivariateNormal - Gaussian prediction for next state

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    def grad_mu(self, x):
        """
        Compute the gradient of the mean function
        Args:
            x: torch.tensor of shape (B, dx + du) concatenated state and action

        Returns:
            grad_mu: torch.tensor of shape (B, dx, dx + du) torch.tensor which is the Jacobian of the mean function

        """
        flatten = False
        if len(x.shape) < 2:
            M = 1
            x = x.reshape(M, -1)
            flatten = True

        # Get GP train x and y
        X = self.train_inputs[0]
        y = self.train_targets
        # N is datset size, M is query size
        N = X.shape[0]
        M = x.shape[0]

        # Compute difference
        diff = x.unsqueeze(1) - X.unsqueeze(0)  # M x N x d difference
        lengthscale = self.covar_module.base_kernel.lengthscale  # 2 x d
        W = 1.0 / (lengthscale ** 2)

        # Compute exponential term
        sq_diff = torch.sum(diff.unsqueeze(0) ** 2 * W.reshape(2, 1, 1, -1), dim=-1)  # 2 x M x N
        exponential_term = torch.exp(-0.5 * sq_diff)  # 2 x M x N

        # Compute gradient of Kernel
        sigma_f_sq = self.covar_module.outputscale

        # grad should be 2 x M x N x d
        grad_K = -W.reshape(2, 1, 1, -1) * diff.reshape(1, M, N, -1) * \
                 sigma_f_sq.reshape(2, 1, 1, 1) * exponential_term.unsqueeze(-1)

        # Compute gradient of mean function
        K = self.covar_module(X).evaluate()
        sigma_n_sq = self.likelihood.noise
        eye = torch.eye(N, device=x.device).reshape(1, N, N).repeat(2, 1, 1)
        mu = torch.linalg.solve(K + sigma_n_sq * eye, y.permute(1, 0).unsqueeze(-1))
        grad_mu = (grad_K.permute(0, 1, 3, 2) @ mu.reshape(2, 1, N, -1)).reshape(2, M, -1)

        if flatten:
            return grad_mu.reshape(2, -1)

        return grad_mu.permute(1, 0, 2)  # return shape M x 2 x 5

class Single_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood,kernel="RBF"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        
        self.mean_module = gpytorch.means.ZeroMean(size=train_x.size)
        if kernel=="RBF":
            self.covar_module = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=5))
        elif kernel=="Polynomial":
            self.covar_module = ScaleKernel(base_kernel=PolynomialKernel(power=5))
        elif kernel=="LinearCosine":
            self.covar_module = ScaleKernel(base_kernel = LinearKernel() + CosineKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_hyperparams(model, likelihood, train_x, train_y, lr):
    """
        Function which optimizes the GP Kernel & likelihood hyperparameters
    Args:
        model: gpytorch.model.ExactGP model
        likelihood: gpytorch likelihood
        train_x: (N, dx) torch.tensor of training inputs
        train_y: (N, dy) torch.tensor of training targets
        lr: Learning rate

    """
    # --- Your code here
    optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter=50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

def plot_gp_predictions(model, likelihood, train_x, train_y, test_x, title):
    """
        Generates GP plots for GP defined by model & likelihood
    """
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        lower, upper = observed_pred.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-10, 10])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(title)
    plt.show()


class MyModel(nn.Module):

    def __init__(self,hidden_dims=100,in_features=2,num_classes=24):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features=self.in_features,out_features=self.hidden_dims)
        self.fc2 = nn.Linear(in_features=self.hidden_dims,out_features=self.hidden_dims)
        self.fc3 = nn.Linear(in_features=self.hidden_dims,out_features=self.hidden_dims)
        self.fc4 = nn.Linear(in_features=self.hidden_dims,out_features=num_classes)
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
            self.softmax)
    
    def forward(self,x):
        out = self.model(x)
        return out

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss,self).__init__()
    
    def forward(self,y_val,num_spatial_points):    
        y_val = y_val.repeat(1,num_spatial_points)
        mean_squared = torch.nn.MSELoss()
        loss = -mean_squared(y_val,y_val.t())
        return loss

def train_step_classifier(model,train_loader,optimizer,loss_class):
    train_loss = 0.
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x = data[:,:2].detach()
        target = data[:,2:].detach()
        predicted_coords = model(x)
        loss = loss_class(input=predicted_coords,target=target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def train_step(model,train_loader,optimizer,loss_class,encoder=True):
    train_loss = 0.

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        if encoder:
            num_spatial_points,num_temporal_points = data.shape
            pred_next_state = model(data)
            loss = loss_class(pred_next_state,num_spatial_points)
        else:
            x = data[:,:2]
            target = data[:,2:]
            pred_next_state = model(x)
            loss = loss_class(input=pred_next_state,target=target)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def train_classifier_model(model,train_loader,lr=0.1,num_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch_i in pbar:
      train_loss_i = None
      optimizer.zero_grad()
      train_loss_i = train_step_classifier(model,train_loader,optimizer,loss_class=loss_fn)
      pbar.set_description(f'Train Loss: {train_loss_i:.4f} ')
      train_losses.append(train_loss_i)
    return train_losses

def train_model(model,train_loader,lr=0.01,num_epochs=1000,loss_fn=None,encoder=True):
    optimizer = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    if encoder:
        loss_fn = CustomLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    for epoch_i in pbar:
      train_loss_i = None
      val_loss_i = None  
      optimizer.zero_grad()
      train_loss_i = train_step(model,train_loader,optimizer,loss_class=loss_fn,encoder=encoder)
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

# Training Encoder

train_loader = DataLoader(train_data,shuffle=False,batch_size=train_data.shape[0])
model = Encoder(in_features=train_data.shape[1])

y = [i for i in range(24)]
latent_space = model(train_data)
latent_space/=torch.mean(latent_space)
plt.scatter(latent_space.detach().numpy(),y,label="Before Training")
#plt.show()

train_losses = train_model(
    model = model,
    train_loader=train_loader,num_epochs=5000)

# Training instantaneous RSSI values

train_final_x = train_data.flatten().unsqueeze(1)

# Inferring from Encoder
latent_space = model(train_data)
# Normalizing Latent Space

latent_space/=torch.mean(latent_space)

plt.scatter(latent_space.detach().numpy(),y,label="After training")
plt.legend()
plt.show()
#print(latent_space)
train_final_x = torch.cat((train_final_x,latent_space.repeat(len(filenames*NUM_POINTS_IN_TIME),1)),dim=1)
encodings = torch.Tensor([i for i in range(NUM_POINTS_IN_SPACE)])
encodings = nn.functional.one_hot(encodings.repeat(len(filenames)*NUM_POINTS_IN_TIME).long())
train = torch.cat((train_final_x,encodings),dim=1)



# Training classifier

# Using encoder to train classifier
train_classification_loader = DataLoader(train,batch_size=len(macs_coords),shuffle=True)

model_classification = MyModel()
#
#train_losses = train_classifier_model(
#    model = model_classification,
#    train_loader=train_classification_loader)

#print(model_classification(train[:,:2]))

"""    
train_x = 
y1,y2,y3,model(x11,x12,x13,x14,..,x1n)
Forward Pass
model = Encoder()
model(x)

"""

#time_kernels=[]
#y_time_kernels=[]




#for i in macs_coords:
#    mac_values = dataset_points1[macs[0]][i]
#    train_x = torch.Tensor(dataset_points1[macs[0]][i])
#    kernel_time = stats.gaussian_kde(train_x)
#    print(len(train_x))
#    y_kernel_time = kernel_time(train_x)
#    time_kernels.append(kernel_time)
    #y_time_kernels.append(y_kernel_time)

#print(y_time_kernels)

#test_x = torch.Tensor(dataset_points2[macs[0]][i])


#mac_values = dataset_points1[macs[0]][(0,0)]

#train_y = torch.Tensor([i for i in range(len(mac_values))])
#train_x = torch.Tensor(dataset_points1[macs[0]][(0,0)])
#test_x = torch.Tensor(dataset_points2[macs[0]][(0,0)])


#for i in range(len)

#train_y = np.array([i for i in range(len(mac_values))])
#train_x = np.array(dataset_points1[macs[0]][(0,0)])
#test_x = np.array(dataset_points2[macs[0]][(0,0)])
#kernel = stats.gaussian_kde(train_x)
#train_Y = kernel(train_x)


#plt.figure()
#plt.plot(train_Y,label="train_x")
#plt.plot(Z,label="Z")
#plt.legend()

#plt.show()

#print(train_x.shape)
#print(test_x.shape)
#print(Z.shape)
#rbf_likelihood = gpytorch.likelihoods.GaussianLikelihood()
#rbf_model = Single_GP(train_x, train_y, rbf_likelihood)

#plot_gp_predictions(rbf_model, rbf_likelihood, train_x, train_y, test_x, title='RBF kernel')