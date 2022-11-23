
import torch
import matplotlib.pyplot as plt

from utils import Denoiser
from utils import train_data
from utils import test_data

# if a GPU is available, use it, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# function to train the model
def train(model, loss_fn, optimizer, train_loader, epochs=1):

    # loop over the epochs
    for epoch in range(epochs):

        # loop over the batches
        for batch, (X, _) in enumerate(train_loader):

            # gather the input data
            X = X.to(device)
            X_noisy = X + 0.2 * torch.randn_like(X)

            # make a prediction
            X_noisy = X_noisy.view(-1, 784)
            X_pred = model(X_noisy)
            X_noisy = X_noisy.view(X.shape)
            X_pred = X_pred.view(X.shape)

            # compute the loss
            loss = loss_fn(X_pred, X)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # update the model
            optimizer.step()

            # compute the reference loss
            loss_ref = loss_fn(X_noisy, X)

            # print the loss
            print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item()}, loss/loss_ref: {loss.item()/loss_ref.item()}')

# function to test the model
def test(model, test_loader):

    # initialize the mse running average
    mse_running_average = 0
    mse_running_average_ref = 0
    

    # do not track gradients during evaluation
    with torch.no_grad():

        # loop over the batches
        for batch, (X, _) in enumerate(test_loader):

            # gather the input data
            X = X.to(device)
            X_noisy = X + 0.2 * torch.randn_like(X)
            
            # make a prediction
            X_pred = model(X_noisy)

            # compute the mse running average for the model prediction
            mse_running_average = mse_running_average*(batch/(batch+1)) + torch.mean((X_pred - X)**2)*(1/(batch+1))

            # compute the mse running average for the reference (noisy input)
            mse_running_average_ref = mse_running_average_ref*(batch/(batch+1)) + torch.mean((X_noisy - X)**2)*(1/(batch+1))

            # print the accuracy
            print(f'batch: {batch}, mse: {mse_running_average}, mse/mse_ref: {mse_running_average/mse_running_average_ref}')


# define the batch size
batch_size = 256

# define the learning_rate
learning_rate = 1e-4

# define the denoiser model
model = Denoiser().to(device)

# uncomment below if multiple GPUs are available to use DataParallel
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

# create the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
# define a mean squared error loss function
loss_fn = torch.nn.MSELoss()

# define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# keep training the model until 2 minutes have passed
import time
start_time = time.time()
while time.time() - start_time < 120:
    train(model, loss_fn, optimizer, train_loader, epochs=1)
test(model, test_loader)