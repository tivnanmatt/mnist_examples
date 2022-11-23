
import torch
import matplotlib.pyplot as plt

from utils import DDPM
from utils import train_data
from utils import test_data


# if a GPU is available, use it, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the batch size
batch_size = 256

# define the learning_rate
learning_rate = 1e-4

# load models?
loadModels = False

# save models?
saveModels = True


# function to train the model
def train(model, loss_fn, optimizer, train_loader, epochs=1):

    # loop over the epochs
    for epoch in range(epochs):

        # loop over the batches
        for batch, (X, _) in enumerate(train_loader):

            # gather the input data
            X = X.to(device)

            # get the batch size
            batch_size = X.shape[0]

            # uniform random time between 0 and 1
            t = torch.rand((batch_size,1,1,1)).to(device)

            # compute alpha_bar
            alpha_bar = torch.exp(-5*t)

            # compute the input image sample at time t
            noise = torch.randn_like(X).to(device)
            X_t = torch.sqrt(alpha_bar) * X + torch.sqrt(1 - alpha_bar)*noise

            # make a prediction
            X_t = X_t.view(batch_size,784)
            noise_pred = model(X_t,t)
            noise_pred = noise_pred.view(X.shape)
            X_t = X_t.view(X.shape)

            # compute the loss
            loss = loss_fn(noise_pred, noise)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # update the model
            optimizer.step()

            # compute the reference loss
            loss_ref = loss_fn(noise_pred*0, noise)

            # print the loss
            print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item()}, loss/loss_ref: {loss.item()/loss_ref.item()}')

# function to test the model
def test(model, batch_size, num_steps):

    # do not track gradients during evaluation
    with torch.no_grad():

        # initialize X_t as noise
        X_t = torch.randn((batch_size,1,28,28)).to(device)

        # determine dt
        dt = 1/num_steps

        # loop over time steps
        for n in torch.linspace(1,0,num_steps):

            # compute the time t
            t = torch.ones((batch_size,1,1,1)).to(device) * n
            
            # compute alpha_bar and alpha
            alpha_bar = torch.exp(-5*t)
            alpha = alpha_bar/torch.exp(-5*(t-dt)) 

            # make a prediction
            noise_pred = model.eval()(X_t.view(-1,784),t)
            noise_pred = noise_pred.view(X_t.shape)

            # update X_t
            X_t = (X_t - ((1-alpha)/torch.sqrt(1-alpha_bar)) * noise_pred)* (1/torch.sqrt(alpha))
            X_t  = X_t + torch.sqrt(1-alpha) * torch.randn_like(X_t)


    # return the final image
    return X_t


# define the model
model = DDPM().to(device)

# uncomment below if multiple GPUs are available to use DataParallel
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

# create the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
# define a categorical cross entropy loss function
loss_fn = torch.nn.MSELoss()

# define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# keep training the model until 1000 seconds have passed
import time
start_time = time.time()


if loadModels:
    model.load_state_dict(torch.load('weights/ddpm.pt'))
    optimizer.load_state_dict(torch.load('weights/ddpm_opt.pt'))

train(model, loss_fn, optimizer, train_loader, epochs=10)

if saveModels:
    torch.save(model.state_dict(), 'weights/ddpm.pt')
    torch.save(optimizer.state_dict(), 'weights/ddpm_opt.pt')

test(model, batch_size, num_steps=512)