
import torch
import matplotlib.pyplot as plt

from utils import Denoiser
from utils import TimeDecoder
from utils import train_data
from utils import test_data

# define a denoiser model
class Denoiser(torch.nn.Module):
    def __init__(self, channel_list=None, activation=None):

        # channel_list: list of integers, each integer is the number of channels in a layer
        # activation: activation function to be used in the model

        # initialize the parent class, this is required for torch.nn.Module
        super().__init__()

        # if no channel_list is provided, use the default channel_list ([128, 64, 32])
        if channel_list is None:
            channel_list = [1024, 1024, 1024]
        assert len(channel_list) > 0, "channel_list must have at least one element"
        self.channel_list = channel_list

        # if no activation function is provided, use the default activation function (ReLU)
        if activation is None:
            activation = torch.nn.ReLU
        assert callable(activation), "activation must be callable"
        self.activation = activation

        # initialize an empty list of layers
        self.layers = torch.nn.ModuleList()

        # add a layer going from the input 784*2=1568 to the number of channels in the first layer
        # the first 784 channels represent the input image
        # the second 784 channels represent the time-decoder output
        self.layers.append(torch.nn.Linear(1568, channel_list[0]))

        # add a layer going from the number of channels in the (n)th layer to the number of channels in the (n+1)th layer
        for i in range(0,len(channel_list) - 1):
            self.layers.append(torch.nn.Linear(channel_list[i], channel_list[i + 1]))
        
        # add a layer going from the number of channels in the last layer to the number of classes (10)
        self.layers.append(torch.nn.Linear(channel_list[-1], 784))

    def forward(self, x):

        # x: input tensor of shape [batch_size, 1568]
        assert x.shape[-1] == (1568), "input tensor must have shape [batch_size, 1568]" 
        x_shape = x.shape
        
        # flatten the input tensor to shape [batch_size, 784]
        x = x.view(-1, 1568)

        # apply each layer in self.layers to the input tensor
        for layer in self.layers[0:-1]:
            # linear part of the layer
            x = layer(x)
            # non-linear activation function
            x = self.activation()(x)
        # linear part of the layer
        x = self.layers[-1](x)

        # reshape the output to the image shape
        x = x.view(x_shape[:-1] + (784,))

        return x



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
            noise_pred = model(X_t,t)

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
            noise_pred = model.eval()(X_t,t)

            # update X_t
            X_t = (X_t - ((1-alpha)/torch.sqrt(1-alpha_bar)) * noise_pred)* (1/torch.sqrt(alpha))
            X_t  = X_t + torch.sqrt(1-alpha) * torch.randn_like(X_t)


    # return the final image
    return X_t


if __name__ == '__main__':

    # if a GPU is available, use it, otherwise use the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define the batch size
    batch_size = 256

    # define the learning_rate
    learning_rate = 1e-4

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
    while time.time() - start_time < 1000:
        train(model, loss_fn, optimizer, train_loader, epochs=1)
    
    test(model, batch_size, num_steps=128)