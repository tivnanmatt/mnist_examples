import torch
from common import train_data, test_data

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

        # add a layer going from the input [28, 28] = 784 to the number of channels in the first layer
        self.layers.append(torch.nn.Linear(784, channel_list[0]))

        # add a layer going from the number of channels in the (n)th layer to the number of channels in the (n+1)th layer
        for i in range(0,len(channel_list) - 1):
            self.layers.append(torch.nn.Linear(channel_list[i], channel_list[i + 1]))
        
        # add a layer going from the number of channels in the last layer to the number of classes (10)
        self.layers.append(torch.nn.Linear(channel_list[-1], 784))

    def forward(self, x):

        # x: input tensor of shape [batch_size, 28,28]
        assert x.shape[-2:] == (28,28), "input tensor must have shape [batch_size, 28, 28]" 
        x_shape = x.shape
        
        # flatten the input tensor to shape [batch_size, 784]
        x = x.view(-1, 784)

        # apply each layer in self.layers to the input tensor
        for layer in self.layers:
            # linear part of the layer
            x = layer(x)
            # non-linear activation function
            x = self.activation()(x)

        # reshape the output to the original shape
        x = x.view(x_shape)

        return x

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
            X_pred = model(X_noisy)

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

if __name__ == '__main__':

    # if a GPU is available, use it, otherwise use the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define the batch size
    batch_size = 256

    # define the learning_rate
    learning_rate = 1e-4

    # define the model
    model = Denoiser([1024, 1024, 1024]).to(device)

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

    # keep training the model until 2 minutes have passed
    import time
    start_time = time.time()
    while time.time() - start_time < 120:
        train(model, loss_fn, optimizer, train_loader, epochs=1)
    test(model, test_loader)