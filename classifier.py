import torch
from common import train_data, test_data

# define a classifier model
class Classifier(torch.nn.Module):
    def __init__(self, channel_list=None, activation=None):

        # channel_list: list of integers, each integer is the number of channels in a layer
        # activation: activation function to be used in the model

        # initialize the parent class, this is required for torch.nn.Module
        super().__init__()

        # if no channel_list is provided, use the default channel_list ([128, 64, 32])
        if channel_list is None:
            channel_list = [128, 64, 32]
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
        self.layers.append(torch.nn.Linear(channel_list[-1], 10))

        # add a head layer to convert the output of the last layer to a number between 0 and 1
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):

        # x: input tensor of shape [batch_size, 28,28]
        assert x.shape[-2:] == (28,28), "input tensor must have shape [batch_size, 28, 28]" 
        
        # flatten the input tensor to shape [batch_size, 784]
        x = x.view(-1, 784)

        # apply each layer in self.layers to the input tensor
        for layer in self.layers:
            # linear part of the layer
            x = layer(x)
            # non-linear activation function
            x = self.activation()(x)

        # apply the softmax function so that the output is between 0 and 1
        x = self.log_softmax(x)
        return x

# function to train the model
def train(model, loss_fn, optimizer, train_loader, epochs=1):

    # loop over the epochs
    for epoch in range(epochs):

        # loop over the batches
        for batch, (X, y) in enumerate(train_loader):

            # gather the input data
            X = X.to(device)
            y = y.to(device)

            # make a prediction
            y_pred = model(X)

            # compute the loss
            loss = loss_fn(y_pred, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # update the model
            optimizer.step()

            # print the loss
            print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item()}')

# function to test the model
def test(model, test_loader):

    # initialize the number of total predictions
    sum_total = 0

    # initialize the number of correct predictions
    sum_correct = 0

    # do not track gradients during evaluation
    with torch.no_grad():

        # loop over the batches
        for batch, (X, y) in enumerate(test_loader):

            # gather the input data
            X = X.to(device)
            y = y.to(device)
            
            # make a prediction
            y_pred = model(X)

            # compute the accuracy
            sum_correct += torch.sum(torch.argmax(y_pred, dim=1) == y)
            sum_total += y.shape[0]

            # print the accuracy
            print(f'batch: {batch}, accuracy: {sum_correct/sum_total}')

if __name__ == '__main__':

    # if a GPU is available, use it, otherwise use the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define the batch size
    batch_size = 1024

    # define the learning_rate
    learning_rate = 1e-4

    # define the model
    model = Classifier([1024, 512, 256, 128, 64, 32, 16]).to(device)

    # uncomment below if multiple GPUs are available to use DataParallel
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
        # batch_size = batch_size * torch.cuda.device_count()

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        
	# define a categorical cross entropy loss function
    loss_fn = torch.nn.NLLLoss()

	# define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # keep training the model until 2 minutes have passed
    import time
    start_time = time.time()
    while time.time() - start_time < 120:
        train(model, loss_fn, optimizer, train_loader, epochs=1)
    test(model, test_loader)