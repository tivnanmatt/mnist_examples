import torch
import matplotlib.pyplot as plt

from utils import Classifier
from utils import train_data
from utils import test_data


# if a GPU is available, use it, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the batch size
batch_size = 1024

# define the learning_rate
learning_rate = 1e-4

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
            y_pred = model(X.view(-1, 784))

            # compute the accuracy
            sum_correct += torch.sum(torch.argmax(y_pred, dim=1) == y)
            sum_total += y.shape[0]

            # print the accuracy
            print(f'batch: {batch}, accuracy: {sum_correct/sum_total}')


# define the model
model = Classifier().to(device)

# uncomment below if multiple GPUs are available to use DataParallel
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = torch.nn.DataParallel(model)

# create the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
# define a categorical cross entropy loss function
loss_fn = torch.nn.NLLLoss()

# define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# keep training the model until 2 minutes have passed
import time
start_time = time.time()
while time.time() - start_time < 120:
    train(model, loss_fn, optimizer, train_loader, epochs=1)

# test the model
test(model, test_loader)