import torch
from common import train_data, test_data

# define some constants
batch_size = 64

# create the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# define a classifier model
class Classifier(torch.nn.Module):
    def __init__(self, channel_list=None, activation=None):
        super().__init__()
        if channel_list is None:
            channel_list = [128, 64, 32]
        self.channel_list = channel_list
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(784, channel_list[0]))
        for i in range(0,len(channel_list) - 1):
            self.layers.append(torch.nn.Linear(channel_list[i], channel_list[i + 1]))
        if activation is None:
            activation = torch.nn.ReLU
        self.activation = activation
        self.head = torch.nn.Linear(channel_list[-1], 10)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation()(x)
        x = self.head(x)
        # x = x/torch.sum(x)
        x = self.log_softmax(x)
        return x

model = Classifier([10,10,10,10,])

# define a categorical cross entropy loss function
loss_fn = torch.nn.NLLLoss()

# define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train the model
def train(model, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            X = X.view(-1, 28 * 28)
            # z = torch.zeros(batch_size, 10)
            # z[range(batch_size), y] = 1
            # y = z
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 1 == 0:
                print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item()}')

# test the model
def test(model, loss_fn):
    sum_correct = 0
    sum_total = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            X = X.view(-1, 28 * 28)
            # z = torch.zeros(batch_size, 10)
            # z[range(batch_size), y] = 1
            # y = z
            y_pred = model(X)
            sum_correct += torch.sum(torch.argmax(y_pred, dim=1) == y)
            sum_total += y.shape[0]
            # find the max value in each row
            if batch % 1 == 0:
                print(f'batch: {batch}, accuracy: {sum_correct/sum_total}')

if __name__ == '__main__':
    train(model, loss_fn, optimizer)
    test(model,loss_fn)

