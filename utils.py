import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

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


# define a time decoder
class TimeDecoder(torch.nn.Module):
    def __init__(self, channel_list=None, activation=None):

        # channel_list: list of integers, each integer is the number of channels in a layer
        # activation: activation function to be used in the model

        # initialize the parent class, this is required for torch.nn.Module
        super().__init__()

        # if no channel_list is provided, use the default channel_list ([1024,1024,1024])
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

        # add a layer going from the input one number (t) to the number of channels in the first layer
        self.layers.append(torch.nn.Linear(1, channel_list[0]))

        # add a layer going from the number of channels in the (n)th layer to the number of channels in the (n+1)th layer
        for i in range(0,len(channel_list) - 1):
            self.layers.append(torch.nn.Linear(channel_list[i], channel_list[i + 1]))
        
        # add a layer going from the number of channels in the last layer to 784 (time-dependent features)
        self.layers.append(torch.nn.Linear(channel_list[-1], 784))

    def forward(self, t):

        # t: input tensor of shape [batch_size, 1]
        assert t.shape[-1] == (1), "input tensor must have shape [batch_size, 1]" 
        
        # flatten the input tensor to shape [batch_size, 784]
        t_decoded = t.view(-1, 1)

        # apply each layer in self.layers to the input tensor
        for layer in self.layers:
            # linear part of the layer
            t_decoded = layer(t_decoded)
            # non-linear activation function
            t_decoded = self.activation()(t_decoded)

        # reshape the output to the original batch shape but with 784 time-dependent features
        t_decoded = t_decoded.view(t.shape[:-1] + (784,))
        return t_decoded


# define the diffusion denoising probabilistic model (DDPM)
class DDPM(torch.nn.Module):
    def __init__(self, denoiser_channel_list=None, denoiser_activation=None, time_decoder_channel_list=None, time_decoder_activation=None):

        # channel_list: list of integers, each integer is the number of channels in a layer
        # activation: activation function to be used in the model

        # initialize the parent class, this is required for torch.nn.Module
        super().__init__()

        # initialize the denoiser
        self.denoiser = Denoiser(channel_list=denoiser_channel_list, activation=denoiser_activation)

        # initialize the time decoder
        self.time_decoder = TimeDecoder(channel_list=time_decoder_channel_list, activation=time_decoder_activation)

    def forward(self, x, t):
            
            # x: input tensor of shape [batch_size, 28,28]
            # t: input tensor of shape [batch_size, 1]

            assert x.shape[-2:] == (28,28), "input tensor must have shape [batch_size, 28, 28]" 
            x_shape = x.shape
            
            # flatten the input tensor to shape [batch_size, 784]
            x = x.view(-1, 784)

            # apply the time decoder to the time
            t_decoded = self.time_decoder(t)
            t_decoded = t_decoded.view(-1, 784)

            # concatenate the image and the time-decoded features
            x_t = torch.cat((x, t_decoded), dim=-1)

            # apply the denoiser to the concatenated tensor
            noise_pred = self.denoiser(x_t)
            
            # reshape the output to the image shape
            noise_pred = noise_pred.view(x_shape)
    
            return noise_pred
