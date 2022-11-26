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

# define a MultiLayerPerceptron model
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, channel_list=None, activation=None, final_activation=None, **kwargs):

        # channel_list: list of integers, each integer is the number of channels in a layer
        # activation: activation function to be used in the model
        # final_activation: activation function to be used in the final layer

        # initialize the parent class, this is required for torch.nn.Module
        super().__init__(*kwargs)

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

        if final_activation is None:
            final_activation = torch.nn.Identity    
        assert callable(final_activation), "final_activation must be callable"
        self.final_activation = final_activation

        # initialize an empty list of layers
        self.linear_layers = torch.nn.ModuleList()
        self.batchnorm_layers = torch.nn.ModuleList()

        # add a layer going from the number of channels in the (n)th layer to the number of channels in the (n+1)th layer
        for i in range(0,len(channel_list) - 1):
            self.linear_layers.append(torch.nn.Linear(channel_list[i], channel_list[i + 1]))
            self.batchnorm_layers.append(torch.nn.BatchNorm1d(channel_list[i + 1]))

    def forward(self, x):

        # x: input tensor of shape [batch_size, 28,28]
        assert x.shape[-1] == self.channel_list[0], "input tensor must have shape [batch_size, " + str(self.channel_list[0]) + "]" 
        x_shape = x.shape

        x = x.view(-1, self.channel_list[0])

        # apply each layer in self.layers to the input tensor
        for iLayer in range(len(self.linear_layers)):
            # linear part of the layer
            x = self.linear_layers[iLayer](x)
            # batch normalization
            x = self.batchnorm_layers[iLayer](x)
            # if its the last layer, apply the final activation function
            if iLayer == len(self.linear_layers) - 1:
                x = self.final_activation()(x)
            # otherwise, apply the activation function
            else:
                x = self.activation()(x)

        # reshape the output to the output shape
        x = x.view(x_shape[:-1] + (self.channel_list[-1],))

        return x


class Classifier(MultiLayerPerceptron):
    def __init__(self, channel_list=None, activation=None, final_activation=None, **kwargs):
            
            # channel_list: list of integers, each integer is the number of channels in a layer
            # activation: activation function to be used in the model
            # final_activation: activation function to be used in the last layer of the model

            if channel_list is None:
                channel_list = [784, 2048, 1024,512,256,128,64,10]

            if activation is None:
                activation = torch.nn.ReLU

            if final_activation is None:
                final_activation = torch.nn.Softmax
    
            # initialize the parent class, this is required for torch.nn.Module
            super().__init__(channel_list, activation, final_activation, **kwargs)


class Denoiser(MultiLayerPerceptron):
    def __init__(self, channel_list=None, activation=None, final_activation=None, **kwargs):
            
            # channel_list: list of integers, each integer is the number of channels in a layer
            # activation: activation function to be used in the model
            # final_activation: activation function to be used in the last layer of the model

            if channel_list is None:
                channel_list = [784, 2048, 2048,2048,2048,784]

            if activation is None:
                activation = torch.nn.ReLU

            if final_activation is None:
                final_activation = torch.nn.Identity
    
            # initialize the parent class, this is required for torch.nn.Module
            super().__init__(channel_list, activation, final_activation, **kwargs)

class TimeDecoder(MultiLayerPerceptron):
    def __init__(self, channel_list=None, activation=None, final_activation=None, **kwargs):
            
            # channel_list: list of integers, each integer is the number of channels in a layer
            # activation: activation function to be used in the model
            # final_activation: activation function to be used in the last layer of the model

            if channel_list is None:
                channel_list = [1, 2048, 1024,512,128,64]

            if activation is None:
                activation = torch.nn.ReLU

            if final_activation is None:
                final_activation = torch.nn.Identity
    
            # initialize the parent class, this is required for torch.nn.Module
            super().__init__(channel_list, activation, final_activation, **kwargs)

# define the diffusion denoising probabilistic model (DDPM)
class DDPM(torch.nn.Module):
    def __init__(self, denoiser_channel_list=None, denoiser_activation=None, denoiser_final_activation=None, time_decoder_channel_list=None, time_decoder_activation=None, time_decoder_final_activation=None):
        super().__init__()

        if denoiser_channel_list is None:
            denoiser_channel_list = [784+64, 2048, 2048,2048,2048,784]

        if denoiser_activation is None:
            denoiser_activation = torch.nn.ReLU

        if denoiser_final_activation is None:
            denoiser_final_activation = torch.nn.Identity

        if time_decoder_channel_list is None:
            time_decoder_channel_list = [1, 2048, 1024,512,128,64]
        
        if time_decoder_activation is None:
            time_decoder_activation = torch.nn.ReLU

        if time_decoder_final_activation is None:
            time_decoder_final_activation = torch.nn.Identity

        # initialize the denoiser
        self.denoiser = Denoiser(denoiser_channel_list, denoiser_activation, denoiser_final_activation)

        # initialize the time decoder
        self.time_decoder = TimeDecoder(time_decoder_channel_list, time_decoder_activation, time_decoder_final_activation)

    def forward(self, x, t):

        # x: input tensor of shape [batch_size, 784]
        # t: time tensor of shape [batch_size, 1]

        # decode the time
        t_decoded = self.time_decoder(t)

        # concatenate the time and the input
        x_t = torch.cat((x, t_decoded), dim=1)

        # denoise the input
        noise_pred = self.denoiser(x_t)

        return noise_pred