import math

import torch
from torch import nn


def __reset_param_impl__(cnn_net):
    """
    """
    # --- do init ---
    conv = cnn_net.conv
    n1 = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n1))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel):
        super().__init__()  # necessary
        self.conv = nn.Conv2d(cin, cout, kernel, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def reset_param(self):
        # normalize the para of cnn network
        __reset_param_impl__(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class cnnNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, hidden_dim,kernel):
        super(cnnNet, self).__init__()


        # convolution layers
        self.conv_1 = nn.Conv2d(3, hidden_dims[0], kernel[0], stride=1)
        self.conv_2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel[1], stride=1)
        self.conv_3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel[2], stride=1)
        self.bn_1 = nn.BatchNorm2d(hidden_dims[0])
        self.bn_2 = nn.BatchNorm2d(hidden_dims[1])
        self.bn_3 = nn.BatchNorm2d(hidden_dims[2])

        self.activate_func = [nn.ReLU(), nn.Tanh()][0]  # Trick10: use tanh

        self.flatten = nn.Flatten()

        # normalize the para of cnn network
        n1 = self.conv_1.kernel_size[0] * self.conv_1.kernel_size[1] * self.conv_1.out_channels
        n2 = self.conv_2.kernel_size[0] * self.conv_2.kernel_size[1] * self.conv_2.out_channels
        n3 = self.conv_3.kernel_size[0] * self.conv_3.kernel_size[1] * self.conv_3.out_channels

        self.conv_1.weight.data.normal_(0, math.sqrt(2. / n1))
        self.conv_2.weight.data.normal_(0, math.sqrt(2. / n2))
        self.conv_3.weight.data.normal_(0, math.sqrt(2. / n3))

        self.cnn = nn.Sequential(
            self.conv_1, self.bn_1, self.activate_func,  # ,
            self.conv_2, self.bn_2, self.activate_func,
            self.conv_3, self.bn_3, self.activate_func
        )

        # check the output of cnn, which is [fc1_dims]
        self.fcn_inputs_length = self.cnn_out_dim(input_dim)

        # fully connected layers
        self.fc1 = nn.Linear(self.fcn_inputs_length, hidden_dim)

        self.fc1.weight.data.normal_(0, 0.1)

        self.fcn = nn.Sequential(
            self.fc1

        )

    def forward(self, x):
        '''
            - x : tensor in shape of (N, state_dim)
        '''
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.reshape(-1, self.fcn_inputs_length)
        fcn_input = self.flatten(cnn_out)
        fea_State = self.fcn(fcn_input)
        return fea_State

    def cnn_out_dim(self, input_dims):
        return self.cnn(torch.zeros(1, *input_dims)
                        ).flatten().shape[0]
