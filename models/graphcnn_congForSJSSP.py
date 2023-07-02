import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
# import sys
# sys.path.append("models/")

'''
class Attention(nn.Module):
    def __init__(self): super(Attention, self).__init__()

    def forward(self, g_fea, candidates_feas):
        attention_score = torch.mm(candidates_feas, g_fea.t())
        attention_weight = F.softmax(attention_score, dim=0)
        representation_weighted = torch.mm(attention_weight.t(), candidates_feas)
        feas_final = torch.cat((g_fea, representation_weighted), dim=1)
        return feas_final
'''
def __reset_param_impl__(cnn_net):
    """
    """
    # --- do init ---
    conv = cnn_net.conv
    n1   = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n1))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel):
        super().__init__() # necessary
        self.conv = nn.Conv2d(cin, cout, kernel, padding=1)
        self.bn   = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def reset_param(self):
        #normalize the para of cnn network
        __reset_param_impl__(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims,kernel):
        super(Net, self).__init__()

        # convolution layers
        self.conv_1  = nn.Conv2d(3 , hidden_dims[0], kernel[0], stride=1)
        self.conv_2  = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel[1], stride=1)
        self.conv_3  = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel[2], stride=1)
        self.bn_1   = nn.BatchNorm2d(hidden_dims[0])
        self.bn_2   = nn.BatchNorm2d(hidden_dims[1])
        self.bn_3   = nn.BatchNorm2d(hidden_dims[2])
        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()

        #normalize the para of cnn network
        n1 = self.conv_1.kernel_size[0] * self.conv_1.kernel_size[1] * self.conv_1.out_channels
        n2 = self.conv_2.kernel_size[0] * self.conv_2.kernel_size[1] * self.conv_2.out_channels
        n3 = self.conv_3.kernel_size[0] * self.conv_3.kernel_size[1] * self.conv_3.out_channels
        self.conv_1.weight.data.normal_(0, math.sqrt(2. / n1))
        self.conv_2.weight.data.normal_(0, math.sqrt(2. / n2))
        self.conv_3.weight.data.normal_(0, math.sqrt(2. / n3))

        self.cnn = nn.Sequential(
            self.conv_1, self.bn_1, self.relu, #,
            self.conv_2, self.bn_2, self.relu,
            self.conv_3, self.bn_3, self.relu
        )

        # check the output of cnn, which is [fc1_dims]
        self.fcn_inputs_length = self.cnn_out_dim(input_dim)

        # fully connected layers
        self.fc1 =  nn.Linear(self.fcn_inputs_length, hidden_dims[2])

        self.fc1.weight.data.normal_(0, 0.1)


        self.fcn = nn.Sequential(
            self.fc1

        )

    def forward(self, x):
        '''
            - x : tensor in shape of (N, state_dim)
        '''
        cnn_out   = self.cnn(x)
        cnn_out   = cnn_out.reshape(-1, self.fcn_inputs_length)
        fcn_input = self.flatten(cnn_out)
        fea_State   = self.fcn(fcn_input)
        return fea_State

    def cnn_out_dim(self, input_dims):
        return self.cnn(torch.zeros(1, *input_dims)
                       ).flatten().shape[0]
# 定义网络结构

class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''

        super(GraphCNN, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):

        # pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            # print(Adj_block.dtype)
            # print(h.dtype)
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj):

        x_concat = x
        graph_pool = graph_pool

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            Adj_block = adj

        # list of hidden representation at each layer (including input)
        h = x_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

        h_nodes = h.clone()
        # print(graph_pool.shape, h.shape)
        pooled_h = torch.sparse.mm(graph_pool, h)
        # pooled_h = graph_pool.spmm(h)

        return pooled_h, h_nodes


if __name__ == '__main__':

    ''' Test attention block
    attention = Attention()
    g = torch.tensor([[1., 2.]], requires_grad=True)
    candidates = torch.tensor([[3., 3.],
                               [2., 2.]], requires_grad=True)

    ret = attention(g, candidates)
    print(ret)
    loss = ret.sum()
    print(loss)

    grad = torch.autograd.grad(loss, g)

    print(grad)
    '''