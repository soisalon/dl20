

import torch
from torch import nn
import torch.nn.functional as F

from vars import ACTIVATIONS


class BaseCNN(nn.Module):

    def __init__(self, params):
        super(BaseCNN, self).__init__()

        n_classes = 126

        in_height, in_width = tuple(map(int, params.input_shape.split('x')))
        n_kernels = params.n_kernels[0]
        kernel_shape = [tuple(map(int, s.split('x'))) for s in params.kernel_shapes][0]
        stride = [tuple(map(int, s.split('x'))) for s in params.strides][0]

        h_units = params.h_units[0]       # hidden units in the fully-connected layer

        # layers
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=kernel_shape,
                              stride=stride)

        pool_size = abs((in_width - kernel_shape[1]) // stride[1]) + 1
        self.pooling = nn.MaxPool1d(kernel_size=pool_size)

        self.fc = nn.Linear(in_features=n_kernels, out_features=h_units)
        self.dropout = nn.Dropout(p=params.dropout)
        self.out = nn.Linear(in_features=h_units, out_features=n_classes)

    def forward(self, x):

        h = self.conv(x)
        h = h.squeeze()
        h = F.relu(h)
        h = self.pooling(h)
        h = torch.flatten(h, start_dim=1)
        h = self.fc(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.out(h)
        return torch.sigmoid(h)


class DocCNN(nn.Module):

    def __init__(self, params):
        super(DocCNN, self).__init__()

        n_classes = 126

        in_height, in_width = tuple(map(int, params.input_shape.split('x')))

        n_conv_layers = params.n_conv_layers
        n_kernels = params.n_kernels

        kernel_shapes = [tuple(map(int, s.split('x'))) for s in params.kernel_shapes]
        strides = [tuple(map(int, s.split('x'))) for s in params.strides]
        pool_sizes = [tuple(map(int, s.split('x'))) for s in params.pool_sizes]

        conv_act_fn = ACTIVATIONS[params.conv_act_fn]      # activation function in conv. layers

        h_units = params.h_units                           # hidden units in each fully-connected layer
        fc_act_fn = ACTIVATIONS[params.fc_act_fn]          # activation function in FC layers

        out_act_fn = ACTIVATIONS[params.out_act_fn]        # activation function in output layer
        dropout = params.dropout

        # get convnet
        conv_layers = []
        ws = [in_width]      # widths after each convolution/pooling operation
        hs = [in_height]
        for i in range(n_conv_layers):
            in_channels = 1 if i == 0 else n_kernels[i - 1]
            conv_layers += [nn.Conv2d(in_channels, n_kernels[i], kernel_size=kernel_shapes[i],
                                      stride=strides[i])]
            conv_layers += [conv_act_fn]
            conv_layers += [nn.MaxPool2d(kernel_size=pool_sizes[i])]

            h = abs((hs[i] - kernel_shapes[i][0]) // strides[i][0]) + 1
            hs += [abs((h - pool_sizes[i][0]) // pool_sizes[i][0]) + 1]
            w = abs((ws[i] - kernel_shapes[i][1]) // strides[i][1]) + 1
            ws += [abs((w - pool_sizes[i][1]) // pool_sizes[i][1]) + 1]

        n_weights = int(n_kernels[-1] * ws[-1] * hs[-1])     # num. of weights after the conv layers

        fc_layers = []
        for i in range(len(h_units)):
            in_feats = n_weights if i == 0 else h_units[i - 1]
            fc_layers += [nn.Linear(in_features=in_feats, out_features=h_units[i])]
            fc_layers += [fc_act_fn]

        fc_layers += [nn.Dropout(p=dropout)]

        fc_layers += [nn.Linear(in_features=h_units[-1], out_features=n_classes)]
        fc_layers += [out_act_fn]

        self.net = nn.Sequential(*conv_layers, nn.Flatten(), *fc_layers)

        # for debugging
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers

    def forward(self, x):
        h = self.conv_layers[0](x)
        for l in self.conv_layers[1:]:
            h = l(h)
        h = torch.flatten(h, start_dim=1)
        for l in self.fc_layers:
            h = l(h)
        return h

        # return self.net(x)
