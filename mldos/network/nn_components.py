import torch
from torch import nn
import numpy as np

from mldos.parameters import defined_symbol_tuple_dict, symbol_one_hot_dict
from ._graph import similarity_dict_7, similarity_dict_5
from ._model_parameter import torch_device, torch_dtype

# normalization
class linearx():
    # x = 1.0 serve as no normalization
    def __init__(self, x: float = 1.0):
        self.factor = x

    def forward(self, X):
        return X * self.factor

    def __call__(self, X):
        return X * self.factor

# components
class ActivationSharing(nn.Module):
    """averaging inputs amoung connected atoms in the periodic table
    both input and output dimension are (Nb, size_in, L), L is excepted to be in the defined elements set
    """
    def __init__(self, input_shape, share_ratio: tuple = (0.5, 0.25)):
        super().__init__()
        self.Cin, self.nele = input_shape
        
        assert self.nele == len(similarity_dict_7)

        symbol_one_hot_dict_reverse = { v:k for k, v in symbol_one_hot_dict.items() }

        self.mask = torch.diag(torch.ones(self.nele)).to(torch_device, torch_dtype)
        # symmetric
        for iele in range(self.nele):
            symbol = symbol_one_hot_dict_reverse[iele]
            similar = similarity_dict_7[symbol]
            for element in similar['1st']:
                index = symbol_one_hot_dict[element]
                self.mask[iele, index] = share_ratio[0]
            for element in similar['2nd']:
                index = symbol_one_hot_dict[element]
                self.mask[iele, index] = share_ratio[1]

    @property
    def input_shape(self):
        return (self.Cin, self.nele)
    @property

    def output_shape(self):
        return self.input_shape

    def forward(self, x):
        return torch.matmul(x, self.mask)


# module
class MultiLinear(nn.Module):
    """ linear layer network with given depth and hidden units
    input (Cin,) -> (Cout,)"""
    def __init__(self, 
        input_length: int, 
        output_length: int,
        linearlayer_connection: tuple,
        last_activation: bool = True,  # whether we add an activation layer in the end
        activation: nn.Module = None
    ):
        super().__init__()

        self.output_length = output_length
        self.input_length = input_length
        prev_connection = self.input_length

        self.activation = activation or nn.ReLU()

        linear_sequence = []
        for nconnection in linearlayer_connection:
            linear_sequence.append(nn.Linear(prev_connection, nconnection))
            linear_sequence.append(self.activation)
            prev_connection = nconnection
        linear_sequence.append(nn.Linear(prev_connection, output_length))  
        if last_activation:
            linear_sequence.append(self.activation)
        self.linearlayer = nn.Sequential(*linear_sequence)

    @property
    def input_shape(self):
        return self.input_length

    @property
    def output_shape(self):
        return self.output_length

    def forward(self, x):
        out = self.linearlayer(x)
        return out


# single linear module
class SelfInteraction(nn.Module):
    """it does self interaction on the first feature dimension"""

    def __init__(self, 
        input_shape: tuple, # (C, L)
        Cout: int,          # (Cout)
    ):
        super().__init__()

        assert len(input_shape) == 2

        self.Cin, self.Lin = input_shape
        self.Cout = Cout

        factor = 1.0 / np.sqrt(self.Cin)

        self.out_shape = (self.Cout, self.Lin)

        self.weights = nn.Parameter(torch.randn(self.Lin, self.Cin, self.Cout) * 2.0 * factor - factor + 0.01)
        self.bias = nn.Parameter(torch.randn(self.out_shape) * 2.0 * factor - factor + 0.01)

    def forward(self, x):
        out = torch.einsum("nij,jim->nmj", x, self.weights)
        out[:] += self.bias
        return out


# feature extraction
class FeatureSelfInteraction(nn.Module):
    """it does local convolution on the periodic table"""

    def __init__(self, 
        input_shape: tuple,  # (Cin, L)
        output_shape: tuple, # (Cout, L)
        hidden: tuple = None,
        activation: nn.Module = None,
        weight_sharing: bool = False
    ):
        super().__init__()
        assert output_shape[1] == input_shape[1]
        assert len(input_shape) == 2
        assert len(output_shape) == 2

        self.activation = activation or nn.ReLU()

        self.Cin, self.Lin = input_shape
        self.Cout, _ = output_shape
        self.hidden = hidden or (self.Cout,)

        linear = []
        previous = self.Cin
        for feature in self.hidden:
            linear.append(SelfInteraction((previous, self.Lin), feature))
            linear.append(self.activation)
            previous = feature
        linear.append(SelfInteraction((previous, self.Lin), self.Cout))
        linear.append(self.activation)
        self.extraction = nn.Sequential(*linear)
        
        self.output_length = previous * self.Lin

        self.flatten = nn.Flatten()
        self.weight_sharing = weight_sharing
        if self.weight_sharing:
            self.sharing = ActivationSharing((previous, self.Lin))

    @property
    def input_shape(self):
        return (self.Cin, self.Lin)

    @property
    def output_shape(self):
        return (self.Cout * self.Lin)
    

    def forward(self, x):
        out = self.extraction(x)

        if self.weight_sharing:
            out = self.sharing(out)

        out = self.flatten(out)
        return out


# feature extraction
class SimpleConvolution(nn.Module):
    def __init__(self, 
        input_shape: tuple, 
        output_shape: tuple, 
        nfeatures: tuple = None, 
        activation: nn.Module = None, 
        weight_sharing: bool = False
    ):
        super().__init__()

        nfeatures = nfeatures or output_shape
        assert len(input_shape) == 2
        assert len(output_shape) == 2

        self.Cin, self.Lin = input_shape
        self.Cout, self.Lout = output_shape
        self.output_length = self.Lout * self.Cout
        nf1, nf2 = nfeatures
        self.weight_sharing = weight_sharing

        if self.weight_sharing:
            self.sharing = ActivationSharing(input_shape)

        self.activation = activation or nn.ReLU()

        self.conv1_1 = nn.Conv1d(self.Cin, nf1, stride = 1, kernel_size = 1)
        self.conv1_2 = nn.Conv1d(nf1, self.Cout, stride = 1, kernel_size = 1)

        self.conv2_1 = nn.Conv1d(self.Lin, nf2, stride = 1, kernel_size = 1)
        self.conv2_2 = nn.Conv1d(nf2, self.Lout, stride = 1, kernel_size = 1)

        self.flatten = nn.Flatten()

    @property
    def output_shape(self):
        return self.Lout * self.Cout
        
    @property
    def input_shape(self):
        return (self.Cin, self.Lin)

    def forward(self, x):
        out = self.activation(self.conv1_1(x))
        out = self.activation(self.conv1_2(out))
        if self.weight_sharing:
            out = self.sharing(out)
        
        out = out.permute((0, 2, 1))                        # out(N, len, kernel1)
        out = self.activation(self.conv2_1(out))
        out = self.activation(self.conv2_2(out))
        #out = out.permute((0, 2, 1))                        # out(N, len, kernel1)
        out = self.flatten(out)
        #out = out.view(-1, self.output_length)
        return out


# output
class SplitOutput(nn.Module):
    def __init__(self,
        input_length: int,
        output_length: int,
        hidden_connection: tuple,
        activation: nn.Module = None
    ) -> None:
        super().__init__()

        self.output_length = output_length
        self.input_length = input_length

        self.activation = activation or nn.ReLU()

        parallel_linear = []
        for _ in range(output_length):
            parallel_linear.append(MultiLinear(self.input_length, 1, hidden_connection, last_activation=False, activation = self.activation))
        self.parallel_linear = nn.ModuleList(parallel_linear)

    @property
    def input_shape(self):
        return self.input_length
    
    @property
    def output_shape(self):
        return self.output_length

    def forward(self, x):
        outs = []
        for linear in self.parallel_linear:
            outs.append(linear(x))
        return torch.hstack(outs)
