from torch import nn

from .nn_components import *
from mldos.parameters import symbol_one_hot_dict
from mldos.data.milad_utility import INVARIANTS

class Network(nn.Module):
    """ a network that combines the components
    feature_extraction
    linear_layers
    output_layers
    """

    def __init__(self, 
        normalization: nn.Module,
        feature_extraction: nn.Module, 
        linear_layers: nn.Module,
        output_layers: nn.Module,
        name: str = "Model",
        dropout: tuple = (0.0,0.0)
    ):
        super().__init__()
        # normalization -> convolution -> linearlayers -> outputlayers


        self.normalization = normalization
        self.convolution = feature_extraction
        self.linearlayer = linear_layers
        self.outputlayer = output_layers
        self._name = name

        assert self.convolution.output_shape == self.linearlayer.input_shape

        assert self.linearlayer.output_shape == self.outputlayer.input_shape

        assert len(dropout) == 2
        self.input_dropout = nn.Dropout(p = dropout[0]) # suggested : 0.8
        self.middle_dropout = nn.Dropout(p = dropout[1]) 

    @property
    def name(self):
        return self._name

    def forward(self, x):
        out = self.normalization(x)
        out = self.input_dropout(out)
        out = self.convolution(out)
        out = self.middle_dropout(out)

        out = self.linearlayer(out)
        out = self.outputlayer(out)
        return out


def get_network_selfinteraction():
    input_feature_shape = (len(INVARIANTS), len(symbol_one_hot_dict))

    extraction = FeatureSelfInteraction(input_feature_shape, (40, len(symbol_one_hot_dict)), weight_sharing=True)
    linear = MultiLinear(len(symbol_one_hot_dict) * 40, 200, (500,))
    output = SplitOutput(200, 2, (100,))    

    return Network(linearx(3.16), extraction, linear, output, dropout=(0.0, 0.0), name = "self-interaction")

def get_network_simpleconvolution():
    input_feature_shape = (len(INVARIANTS), len(symbol_one_hot_dict))

    extraction = SimpleConvolution(input_feature_shape, output_shape=(60,40), weight_sharing = True)
    linear = MultiLinear(2400, 200, (500,))
    output = SplitOutput(200, 2, (100,))    

    return Network(linearx(3.16), extraction, linear, output, dropout=(0.1, 0.0), name = "simple-conv")