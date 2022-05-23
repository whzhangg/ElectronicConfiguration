import numpy as np
import math
import torch
from collections import namedtuple
from ase.data import chemical_symbols

from mldos.DFT import estep_dos
from mldos.parameters import r_seed_testsamples, TM_3D
from mldos.data import OneHotDescription, Data, get_features
from mldos.tools import read_generic_to_cpt
from mldos.network import (
    DosDataset, SimpleConvolution,
    ConvNet, EarlyStop, NMFWrapper,
    TrainingBase, torch_device, torch_dtype,
    encode_feature
)
from mldos.models import save_model, load_model

# rng_nmf = np.random.RandomState(0)
# this is problematic because rng_nmf is used multiplt times

PredictDosResult = namedtuple('PredictionResult', ['formula', 'energy', 'pred_up', 'pred_dw'])
CompareDosResult = namedtuple('PredictionResult', ['formula', 'energy', 'pred_up', 'pred_dw', 
                                                    'nmf_up', 'nmf_dw', 'calc_up', 'calc_dw'])

default_ncomponent = 30
model_name = "current_model_dos.pt"

def make_plots_dos(ys: list, filename: str):
    import matplotlib.pyplot as plt
    from mldos.tools import format_formula

    total_number = len(ys)
    nrow = math.floor(np.sqrt(total_number))
    ncolumn = total_number // nrow
    nrow, ncolumn = sorted((nrow, ncolumn))
    size = (nrow * 4, ncolumn * 2.5)  # inch

    fig = plt.figure()
    axs = fig.subplots(ncolumn, nrow, sharex = True)

    fig.set_size_inches(size[0], size[1])
    #fig.tight_layout()
    #fig.suptitle("test result")

    for (ic, ir), y in zip([ (c,r) for c in range(ncolumn) for r in range(nrow) ], ys):
        axis = axs[ic, ir]
        if ir == 0:
            axis.set_ylabel('DOS')
        if ic == ncolumn - 1:
            axis.set_xlabel('Energy (eV)')

        axis.plot(y.energy,  y.calc_up, color = 'black')
        axis.plot(y.energy, -y.calc_dw, color = 'black')
        axis.plot(y.energy,  y.nmf_up, color = 'blue')
        axis.plot(y.energy, -y.nmf_dw, color = 'blue')
        axis.plot(y.energy,  y.pred_up, color = 'red')
        axis.plot(y.energy, -y.pred_dw, color = 'red')

        if y.formula != "":
            f = format_formula(str(y.formula))
            axis.legend(labels = ["${:s}$".format(f)])

    plt.savefig(filename)


def prepare(ncomp):
    # build network, also return the dataset
    
    alldata = DosDataset().get_all_data()
    featureshape = tuple(alldata.featureshape)

    conv = SimpleConvolution(featureshape, output_shape=(60,40), weight_sharing = True )
    net = ConvNet(featureshape, ncomp * 2, convolution = conv, 
                 linearlayer_connection = (400,100))
    stop = EarlyStop(max_epochs=100, early_stop=True)

    trainer = TrainingBase(net, stop_function = stop)
    wrapper = NMFWrapper(trainer, ncomponents = ncomp)

    return alldata, wrapper


def training_dos(ncomp = default_ncomponent):
    print("Using random seed {:d}".format(r_seed_testsamples))
    alldata, wrapper = prepare(ncomp=ncomp)
    
    transformed = wrapper.train_nmf(alldata)
    _, training = transformed.split(0.02, seed = r_seed_testsamples)
    wrapper.train(training)
    save_model(wrapper.trainer.network, filename=model_name)


def predict_test(ncomp = default_ncomponent):
    print("Using random seed {:d}".format(r_seed_testsamples))
    alldata, wrapper = prepare(ncomp=ncomp)

    load_model(wrapper.trainer.network, filename=model_name)
    transformed = wrapper.train_nmf(alldata)

    test, _ = alldata.split(0.02, seed = r_seed_testsamples)
    test_transformed, _ = transformed.split(0.02, seed = r_seed_testsamples)
    
    Pred_Result = []
    for (x, y), (xt, yt), formula in zip(test, test_transformed, test.names):
        # y is nspin, ndos : the calculated dos
        # yt is nspin * ncomp : the flattened transformed y
        # x and xt are features which should be the same
        ntics = len(y[0])
        energy = np.arange( -ntics / 2 * estep_dos, ntics / 2 * estep_dos, estep_dos )

        coeff = wrapper.predict(xt)
        pred = wrapper.coefficients_to_dos(coeff)
        ytt = wrapper.coefficients_to_dos(yt)
        
        res = CompareDosResult(formula, energy, pred[0], pred[1], ytt[0], ytt[1], y[0], y[1])
        Pred_Result.append(res)

    make_plots_dos(Pred_Result, "pred.pdf")
    return Pred_Result


def predict_new(structfile, index = None, ncomp = default_ncomponent):
    alldata, wrapper = prepare(ncomp=ncomp)
    load_model(wrapper.trainer.network, filename=model_name)
    wrapper.train_nmf(alldata)


    formula, feature = get_features(structfile, index = index)
    
    features = encode_feature( feature )
    X = torch.from_numpy(features.T).to(torch_device, torch_dtype)

    coeff = wrapper.predict(X)
    y = wrapper.coefficients_to_dos(coeff)
    ntics = len(y[0])
    energy = np.arange( -ntics / 2 * estep_dos, ntics / 2 * estep_dos, estep_dos )

    result = PredictDosResult(formula, energy, y[0], y[1])
    return result

    
if __name__ == "__main__":
    #DosDataset.make_h5py_data()
    training_dos()
    predict_test()