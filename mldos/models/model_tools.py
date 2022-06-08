import os
import torch
from mldos.tools import read_json
import numpy as np
from mldos.parameters import root

current_path = os.path.split(__file__)[0]
modelfolder = os.path.join(current_path, "saved")


def save_model(model, filename = "current_model.pt"):
    filename = os.path.join(modelfolder, filename)
    torch.save(model.state_dict(), filename)


def load_model(model, filename = "current_model.pt"):
    filename = os.path.join(modelfolder, filename)
    model.load_state_dict(torch.load(filename))


def get_mae_error(filename, root_folder = "mldos/models/error_10fold", print_result = False):
    filename = os.path.join(root_folder, filename)
    data = read_json(filename)
    result = {}
    for step, eachstep in data.items():
        all_train_mae = np.array([ d["mae"] for d in eachstep["train"]])
        all_test_mae = np.array([ d["mae"] for d in eachstep["test"]])
        average_train = np.mean(all_train_mae)
        average_test = np.mean(all_test_mae)
        train_variance = np.sqrt(np.var(all_train_mae))
        test_variance = np.sqrt(np.var(all_test_mae))
        result[step] = {"train": (average_train, train_variance),
                        "test": (average_test, test_variance)
                        }
        if print_result:
            print(step)
            print("train: {:>8.3f} +-{:>8.3f}".format(average_train, train_variance))
            print("test : {:>8.3f} +-{:>8.3f}".format(average_test, test_variance))

    return result