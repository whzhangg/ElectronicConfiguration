from mldos.models.model_tools import get_mae_error

datasetfile = "compare_network/N2007_bonded_nodropout.errorlog.json"


print("-----------------------------")
get_mae_error("compare_network/N2007_bonded_dropout_0.1_0.1.errorlog.json", print_result = True)

print("-----------------------------")
get_mae_error("compare_network/N2007_bonded_dropout.errorlog.json", print_result = True)

print("-----------------------------")
get_mae_error("compare_network/N2007_bonded_nodropout.errorlog.json", print_result = True)