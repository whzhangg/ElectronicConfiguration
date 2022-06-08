import matplotlib.pyplot as plt
import numpy as np
from acs_sty import get_parameter, set1_cycler, tableau_cycler

resultfile = "predicted.all.txt"
calculated_result = "calculated.txt"

def plot_prediction_large_cell(filename = "figure5.pdf", broken = True):
    
    with open(resultfile, "r") as f:
        all_lines = f.readlines()
    predicted_data = np.array([ float(aline.split()[-1]) for aline in all_lines ])

    with open(calculated_result, "r") as f:
        all_lines = f.readlines()
    calculated_data = np.array([ float(aline.split()[-1]) for aline in all_lines ])

    with plt.rc_context(get_parameter(width = "single", n = 1, color_cycler = set1_cycler)):
    
        fig = plt.figure()
        if broken:
            
            color_tabluau = [ [c['color']] for c in set1_cycler]

            axs1, axs2 = fig.subplots(2, 1, sharex = True)
            fig.subplots_adjust(hspace=0.05)
            n, bins, patches = axs1.hist(predicted_data, bins = 50, color = color_tabluau[0], density = False, alpha = 0.8, label = "Predicted (220)")
            axs1.hist(calculated_data, bins = bins, color = color_tabluau[1], density = False, alpha = 0.8, label = "Calculated (50)")
            axs1.set_ylim(10, 100)
            n, bins, patches = axs2.hist(predicted_data, bins = 50, color = color_tabluau[0], density = False, alpha = 0.8, label = "Predicted (220)")
            axs2.hist(calculated_data, bins = bins, color = color_tabluau[1], density = False, alpha = 0.8, label = "Calculated (50)")
            axs2.set_ylim(0, 9.9)
            axs1.spines["bottom"].set_visible(False)
            axs2.spines["top"].set_visible(False)

            axs1.xaxis.tick_top()
            axs1.tick_params(labeltop=False)  # don't put tick labels at the top
            axs2.xaxis.tick_bottom()

            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                        linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            axs1.plot([0, 1], [0, 0], transform=axs1.transAxes, **kwargs)
            axs2.plot([0, 1], [1, 1], transform=axs2.transAxes, **kwargs)

            axs1.legend()
            axs2.set_xlim(-0.5, 5.0)
            axs2.set_xlabel("Unpaired electron numbers")
            fig.text(0.045, 0.5, 'Compound numbers', ha='center', va='center', rotation='vertical')
            # https://stackoverflow.com/questions/6963035/pyplot-common-axes-labels-for-subplots
            
        else:
            axs = fig.subplots()

            n, bins, patches = axs.hist(predicted_data, bins = 50, density = False, alpha = 0.8, label = "Predicted (220)")
            axs.hist(calculated_data, bins = bins, density = False, alpha = 0.8, label = "Calculated (50)")
            
            axs.set_ylim(0, 100)
            axs.set_ylabel("Compound numbers")
            axs.legend()

            axs.set_xlabel("Unpaired electron numbers")
            axs.set_xlim(-0.5, 5.0)
        fig.savefig(filename)


plot_prediction_large_cell()
    