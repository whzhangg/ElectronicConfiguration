import matplotlib.pyplot as plt
import numpy as np
from acs_sty import get_parameter, set1_cycler, tableau_cycler
from mldos.models.model_tools import get_mae_error

datafiles = errorfiles = {
        "All" : "occ_all_rcut=5.0_nele=65_single_smoothed_N2007.h5.errorlog.json",
        "Bonded" : "occ_bonded_rcut=5.0_nele=65_single_smoothed_N2007.h5.errorlog.json",
        "First shell" : "occ_firstshell_rcut=5.0_nele=65_single_smoothed_N2007.h5.errorlog.json",
    }

def plot_10fold_errors(filename = "figure3.pdf"):

    results = {}
    for title, fn in datafiles.items():
        results[title] = get_mae_error(fn, ".")

    with plt.rc_context(get_parameter(width = "single", n = 2, color_cycler = set1_cycler)):

        fig = plt.figure()
        axes = fig.subplots(2,1, sharex = False)

        fig.subplots_adjust(bottom = 0.1, top = 0.95)
        axes[0].text(0.03,0.96, "a)", size = 10, ha = "left", va = "top", transform = axes[0].transAxes)
        axes[1].text(0.03,0.96, "b)", size = 10, ha = "left", va = "top", transform = axes[1].transAxes)
        axes[0].text(0.03,0.03, "Training", ha = "left", va = "bottom", transform = axes[0].transAxes)
        axes[1].text(0.03,0.03, "Testing", ha = "left", va = "bottom", transform = axes[1].transAxes)

            
        axes[0].set_ylabel("Mean absolute error")
        axes[1].set_ylabel("Mean absolute error")
        axes[1].set_xlabel("Number of epoches")
        axes[1].set_xlim(40, 210)
        axes[1].set_ylim(0.21, 0.42)
        axes[0].set_xlim(40, 210)
        axes[0].set_ylim(0.05, 0.35)
        axes[0].yaxis.set_major_locator(plt.MultipleLocator(0.05))
        axes[1].yaxis.set_major_locator(plt.MultipleLocator(0.05))
        axes[0].xaxis.set_major_locator(plt.FixedLocator([50,100,150,200]))
        axes[1].xaxis.set_major_locator(plt.FixedLocator([50,100,150,200]))

        for title, result in results.items():
            x = [ float(s) for s,e in result.items() ]
            y = [ e["train"][0] for s,e in result.items() ]
            yerr = np.array([ e["train"][1] for s,e in result.items() ])
            
            axes[0].errorbar(x, y, yerr = yerr, fmt='-o', label = title, elinewidth=0.5, capsize=2)
            y = [ e["test"][0] for s,e in result.items() ]
            yerr = np.array([ e["test"][1] for s,e in result.items() ])
            
            axes[1].errorbar(x, y, yerr = yerr, fmt='-o', label = title, elinewidth=0.8, capsize=2)
        
            axes[0].legend()

        fig.savefig(filename)

if __name__ == "__main__":
    plot_10fold_errors()