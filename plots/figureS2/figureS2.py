import matplotlib.pyplot as plt
import numpy as np
from acs_sty import get_parameter, set1_cycler, tableau_cycler

def plot_distance_subset_supplementary(filename = "figureS2.pdf"):
    subset = ["TM + Main Group", "TM + Main Group + Simple Metal", "All"]
    label = {"TM + Main Group": "TM+MG", "TM + Main Group + Simple Metal": "+ $s$-block metal", "All": "+ heavier $d$-block"}

    from mldos.statistic import training_statistics, info_filepath

    with plt.rc_context(get_parameter(width = "double", n = 1, color_cycler = tableau_cycler)):
        fig = plt.figure()
        axes = fig.subplots(1,3)
        fig.subplots_adjust(left = 0.075, right = 0.95)
        axes[0].text(0.03,0.96, "a)", size = 10, ha = "left", va = "top", transform = axes[0].transAxes)
        axes[1].text(0.03,0.96, "b)", size = 10, ha = "left", va = "top", transform = axes[1].transAxes)
        axes[2].text(0.03,0.96, "c)", size = 10, ha = "left", va = "top", transform = axes[2].transAxes)

        axes[0].set_ylabel("Mean absolute error")
        for i, s in enumerate(subset):
            error_dist = training_statistics.estmiate_error_by_distance(info_filepath, given_element_set= s)
            
            key = list(error_dist.keys())
            key.sort(key = float)
            axes[i].set_xlim(2.2, 7.0)
            axes[i].set_ylim(0, 1.75)
            axes[i].set_xlabel("Metal-metal distance ($\mathrm{\AA}$)")
            
            index = [ float(k) for k in key ]
            bar_x = index
            bar_y = np.array([ error_dist[tm]["freq"] for tm in key ], dtype= float)
            err_y = np.array([ error_dist[tm]["error"]["mae"] for tm in key ], dtype= float)
            bar_y /= np.sum(bar_y)
            
            axes[i].plot(bar_x, err_y, "o-", label = label[s])
            axs2_2 = axes[i].twinx()
            axs2_2.set_ylim(0, 0.12)
            axs2_2.yaxis.set_major_locator(plt.NullLocator())
            axs2_2.yaxis.set_major_formatter(plt.NullFormatter())
            axs2_2.bar(bar_x, bar_y, alpha = 0.6, width = 0.1, color = "gray")
            axs2_2.set_ylim(0, 0.11)
            axes[i].legend(loc = "upper right")
        
        fig.savefig(filename)

if __name__ == "__main__":
    plot_distance_subset_supplementary()