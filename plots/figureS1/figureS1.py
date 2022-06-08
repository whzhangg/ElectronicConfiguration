import matplotlib.pyplot as plt
import numpy as np
from acs_sty import get_parameter, set1_cycler, tableau_cycler


def plot_analysize_supplementary(filename = "figureS1.pdf"):
    from mldos.statistic import training_statistics, info_filepath
    #error_tm = training_statistics.estmiate_error_by_key(info_file, "T.M.")

    with plt.rc_context(get_parameter(width = "double", n = 4, color_cycler = tableau_cycler)):
        fig = plt.figure()
        fig.subplots_adjust(left = 0.075, right = 0.95, top = 0.97, bottom = 0.05)
        grid = plt.GridSpec(5, 6, hspace=0.2, wspace=0.2)

        metal = "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
        for i, m in enumerate(metal):
            print(m)
            error_shape = training_statistics.estmiate_error_by_key(info_filepath, "shape", given_element=m)
            error_dist = training_statistics.estmiate_error_by_distance(info_filepath, given_element=m)

            row = i//2
            column = 3 * (i % 2)

            axs1 = fig.add_subplot(grid[row,column])
            axs2 = fig.add_subplot(grid[row,column+1:column+3])

            if column == 0:
                axs1.set_ylabel("Mean absolute error")
                axs2.yaxis.set_major_formatter(plt.NullFormatter())
            else:
                axs1.yaxis.set_major_formatter(plt.NullFormatter())
                axs2.yaxis.set_major_formatter(plt.NullFormatter())

            axs1.set_xlabel("Coord. shape")
            axs2.set_xlabel("Metal-metal distance ($\mathrm{\AA}$)")
            
            axs1.set_ylim(0, 1)
            axs2.set_ylim(0, 1)
            axs1.text(0.95,0.96, m, size = 10, ha = "right", va = "top", transform = axs1.transAxes)
            axs2.text(0.95,0.96, m, size = 10, ha = "right", va = "top", transform = axs2.transAxes)
            
            axs1.set_xlim(0, len(error_shape.keys()))
            axs2.set_xlim(2.2, 7.0)

            key = "Octa Tetra Other".split()
            index = np.arange(len(key)) + 0.5
            bar_x = index
            bar_y = np.array([ error_shape[tm]["freq"] for tm in key ], dtype= float)
            err_y = np.array([ error_shape[tm]["error"]["mae"] for tm in key ], dtype= float)
            bar_y /= np.sum(bar_y)
            axs1.xaxis.set_major_locator(plt.FixedLocator(bar_x))
            axs1.xaxis.set_major_formatter(plt.FixedFormatter(key))
            
            axs1.plot(bar_x, err_y, "o-")
            axs1_2 = axs1.twinx()
            axs1_2.set_ylim(0, 0.6)
            axs1_2.yaxis.set_major_locator(plt.NullLocator())
            axs1_2.yaxis.set_major_formatter(plt.NullFormatter())
            axs1_2.bar(bar_x, bar_y, alpha = 0.6, width = 0.4, color = "gray")

            
            key = list(error_dist.keys())
            key.sort(key = float)
            
            index = [ float(k) for k in key ]
            bar_x = index
            bar_y = np.array([ error_dist[tm]["freq"] for tm in key ], dtype= float)
            err_y = np.array([ error_dist[tm]["error"]["mae"] for tm in key ], dtype= float)
            bar_y /= np.sum(bar_y)
            
            axs2.plot(bar_x, err_y, "o-")
            axs2_2 = axs2.twinx()
            axs2_2.set_ylim(0, 0.12)
            axs2_2.yaxis.set_major_locator(plt.NullLocator())
            axs2_2.yaxis.set_major_formatter(plt.NullFormatter())
            axs2_2.bar(bar_x, bar_y, alpha = 0.6, width = 0.1, color = "gray")
            
        fig.savefig(filename)

if __name__ == "__main__":
    plot_analysize_supplementary()