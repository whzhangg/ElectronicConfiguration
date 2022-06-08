import matplotlib.pyplot as plt
import numpy as np
from acs_sty import get_parameter, set1_cycler, tableau_cycler
from mldos.models.model_tools import get_mae_error
from mldos.predict_ensemble import ensemble_partitioned_predict
from mldos.statistic import training_statistics, info_filepath

def plot_learning_and_analysize_error(filename = "figure4.pdf"):
    
    error_tm = training_statistics.estmiate_error_by_key(info_filepath, "T.M.")
    error_shape = training_statistics.estmiate_error_by_key(info_filepath, "shape")
    error_dist = training_statistics.estmiate_error_by_distance(info_filepath)

    result = ensemble_partitioned_predict("set")

    result_tm = ensemble_partitioned_predict("tm")

    width = 0.05
    bins = np.arange(-0.5, 5.5, width)
    distribution = {}
    for k, value in result_tm.items():
        calc, edge = np.histogram(value["calc"][:, 0] - value["calc"][:,1], bins, density = True)
        pred, edge = np.histogram(value["pred"][:, 0] - value["pred"][:,1], bins, density = True)
        maximum = max( max(calc),max(pred) )
        calc /= maximum
        pred /= maximum
        distribution[k] = {"x": bins, "calc":calc, "pred":pred}

    with plt.rc_context(get_parameter(width = "double", n = 2.5, color_cycler = tableau_cycler)):
        w, h = plt.rcParams["figure.figsize"]
        ratio = w / h
        #h = w * 2
        #plt.rcParams["figure.figsize"] = (w,h)

        fig = plt.figure()
        axs = []
        axs.append(fig.add_axes([0.07, 0.461 , 0.474 / ratio , 0.474]))
        axs.append(fig.add_axes([0.5,  0.725 , 0.45 , 0.21 ]))
        axs.append(fig.add_axes([0.5,  0.461 , 0.45 , 0.21 ]))
        axs[0].set_xlim(0, 5.1)
        axs[0].set_ylim(0, 5.1)
        axs[0].set_ylabel("DFT-calculated electron occupation")
        axs[0].set_xlabel("Predicted electron occupation")
        axs[0].plot(np.arange(10), np.arange(10), linestyle = ":", color = "grey")
        axs[0].fill_between(np.arange(10), np.arange(10) - 0.3, np.arange(10) + 0.3, color = "grey", alpha = 0.4)
        
        color_tabluau = [ [c['color']] for c in tableau_cycler]
        color_set1 = [ [c['color']] for c in set1_cycler]
        
        counter = 0
        labels = {"TM + Main Group": "TM+MG", "TM + Main Group + Simple Metal": "+ $s$-block metal", "All": "+ heavier $d$-block"}
        for k, value in result.items():

            px_up = value["calc"][:,0].flatten()
            px_dw = value["calc"][:,1].flatten()
            py_up = value["pred"][:,0].flatten()
            py_dw = value["pred"][:,1].flatten()
            axs[0].scatter(px_up, py_up, marker = 'o', s = 2, c = color_tabluau[counter], label = labels[k])
            axs[0].scatter(px_dw, py_dw, marker = 'o', s = 2, c = color_tabluau[counter], )
            counter += 1
            
        axs[0].legend()
        first_row = "Sc Ti V Cr Mn".split()
        second_row = "Fe Co Ni Cu Zn".split()
        tick_x = np.arange(5) + 0.5
        axs[1].set_xlim(0, 5)
        axs[2].set_xlim(0, 5)
        axs[1].set_ylim(min(bins), max(bins))
        axs[2].set_ylim(min(bins), max(bins))
        axs[1].xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axs[1].xaxis.set_major_formatter(plt.FixedFormatter(first_row))
        axs[2].xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(second_row))
        axs[1].set_ylabel("Unpaired electrons")
        axs[2].set_ylabel("Unpaired electrons")
        axs[2].set_xlabel("Transition metal elements")

        y = bins[0:-1]
        for i, ele in enumerate(first_row):
            x0 = np.zeros_like(y)
            x0 += i + 0.5
            x_cal = x0 - distribution[ele]["calc"] * 0.5
            x_pre = x0 + distribution[ele]["pred"] * 0.5
            axs[1].fill_betweenx(y, x_cal, x0, color = color_set1[0])
            axs[1].fill_betweenx(y, x0, x_pre, color = color_set1[1])


        for i, ele in enumerate(second_row):
            x0 = np.zeros_like(y)
            x0 += i + 0.5
            x_cal = x0 - distribution[ele]["calc"] * 0.5
            x_pre = x0 + distribution[ele]["pred"] * 0.5
            axs[2].fill_betweenx(y, x_cal, x0, color = color_set1[0])
            axs[2].fill_betweenx(y, x0, x_pre, color = color_set1[1])

        
        axs[0].text(0.03,0.975, "a)", size = 10, ha = "left", va = "top", transform = axs[0].transAxes)
        axs[1].text(0.02,0.96, "b)", size = 10, ha = "left", va = "top", transform = axs[1].transAxes)
        axs[2].text(0.02,0.96, "c)", size = 10, ha = "left", va = "top", transform = axs[2].transAxes)

        axs1 = fig.add_axes([0.07,  0.06, 0.3, 0.316])
        axs2 = fig.add_axes([0.395, 0.06, 0.15, 0.316])
        axs3 = fig.add_axes([0.57,  0.06, 0.38, 0.316])
        axs1.set_ylim(0, 1)
        axs2.set_ylim(0, 1)
        axs3.set_ylim(0, 1)

        
        axs1.text(0.02,0.96, "d)", size = 10, ha = "left", va = "top", transform = axs1.transAxes)
        axs2.text(0.04,0.96, "e)", size = 10, ha = "left", va = "top", transform = axs2.transAxes)
        axs3.text(0.02,0.96, "f)", size = 10, ha = "left", va = "top", transform = axs3.transAxes)
        
        axs2.yaxis.set_major_formatter(plt.NullFormatter())
        axs3.yaxis.set_major_formatter(plt.NullFormatter())
        axs1.set_ylabel("Prediction error (MAE)")

        axs1.set_xlabel("Transition metal")
        axs2.set_xlabel("Coord. shape")
        axs3.set_xlabel("Metal-metal distance ($\mathrm{\AA}$)")

        axs1.set_xlim(0, len(error_tm.keys()))
        axs2.set_xlim(0, len(error_shape.keys()))
        axs3.set_xlim(2.2, 7.0)

        
        key = "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
        index = np.arange(len(key)) + 0.5
        bar_x = index
        bar_y = np.array([ error_tm[tm]["freq"] for tm in key ], dtype= float)
        err_y = np.array([ error_tm[tm]["error"]["mae"] for tm in key ], dtype= float)
        bar_y /= np.sum(bar_y)
        axs1.xaxis.set_major_locator(plt.FixedLocator(bar_x))
        axs1.xaxis.set_major_formatter(plt.FixedFormatter(key))

        axs1.plot(bar_x, err_y, "o-", label = "error")
        axs1_2 = axs1.twinx()
        axs1_2.set_ylim(0, 0.3)
        axs1_2.yaxis.set_major_locator(plt.NullLocator())
        axs1_2.yaxis.set_major_formatter(plt.NullFormatter())
        axs1_2.bar(bar_x, bar_y, alpha = 0.6, width = 0.6, color = "gray")
        
        key = "Octa Tetra Other".split()
        index = np.arange(len(key)) + 0.5
        bar_x = index
        bar_y = np.array([ error_shape[tm]["freq"] for tm in key ], dtype= float)
        err_y = np.array([ error_shape[tm]["error"]["mae"] for tm in key ], dtype= float)
        bar_y /= np.sum(bar_y)
        axs2.xaxis.set_major_locator(plt.FixedLocator(bar_x))
        axs2.xaxis.set_major_formatter(plt.FixedFormatter(key))
        
        axs2.plot(bar_x, err_y, "o-")
        axs2_2 = axs2.twinx()
        axs2_2.set_ylim(0, 0.6)
        axs2_2.yaxis.set_major_locator(plt.NullLocator())
        axs2_2.yaxis.set_major_formatter(plt.NullFormatter())
        axs2_2.bar(bar_x, bar_y, alpha = 0.6, width = 0.4, color = "gray")

        
        key = list(error_dist.keys())
        key.sort(key = float)
        
        index = [ float(k) for k in key ]
        bar_x = index
        bar_y = np.array([ error_dist[tm]["freq"] for tm in key ], dtype= float)
        err_y = np.array([ error_dist[tm]["error"]["mae"] for tm in key ], dtype= float)
        bar_y /= np.sum(bar_y)
        
        axs3.plot(bar_x, err_y, "o-")
        axs3_2 = axs3.twinx()
        axs3_2.set_ylim(0, 0.12)
        axs3_2.yaxis.set_major_locator(plt.NullLocator())
        axs3_2.yaxis.set_major_formatter(plt.NullFormatter())
        axs3_2.bar(bar_x, bar_y, alpha = 0.6, width = 0.1, color = "gray")

        axs1.legend(loc = 'center left')
        fig.savefig(filename)

if __name__ == "__main__":
    plot_learning_and_analysize_error()