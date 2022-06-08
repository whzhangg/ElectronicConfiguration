import matplotlib.pyplot as plt
import numpy as np
from acs_sty import get_parameter, set1_cycler, tableau_cycler

def plot_model_prediction_supplementary(filename = "supplementary_model_prediction.pdf"):
    from mldos.tests.model_problem import ModelProblem

    model = ModelProblem(50)
    alpha = np.array(model.train_and_test("alpha", 50))
    
    model = ModelProblem(50)
    beta = np.array(model.train_and_test("beta", 50))

    with plt.rc_context(get_parameter(width = "double", n = 1.5, color_cycler = tableau_cycler)):
        w, h = plt.rcParams["figure.figsize"]
        ratio = w / h
        #h = w * 2
        #plt.rcParams["figure.figsize"] = (w,h)

        fig = plt.figure()
        axs1 = fig.add_axes([0.12,  0.135 , 0.79 / ratio , 0.79])
        axs2 = fig.add_axes([0.55,  0.135 , 0.79 / ratio , 0.79])

        axs1.set_xlim(0, 45)
        axs1.set_ylim(0, 45)
        axs1.set_ylabel("Predict $\\alpha$ (degree)")
        axs1.set_xlabel("Actual $\\alpha$ (degree)")
        axs1.plot(np.arange(50), np.arange(50), linestyle = ":", color = "grey")
        axs1.fill_between(np.arange(50), np.arange(50) - 1, np.arange(50) + 1, color = "grey", alpha = 0.4)
        axs1.scatter(alpha[:,0], alpha[:,1], marker = 'o', s = 3, label = "Distortion angle")
        axs1.legend(loc = "lower right")

        axs2.set_xlim(0.6, 1.6)
        axs2.set_ylim(0.6, 1.6)
        axs2.set_ylabel("Predict $\\beta$")
        axs2.set_xlabel("Actual $\\beta$")
        axs2.plot(np.arange(3), np.arange(3), linestyle = ":", color = "grey")
        axs2.fill_between(np.arange(3), np.arange(3) - 0.01, np.arange(3) + 0.01, color = "grey", alpha = 0.4)
        axs2.scatter(beta[:,0], beta[:,1], marker = 'o', s = 3, label = "Elongation ratio")
        axs2.legend(loc = "lower right")
        
        axs1.text(0.03,0.975, "a)", size = 10, ha = "left", va = "top", transform = axs1.transAxes)
        axs2.text(0.03,0.975, "b)", size = 10, ha = "left", va = "top", transform = axs2.transAxes)

        fig.savefig(filename)

if __name__ == "__main__":
    plot_model_prediction_supplementary()