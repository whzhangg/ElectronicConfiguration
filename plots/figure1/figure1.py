import matplotlib.pyplot as plt
import numpy as np
from acs_sty import get_parameter, set1_cycler, tableau_cycler
from mldos.tools import read_generic_to_cpt

c, p, t = read_generic_to_cpt("BaCoS2.cif")


def plot_periodic_table(filename = "periodic_table.pdf"):
    import matplotlib.pyplot as plt
    from mldos.parameters import TM_3D, MAIN_GROUP, Simple_METAL, symbol_one_hot_dict
    from ase.data import atomic_numbers
    from pymatgen.core.periodic_table import Element

    default = 5

    elemental_data = { key:default for key in atomic_numbers.keys() }
    for key in symbol_one_hot_dict.keys():
        elemental_data[key] = 4
    for key in MAIN_GROUP:
        elemental_data[key] = 3
    for key in Simple_METAL:
        elemental_data[key] = 2
    for key in TM_3D:
        elemental_data[key] = 1

    max_val = max(elemental_data.values())
    min_val = min(elemental_data.values())
    value_format=None
    cmap="YlGnBu"

    min_row = 0
    max_row = 7

    min_col = 0
    max_col = 18

    value_table = np.empty((max_row - min_row, max_col - min_col)) * np.nan
    blank_value = min_val - 0.01

    for el in Element:
        if el.row > max_row or el.row < min_row:
            continue
        if el.group > max_col or el.group < min_col:
            continue

        value = elemental_data.get(el.symbol, blank_value)
        value_table[el.row - 1 - min_row, el.group - 1 - min_col] = value

    fig = plt.figure()
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    fig.set_size_inches(2.8, 1.6)

    # We set nan type values to masked values (ie blank spaces)
    data_mask = np.ma.masked_invalid(value_table.tolist())
    heatmap = ax.pcolor(
        data_mask,
        cmap=cmap,
        edgecolors="w",
        linewidths=1,
        vmin=min_val - 1,
        vmax=max_val + 3,
    )

    # Refine and make the table look nice
    ax.axis("off")
    ax.invert_yaxis()

    # Label each block with corresponding element and value
    for i, row in enumerate(value_table):
        for j, el in enumerate(row):
            if not np.isnan(el):
                try:
                    symbol = Element.from_row_and_group(i + 1 + min_row, j + 1 + min_col).symbol
                
                    plt.text(
                        j + 0.5,
                        i + 0.5,
                        symbol,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=5
                    )
                except:
                    pass
    plt.savefig(filename)


def structure_representation(filename = "figure1.pdf"):
    from mldos.data.plot_compounds import SimplePlotter
    from mldos.data.criteria import occ_firstshell_rcut_5_smoothed_all, occ_bonded_rcut_5_smoothed_all, occ_all_rcut_5_smoothed_all

    x = np.arange(117)
    y = np.sin(x)

    color_tabluau = [ c['color'] for c in tableau_cycler]
    color_element = {"Co": color_tabluau[0],
                     "S" : color_tabluau[1],
                     "Ba": color_tabluau[2]}

    with plt.rc_context(get_parameter(width = "double", n = 2, color_cycler = set1_cycler)):
        fig = plt.figure()
        axs = fig.add_axes([0.475, 0.05,0.22,0.22])
        axs.set_xlabel("Fingerprint index")
        axs.set_ylabel("Fingerprint value")
        axs.yaxis.set_major_formatter(plt.NullFormatter())
        axs.xaxis.set_major_formatter(plt.NullFormatter())

        sp = SimplePlotter(c, p, t, occ_bonded_rcut_5_smoothed_all)
        site = 2
        sp.write_cif(2, "BaCoS2_bonded.cif")
        fp = sp.fingerprint(site)
        counter = 0
        for key in color_element.keys():
            if key in fp.keys():
                val = fp[key]
                x = np.arange(len(val))
                axs.plot(x, val + 1.5 * counter, color = color_element[key], label = key, alpha = 0.75)
                counter += 1
        axs.legend(frameon = True, loc = "upper right")
        
        
        axs = fig.add_axes([0.755, 0.05,0.22,0.22])
        axs.set_xlabel("Fingerprint index")
        axs.set_ylabel("Fingerprint value")
        axs.yaxis.set_major_formatter(plt.NullFormatter())
        axs.xaxis.set_major_formatter(plt.NullFormatter())
        
        sp = SimplePlotter(c, p, t, occ_all_rcut_5_smoothed_all)
        site = 2
        sp.write_cif(2, "BaCoS2_all.cif")
        fp = sp.fingerprint(site)
        counter = 0
        for key in color_element.keys():
            if key in fp.keys():
                val = fp[key]
                x = np.arange(len(val))
                axs.plot(x, val + 1.5 * counter, color = color_element[key], label = key, alpha = 0.75)
                counter += 1
        axs.legend(frameon = True, loc = "upper right")

        axs = fig.add_axes([0.755, 0.55,0.22,0.22])
        axs.set_xlabel("Fingerprint index")
        axs.set_ylabel("Fingerprint value")
        axs.yaxis.set_major_formatter(plt.NullFormatter())
        axs.xaxis.set_major_formatter(plt.NullFormatter())
        
        sp = SimplePlotter(c, p, t, occ_firstshell_rcut_5_smoothed_all)
        site = 2
        sp.write_cif(2, "BaCoS2_firstshell.cif")
        fp = sp.fingerprint(site)
        counter = 0
        for key in color_element.keys():
            if key in fp.keys():
                val = fp[key]
                x = np.arange(len(val))
                axs.plot(x, val + 1.5 * counter, color = color_element[key], label = key, alpha = 0.75)
                counter += 1
        axs.legend(frameon = True, loc = "upper right")
    
    
        fig.savefig(filename)



if __name__ == "__main__":
    plot_periodic_table()
    structure_representation()