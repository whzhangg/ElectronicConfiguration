# =========================================================================================
# Element groups
# =========================================================================================

MAIN_GROUP = set(["B" , "C" , "N" , "O" , "F" ,
              "Al", "Si", "P" , "S" , "Cl",
              "Ga", "Ge", "As", "Se", "Br",
              "In", "Sn", "Sb", "Te", "I" ,
              "Tl", "Pb", "Bi", "Po", "At"])

REDUCED_MAIN_GROUP = set(["Al", "Si", "P" , "S" , "Cl",
              "Ga", "Ge", "As", "Se", "Br",
              "In", "Sn", "Sb", "Te", "I" ,
              "Tl", "Pb", "Bi", "Po", "At"])

Simple_METAL = set("Li Be Na Mg K Ca Rb Sr Cs Ba".split())

TM_3D = set(["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"])
TM_4D = set(["Y","Zr","Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"])
TM_5D = set(["Lu", "Hf", "Ta", "W", "Re", "Os", "Ir","Pt", "Au", "Hg"])

NOTCLUDE = set(["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
            "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
            "Pt", "Au", "Hg"])