# =========================================================================================
# 
# this script defines the assicoated atomic information, 
# but I should not use them because they introduce bias from the arbitrarily assigned number
#
# =========================================================================================


from ase.data import chemical_symbols, covalent_radii
import re

# covalent radius, used from ase database -------------------------------------------------
covalent_radius = {}
for i in range(len(covalent_radii)):
    covalent_radius[chemical_symbols[i]] = covalent_radii[i]


# self-defined number of valence electrons ------------------------------------------------
defined_valence = """
   H  1  He  2
  Li  1  Be  2   B  3   C  4   N  5   O  6   F  7  Ne  8
  Na  1  Mg  2  Al  3  Si  4   P  5   S  6  Cl  7  Ar  8
  K   1  Ca  2  Sc 11  Ti 12   V 13  Cr 14  Mn 15  Fe 16  Co 17  Ni 18  Cu 19  Zn 20  Ga  3  Ge  4  As  5  Se  6  Br  7  Kr  8
  Rb  1  Sr  2   Y 11  Zr 12  Nb 13  Mo 14  Tc 15  Ru 16  Rh 17  Pd 18  Ag 19  Cd 20  In  3  Sn  4  Sb  5  Te  6   I  7  Xe  8
  Cs  1  Ba  2  Lu 11  Hf 12  Ta 13   W 14  Re 15  Os 16  Ir 17  Pt 18  Au 19  Hg 20  Tl  3  Pb  4  Bi  5  Po  6  At  7  Rn  8
                La 10  Ce 10  Pr 10  Nd 10  Pm 10  Sm 10  Eu 10  Gd 10  Tb 10  Dy 10  Ho 10  Er 10  Tm 10  Yb 10
  Fr  1  Ra  2  Lr 11  Rf 12  Db 13  Sg 14  Bh 15  Hs 16  Mt 17  Ds 18  Rg 19  Cn 20  Nh  3  Fl  4  Mc  5  Lv  6  Ts  7  Og  8
                Ac 10  Th 10  Pa 10   U 10  Np 10  Pu 10  Am 10  Cm 10  Bk 10  Cf 10  Es 10  Fm 10  Md 10  No 10  
"""
parts = re.findall(r'\S+', defined_valence)
valence_count = {}
for i in range(0, len(parts), 2):
    valence_count[parts[i]] = parts[i+1]
