# this script should define the configurations that generate the dataset
from mldos.parameters import TM_3D, REDUCED_MAIN_GROUP, Simple_METAL, symbol_one_hot_dict
"""
a dictionary given 
we can allow three way to give the coordinate environment:
1) the previous way of Voronoi bonded components
2) all atoms within a given radius
3) all atoms within the nearest shell r ~ r+d
In all methods, we specify a cutoff radius that will be used to scale the radius

n_neighbors will be removed because it is not very useful, it could be added later
{ 
  "name" : "electron_filling_rcut_5_smoothed",   # name that will be used as filename "electron_filling_rcut_5_smoothed.h5"
  "bonded_only" : False,                         # if true, we only consider the bonded elements
  "nneighbors" : 10,                              # the number of neigbhoring elements
  "R" : 5.0,                                     # radius cutoff
  "smooth" : True,                               # if a smooth function is applied as weight
  "method" : "geometry",                         # geometry, gaussian or spherical moments
  "tag" : "occupation" ,                         # what kind of final properties
  "firstshell_only" : False                      # if we only take first shell
}
"""
tm_maingroup = TM_3D | REDUCED_MAIN_GROUP
tm_maingroup_metal = TM_3D | REDUCED_MAIN_GROUP | Simple_METAL
all = set(symbol_one_hot_dict.keys())

occ_bonded_rcut_5_smoothed_3d_maingroup = { 
  "name" : "occ_bonded_rcut_5_smoothed",
  "firstshell_only" : False,
  "bonded_only" : True, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup,
  "single_tm" : True
}

occ_firstshell_rcut_5_smoothed_3d_maingroup = { 
  "name" : "occ_firstshell_rcut_5_smoothed",
  "firstshell_only" : True,
  "bonded_only" : False, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup,
  "single_tm" : True
}

occ_all_rcut_5_smoothed_3d_maingroup = { 
  "name" : "occ_all_rcut_5_smoothed",
  "firstshell_only" : False,
  "bonded_only" : False, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup,
  "single_tm" : True
}

occ_all_rcut_7_smoothed_3d_maingroup = { 
  "name" : "occ_all_rcut_7_smoothed",
  "firstshell_only" : False,
  "bonded_only" : False, 
  "cutoff" : 7.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup,
  "single_tm" : True
}

#=========================================

occ_bonded_rcut_5_smoothed_3d_maingroup_metal = { 
  "name" : "occ_bonded_rcut_5_smoothed",
  "firstshell_only" : False,
  "bonded_only" : True, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup_metal,
  "single_tm" : True
}

occ_firstshell_rcut_5_smoothed_3d_maingroup_metal = { 
  "name" : "occ_firstshell_rcut_5_smoothed",
  "firstshell_only" : True,
  "bonded_only" : False, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup_metal,
  "single_tm" : True
}

occ_all_rcut_5_smoothed_3d_maingroup_metal = { 
  "name" : "occ_all_rcut_5_smoothed",
  "firstshell_only" : False,
  "bonded_only" : False, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup_metal,
  "single_tm" : True
}

occ_all_rcut_7_smoothed_3d_maingroup_metal = { 
  "name" : "occ_all_rcut_7_smoothed",
  "firstshell_only" : False,
  "bonded_only" : False, 
  "cutoff" : 7.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : tm_maingroup_metal,
  "single_tm" : True
}

#=========================================

occ_bonded_rcut_5_smoothed_all = { 
  "name" : "occ_bonded_rcut_5_smoothed",
  "firstshell_only" : False,
  "bonded_only" : True, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : all,
  "single_tm" : True
}

occ_firstshell_rcut_5_smoothed_all = { 
  "name" : "occ_firstshell_rcut_5_smoothed",
  "firstshell_only" : True,
  "bonded_only" : False, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : all,
  "single_tm" : True
}

occ_all_rcut_5_smoothed_all = { 
  "name" : "occ_all_rcut_5_smoothed",
  "firstshell_only" : False,
  "bonded_only" : False, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : all,
  "single_tm" : True
}

occ_all_rcut_7_smoothed_all = { 
  "name" : "occ_all_rcut_7_smoothed",
  "firstshell_only" : False,
  "bonded_only" : False, 
  "cutoff" : 7.0, 
  "smooth" : True,
  "target" : "filling",
  "elements" : all,
  "single_tm" : True
}