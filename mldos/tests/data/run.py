from mldos.data.datasets import make_Geometry_data_with_feature
from mldos.data.criteria import occ_bonded_rcut_5_smoothed_3d_maingroup
from mldos.data.datasets import write_hdf

write_hdf(occ_bonded_rcut_5_smoothed_3d_maingroup, "tmp.h5")