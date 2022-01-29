# This code uses the pre-operative image to generate an atlas segmentation of the brain.

# Usage: python3 register_atlas_to_preop.py pat_id preop.nii output_dir
# Example: python3 register_atlas_to_preop.py patient1 masks/patient1_mask.nii results/patient1

# 6/8/20 - created

import ants
import numpy as np
import scipy
import sklearn
import pandas as pd
import os
import sys
import antspynet

pat_id = sys.argv[1]
PRE2POST = sys.argv[2]
OUT_DIR = sys.argv[3]

preop = ants.image_read(PRE2POST)
DKT = antspynet.utilities.desikan_killiany_tourville_labeling(preop, do_preprocessing=True, return_probability_images=False, do_lobar_parcellation=False, antsxnet_cache_directory=None, verbose=False)
ants.image_write(DKT, os.path.join(OUT_DIR, f'{pat_id}_DKT_DL.nii.gz'))
