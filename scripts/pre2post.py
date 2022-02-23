# This code performs a non-diffeomorphic registration of the pre-operative image to the post-operative image. It also 
# provides atlas segmentations based on the pre-operative image.

# Usage: python3 pre2post.py subID preop.nii.gz postop.nii.gz output_dir
# Example: python3 pre2post.py pati1 pat1/preop.nii.gz pat1/postop.nii.gz results/pat1

# 2/22/22 - created

import ants
import sys
import os

sub_id = sys.argv[1]
preop = sys.argv[2]
postop = sys.argv[3]
output_dir = sys.argv[4]
preop_onlyfile = os.path.basename(preop)
output_filename = os.path.join(output_dir, f'pre2post_{preop_onlyfile}')

# read in images
fixed = ants.image_read(postop)
moving = ants.image_read(preop)

# register preop to postop, with the resection zone masked
mytx = ants.registration(fixed=fixed , moving=moving , type_of_transform = 'Affine' )

# transform preop to postoperative space
mywarped_image = ants.apply_transforms( fixed=fixed, moving=moving, interpolator='linear', transformlist=mytx['fwdtransforms'] )

# write out transformed image
ants.image_write(mywarped_image, output_filename)