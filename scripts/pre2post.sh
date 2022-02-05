#!/bin/bash
#
# This code performs a non-diffeomorphic registration of the pre-operative image to the post-operative image. It also 
# provides atlas segmentations based on the pre-operative image.
#
# Usage: pre2post.sh subID preop.nii.gz postop.nii.gz output_dir
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
#
# 5/17/2019 - created
# 6/5/2019 - seperated into atlas2NickOasis.sh and atlas2native.sh
# 5/17/2019 - created
# 6/5/2019 - seperated into atlas2NickOasis.sh and atlas2native.sh
# 9/19/2019 - converted to run on ANDI normal dataset, renamed atlas2ADNI_normal.sh
# 10/3/2019 - use the origin_MNI image to 


# arguements
patient_id=${1}
preop_file=${2}
postop_file=${3}
output_dir=${4}

# get fileparts
preop_onlypath="$(dirname $preop_file)"
preop_onlyfile="$(basename $preop_file)"
postop_onlypath="$(dirname $postop_file)"
postop_onlyfile="$(basename $postop_file)"

####### Affine registstration of preop to postop image ######## 

# Registration of preop to postop
antsRegistration \
--dimensionality 3 \
--float 0 \
--output ${output_dir}/pre2post_ \
--interpolation Linear \
--use-histogram-matching 0 \
--initial-moving-transform [${postop_file},${preop_file},1] \
--transform Rigid[0.1] \
--metric MI[${postop_file},${preop_file},1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform Affine[0.1] \
--metric MI[${postop_file},${preop_file},1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox

# transform pre-operative T1 to post-operative T1 space
antsApplyTransforms \
-d 3 \
-i ${preop_file} \
-o ${output_dir}/pre2post_${preop_onlyfile} \
-t ${output_dir}/pre2post_0GenericAffine.mat \
-r ${postop_file} \
-n Linear
