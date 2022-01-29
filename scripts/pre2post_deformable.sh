#!/bin/bash
#
# This code performs a non-diffeomorphic registration of the pre-operative image to the post-operative image. It also 
# provides atlas segmentations based on the pre-operative image.
#
# Usage: pre2post.sh subID preop.nii.gz postop.nii.gz
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


# arguments
patient_id=${1}
preop_file=${2}
postop_file=${3}
output_dir=${4}
brainlesionmask=${5}

# get fileparts
preop_onlypath="$(dirname $preop_file)"
preop_onlyfile="$(basename $preop_file)"
postop_onlypath="$(dirname $postop_file)"
postop_onlyfile="$(basename $postop_file)"

# make output folders
mkdir ${output_dir}
TEMPLATE_DIR=./tools/OasisTemplate/

####### brain extraction #######

mkdir ${output_dir}/brainExtract

antsBrainExtraction.sh \
-d 3 \
-a ${preop_file} \
-e ${TEMPLATE_DIR}T_template0.nii.gz \
-m ${TEMPLATE_DIR}T_template0Mask.nii.gz \
-o ${output_dir}/brainExtract/pre_

antsBrainExtraction.sh \
-d 3 \
-a ${postop_file} \
-e ${TEMPLATE_DIR}T_template0.nii.gz \
-m ${TEMPLATE_DIR}T_template0Mask.nii.gz \
-o ${output_dir}/brainExtract/post_

######## Affine registstration of preop to postop image ######## 

# Registration of preop to postop
antsRegistration \
--dimensionality 3 \
--float 0 \
--output ${output_dir}/pre2post_ \
--interpolation Linear \
--use-histogram-matching 0 \
--initial-moving-transform [${output_dir}/brainExtract/post_BrainExtractionBrain.nii.gz,${output_dir}/brainExtract/pre_BrainExtractionBrain.nii.gz,1] \
--transform Rigid[0.1] \
--metric MI[${output_dir}/brainExtract/post_BrainExtractionBrain.nii.gz,${output_dir}/brainExtract/pre_BrainExtractionBrain.nii.gz,1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform Affine[0.1] \
--metric MI[${output_dir}/brainExtract/post_BrainExtractionBrain.nii.gz,${output_dir}/brainExtract/pre_BrainExtractionBrain.nii.gz,1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform SyN[0.1,3,0] \
--metric CC[${output_dir}/brainExtract/post_BrainExtractionBrain.nii.gz,${output_dir}/brainExtract/pre_BrainExtractionBrain.nii.gz,1,4] \
--convergence [100x70x50x20,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--x $brainlesionmask

# transform pre-operative T1 to post-operative T1 space
antsApplyTransforms \
-d 3 \
-i ${preop_file} \
-o ${output_dir}/pre2post_${preop_onlyfile} \
-t ${output_dir}/pre2post_1Warp.nii.gz \
-t ${output_dir}/pre2post_0GenericAffine.mat \
-r ${postop_file} \
-n Linear

# transform pre-operative T1 to post-operative T1 space
antsApplyTransforms \
-d 3 \
-i ${output_dir}/brainExtract/pre_BrainExtractionMask.nii.gz \
-o ${output_dir}/pre2post_pre_BrainExtractionMask.nii.gz \
-t ${output_dir}pre2post_1Warp.nii.gz \
-t ${output_dir}/pre2post_0GenericAffine.mat \
-r ${postop_file} \
-n NearestNeighbor

