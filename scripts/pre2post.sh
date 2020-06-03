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

#TEMPLATE_DIR=./tools/ADNI_normal_atlas/
ATLAS_DIR=./tools/atlases/
OUT_DIR=./analysis/${1}/
mkdir ${OUT_DIR}
mkdir ${OUT_DIR}pre2post
OUT_DIR=./analysis/${1}/pre2post/

################### Affine registstration of preop to postop image ################### 

# Registration of preop to postop
antsRegistration \
--dimensionality 3 \
--float 0 \
--output ${OUT_DIR}pre2post_ \
--interpolation Linear \
--use-histogram-matching 0 \
--initial-moving-transform [./data/${1}/${3},./data/${1}/${2},1] \
--transform Rigid[0.1] \
--metric MI[./data/${1}/${3},./data/${1}/${2},1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform Affine[0.1] \
--metric MI[./data/${1}/${3},./data/${1}/${2},1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox

# transform pre-operative T1 to post-operative T1 space
antsApplyTransforms \
-d 3 \
-i ./data/${1}/${2} \
-o ${OUT_DIR}pre2post_${2} \
-t ${OUT_DIR}pre2post_0GenericAffine.mat \
-r ./data/${1}/${3} \
-n Linear

################### Register atlas to preop image ################### 

mkdir ./analysis/${1}/atlas2post/

# Registration of atlas to preop
antsRegistration \
--dimensionality 3 \
--float 0 \
--output ./analysis/${1}/atlas2post/atlas2post_ \
--interpolation Linear \
--use-histogram-matching 0 \
--initial-moving-transform [${OUT_DIR}pre2post_${2},${ATLAS_DIR}MNI_T1.nii,1] \
--transform Rigid[0.1] \
--metric MI[${OUT_DIR}pre2post_${2},${ATLAS_DIR}MNI_T1.nii,1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform Affine[0.1] \
--metric MI[${OUT_DIR}pre2post_${2},${ATLAS_DIR}MNI_T1.nii,1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform SyN[0.1,3,0] \
--metric CC[${OUT_DIR}pre2post_${2},${ATLAS_DIR}MNI_T1.nii,1,4] \
--convergence [100x70x50x20,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox

# change output directory
OUT_DIR=./analysis/${1}/atlas2post/

# transform MNI-T1 to pre-operative T1 space
antsApplyTransforms \
-d 3 \
-i ${ATLAS_DIR}MNI_T1.nii \
-o ${OUT_DIR}atlas2post_MNI_T1.nii \
-t ${OUT_DIR}atlas2post_1Warp.nii.gz \
-t ${OUT_DIR}atlas2post_0GenericAffine.mat \
-r ./analysis/${1}/pre2post/pre2post_${2} \
-n Linear

# transform MNI-segmentation to pre-operative T1 space
antsApplyTransforms \
-d 3 \
-i ${ATLAS_DIR}AAL116_origin_MNI_T1.nii \
-o ${OUT_DIR}atlas2post_AAL116_origin_MNI_T1.nii \
-t ${OUT_DIR}atlas2post_1Warp.nii.gz \
-t ${OUT_DIR}atlas2post_0GenericAffine.mat \
-r ./analysis/${1}/pre2post/pre2post_${2} \
-n NearestNeighbor
