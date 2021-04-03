# This code uses the given pre-operative and post-operative images and outputs 
# a resection segmentation.

# Usage: resection_pipeline.sh patient_name preop.nii postop.nii output_dir is_continuous(boolean)
# Example: ./scripts/resection_pipeline.sh 25_f 25_f_preop.nii.gz 25_f_postop.nii.gz analysis/25_f true

# 6/11/20 - created

# name input variables and mask output filename
patient_id=${1}
preop_file=${2}
postop_file=${3}
output_dir=${4}
is_continuous=${5}
mask_name="${patient_id}_predicted_mask.nii.gz"

# generate a predicted mask NIFTI file for the post-operative image
python3 ./scripts/generate_mask.py ${postop_file} ${output_dir} ${mask_name} ${is_continuous}
