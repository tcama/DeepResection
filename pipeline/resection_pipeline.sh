# This code uses the given pre-operative and post-operative images and outputs the resection percentages by brain region as a json file
# and prints percent volume resected for each ROI in the brain

# Usage: resection_pipeline.sh patient_name preop.nii postop.nii output_dir
# Example: ./scripts/resection_pipeline.sh 25_f 25_f_preop.nii.gz 25_f_postop.nii.gz analysis/25_f

# 6/5/20 - created

# apply an atlas to pre-operative image, register atlas to post-operative image
./scripts/pre2post.sh ${1} ${2} ${3}

mask_name="${1}_predicted_mask.nii.gz"

# generate a predicted mask NIFTI file for the post-operative image
postop_file="data/${1}/${3}"
python3 ./scripts/generate_mask.py ${postop_file} ${4} ${mask_name}

mask_file="${4}/${mask_name}"
atlas_file="analysis/${1}/atlas2post/atlas2post_AAL116_origin_MNI_T1.nii"

# generate a json file that calculates the percentages by brain region
python3 ./scripts/calculate_resected_volumes.py ${mask_file} ${atlas_file} ${4}