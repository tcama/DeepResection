# This code uses the given pre-operative and post-operative images and outputs the resection volumes by brain region as a json file
# and prints percent volume resected for each ROI in the brain

# Usage: resection_pipeline.sh patient_name preop.nii postop.nii output_dir

# 6/5/20 - created

# apply an atlas to pre-operative image, register atlas to post-operative image
./scripts/pre2post.sh ${1} ${2} ${3}

mask_name="${1}_predicted_mask.nii.gz"

# generate a predicted mask NIFTI file for the post-operative image
python3 generate_mask.py ${3} ${4} ${mask_name}

mask_file="${4}/${mask_name}"
atlas_file="analysis/${1}/atlas2post_AAL116_origin_MNI_T1.nii.gz"
python3 calculate_resected_volumes.py ${mask_file} ${atlas_file} ${4}