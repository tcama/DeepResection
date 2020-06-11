# This code uses the given pre-operative and post-operative images and outputs the resection percentages by brain region as a json file
# and prints percent volume resected for each ROI in the brain

<<<<<<< HEAD
# Usage: resection_pipeline.sh patient_name preop.nii postop.nii output_dir
# Example: ./pipeline/resection_pipeline.sh 25_f 25_f_preop.nii.gz 25_f_postop.nii.gz analysis/25_f
=======
# Usage: resection_pipeline.sh patient_name preop.nii postop.nii output_dir is_continuous(boolean)
# Example: ./scripts/resection_pipeline.sh 25_f 25_f_preop.nii.gz 25_f_postop.nii.gz analysis/25_f true
>>>>>>> 6d18283ab186e83fb535cfa295dd0e9647d9bd8b

# 6/11/20 - created

# apply an atlas to pre-operative image, register atlas to post-operative image
patient_id=${1}
preop_file=${2}
postop_file=${3}
./scripts/pre2post.sh ${patient_id} ${preop_file} ${postop_file}

mask_name="${patient_id}_predicted_mask.nii.gz"

# generate a predicted mask NIFTI file for the post-operative image
output_dir=${4}
is_continuous=${5}
postop_full_file="data/${patient_id}/${postop_file}"
python3 ./scripts/generate_mask.py ${postop_full_file} ${output_dir} ${mask_name} ${is_continuous^}

# generate a txt file that calculates the resection volume and percent remaining by brain region
mask_file="${output_dir}/${mask_name}"
atlas_file="analysis/${patient_id}/atlas2post/atlas2post_AAL116_origin_MNI_T1.nii"

atlas_mappings="tools/atlas_mappings/AAL116.txt"

python3 ./scripts/calculate_resected_volumes.py ${mask_file} ${atlas_file} ${atlas_mappings} ${output_dir}