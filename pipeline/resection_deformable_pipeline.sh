# This code uses the given pre-operative and post-operative images and outputs the resection percentages by brain region as a json file
# and prints percent volume resected for each ROI in the brain

# Usage: resection_pipeline.sh patient_name preop.nii postop.nii output_dir
# Example: ./pipeline/resection_deformable_pipeline.sh 25_f ./data/25_f/25_f_preop.nii.gz ./data/25_f/25_f_postop.nii.gz ./analysis/25_f/
# 4/14/21 - forked from pre2post.sh

# define inputs and prompt user for the is_continuous input
patient_id=${1}
preop_file=${2}
postop_file=${3}
output_dir=${4}
preop_onlyfile="$(basename $preop_file)"

while true; do
    read -p "Is the entire resection continuous? [y/n]" yn
    case $yn in
        [Yy]* ) is_continuous=1; break;;
        [Nn]* ) is_continuous=0; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# create output directory
mkdir ${output_dir}

# generate a predicted mask NIFTI file for the post-operative image
mask_name="${patient_id}_predicted_mask.nii.gz"
inv_mask_name="${output_dir}/inv_${mask_name}"
python3 ./scripts/generate_mask.py ${postop_file} ${output_dir} ${mask_name} ${is_continuous}

# apply an atlas to pre-operative image, register atlas to post-operative image
./scripts/pre2post_deformable.sh ${patient_id} ${preop_file} ${postop_file} ${output_dir} ${inv_mask_name}

# register DKL atlas to preoperative image
python3 ./scripts/register_atlas_to_preop.py ${patient_id} ${output_dir}/pre2post_${preop_onlyfile} ${output_dir}

# generate a txt file that calculates the resection volume and percent remaining by brain region
mask_file="${output_dir}/${mask_name}"
atlas_file="${output_dir}/${patient_id}_DKT_DL.nii.gz"
atlas_mappings="atlas/dkt_atlas_mappings.txt"

python3 ./scripts/calculate_resected_volumes.py ${postop_file} ${mask_file} ${atlas_file} ${atlas_mappings} ${output_dir}
