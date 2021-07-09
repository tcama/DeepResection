# This code uses the given pre-operative and post-operative images and outputs 
# a resection segmentation.

# Usage: resection_pipeline.sh patient_name postop.nii output_dir
# Example: ./pipeline/resection_pipeline.sh 25_f 25_f_postop.nii.gz analysis/25_f

# 6/11/20 - created

# name input variables and mask output filename
patient_id=${1}
postop_file=${2}
output_dir=${3}

while true; do
    read -p "Is the entire resection continuous? [y/n]" yn
    case $yn in
        [Yy]* ) is_continuous=1; break;;
        [Nn]* ) is_continuous=0; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

mask_name="${patient_id}_predicted_mask.nii.gz"

# generate a predicted mask NIFTI file for the post-operative image
python3 ./scripts/generate_mask.py ${postop_file} ${output_dir} ${mask_name} ${is_continuous}
