# DeepResection #

Deep Learning-Based Automated Segmentation of Resection Cavities on Postsurgical Epilepsy MRI

T. Campbell Arnold*, Ramya Muthukrishnan*, Akash R. Pattnaik, Adam Gibson, Nishant Sinha, Sandhitsu R. Das, Brian Litt, Dario J, Englot, Victoria L. Morgan, Kathryn A. Davis, Joel M. Stein

*These authors contributed equally

Center for Neuroengineering and Therapeutics, University of Pennsylvania

Deep learning code for neurosurgery resection zone segmentation on T1 MRI, implemented in Keras with Tensorflow backend.

## Prerequisites ##

- Linux OS / Ubuntu

- Python package dependencies in `requirements.txt`

- Postoperative (and preoperative) MRI in NIfTI format

## Getting Started ##

Clone the repo: `git clone https://github.com/tcama/DeepResection.git`

## Running the Pipeline ##

Run segmentation only:

`./pipeline/resection_segmentation_only_pipeline.sh patient_name postop_mri.nii output_dir`

Run entire pipeline, including volumetric resection report:

`./pipeline/resection_pipeline.sh patient_name preop_mri.nii postop_mri.nii output_dir`

Run pipeline with deformable registration:

`./pipeline/resection_deformable_pipeline.sh patient_name preop_mri.nii postop_mri.nii output_dir`

## Using the GUI ##

## Example outputs ##

### Predicted mask ###

After running the pipeline, the predicted mask should be a NIfTI file ending with `predicted_mask.nii` in the specified output directory. It can then be opened in an image viewer, such as ITK-Snap, alongside the postoperative input.

<img src='images/sample_predicted_mask.png' align="center" width=500>

### Volumetric resection report ###

If running the full pipeline, the numeric volumetric resection results should be in `output_dir/resected_results.txt`, the HTML report should be in `output_dir/resection_report.html`, and a visualization of the resection should be in `output_dir/resection_views.png`. Examples are shown below.

Text file output:

```
Total resection volume (cubic cm): 19.32525634765625
Frontal_Inf_Orb_R: 99.947% remaining
Hippocampus_R: 91.254% remaining
ParaHippocampal_R: 45.888% remaining
Amygdala_R: 81.513% remaining
Fusiform_R: 79.39699999999999% remaining
Temporal_Sup_R: 99.963% remaining
Temporal_Pole_Sup_R: 68.662% remaining
Temporal_Mid_R: 92.673% remaining
Temporal_Pole_Mid_R: 41.357% remaining
Temporal_Inf_R: 82.94% remaining
Cerebelum_3_R: 99.81% remaining
```

HTML report:

<img src='images/sample_html_report.png' align="center" width=500>

Resection views PNG output:

<img src='images/sample_resection_views.png' align="center" width=500>

### GUI ###


