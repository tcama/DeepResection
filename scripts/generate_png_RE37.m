% This script takes images in ./data/nii and converts them into png files
% in ./data/png. Each subject folder will contain and img and mask
% directory.

subjects_path = fullfile( 'data', 'nii' ); % path to subject folders
subs = dir( subjects_path ); % get subjects foldernames
subs = subs(3:end); % remove . and ..

% loop through each subject
for i = 1:length(subs)
    
    subject_path = fullfile( subs(i).folder, subs(i).name ); % path to data
    
    % generate post-op png
    filename = fullfile( subject_path, 'postop.nii.gz' ); % path to postop image
    out_dir = fullfile( 'data', 'png', subs(i).name, 'img' ); % output path for postop image
    mkdir( out_dir ); % make output directory if it doesn't exist
    nii2png(filename, out_dir); % output png data
    
    % generate mask png
    filename = fullfile( subject_path, 'resection_mask.nii.gz' ); % path to postop image
    out_dir = fullfile( 'data', 'png', subs(i).name, 'mask' ); % output path for postop image
    mkdir( out_dir ); % make output directory if it doesn't exist
    nii2png(filename, out_dir); % output png data
    
end


%nii2png(filename, out_dir)

