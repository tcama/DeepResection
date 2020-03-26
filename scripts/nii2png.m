function [] = nii2png(filename, out_dir)
% This function takes in a nifti image and converts it into png files along
% the axial dimension.

% read in the image
img = niftiread( filename );
img = double( img ); % recast to double

% loop through the slices
for i = 1:size(img,3)
    
    % zeros padding
    if i < 10
        out_name = ['img_00' num2str(i) '.png'];
    elseif i < 100
        out_name = ['img_0' num2str(i) '.png'];
    else
        out_name = ['img_' num2str(i) '.png'];
    end
    
    % output filepath
    out_name = fullfile( out_dir, out_name );
    
    % save image
    A = squeeze( img(:,:,i) );
    A = A ./ max( img(:) );
    imwrite(A, out_name);

end