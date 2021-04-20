function [img] = png2nii(foldername, varargin)
% This function takes in a folder name (str) and an optional output
% filepath (str) and NII file to copy header information from. The folder
% should contain a series of ordered PNG files, which will be converted
% into a single NII image.
%
% Usage:
%       [img] = png2nii(foldername);
%       [img] = png2nii(foldername,output_filepath);            % specify output directory
%       [img] = png2nii(foldername,output_filepath,hdr_img);    % grabs hdr information from  template file
%       [img] = png2nii(foldername,output_filepath,hdr_img,1);  % will reslice image
%       [img] = png2nii('D:\GANs\test\P004','D:\GANs\test\P004','D:\GANs\test\P004_orig.nii');
%
% Input:
%       foldername (str): filepath to folder containing PNG files
%       output_filepath (str): the filepath and name of the NII file you will
%       output. Do not include the .nii extension in this variable.
%       hdr_img (str): filepath to the NII file you would like to grab hdr
%       informatation from.
%       resize_flag (1 or any other value): 1 will resize the image to
%       match the hdr image size. Any other values will replace the hdr
%       information with the new volumes size.
%
% Notes:
%   1.  Be certain PNG files will be sorted in
%       the appropriate order by dir command (use zero padding if numbered).
%   2.  This function is designed for axial sliced PNG files. It can be
%       used for other planes of view, but be conscious of output
%       orientation.
%   3.  If your PNG files have been padded (such as to make square images
%       for a CNN) then you may want to remove padding first. It will lead to
%       squished dimensions otherwise.

% set defaults
[filepath,name,ext] = fileparts( foldername );

% set defaults
switch length( varargin )
    case 0
        output_filepath = [filepath filesep name]; % set filename to foldername
        hdr_img = []; % no hdr info
    case 1
        output_filepath = varargin{1} % set output filename
        hdr_img = []; % no hdr info
    case 2
        output_filepath = varargin{1}; % set output filename
        hdr_img = varargin{2};
        resize_flag = 0;
    case 3
        output_filepath = varargin{1}; % set output filename
        hdr_img = varargin{2};
        resize_flag = varargin{3};
    otherwise
        'Too many or too few inputs'
end

% get the files
files = dir( [foldername filesep '*.png'] );

% preallocate NII image space
[x,y] = size( imread( [files(1).folder filesep files(1).name] ) ); % get xy dimensions based on first png
z = length( files ); % set z dimension based on number of files
img = zeros( x, y, z); % preallocate 3D volume

% loop through the slice and input into NII volume
for i = 1:z
    img(:,:,i) = imread( [files(i).folder filesep files(i).name] );
end

% save out nifti image
if hdr_img % if hdr image, get hdr info and save out image
    
    info = niftiinfo( hdr_img ); % get hdr info
    
    if setdiff( info.ImageSize, [x y z] ) % if hdr size and volume size differ, make adjustment
        if resize_flag == 1 % resize image or change hdr info
            img = imresize3(img,[info.ImageSize(1) info.ImageSize(2) info.ImageSize(3)]); % Option1: resize image to match hdr
        else
            info.ImageSize = [x y z]; % Option2: replace hdr info with new size
        end
    end
    
    img = cast( img, info.Datatype ); % cast as hdr_img type
    niftiwrite( img, output_filepath ,info ,'Compressed',true); % save out image to specified path
    
else % if no hdr img, simply write out image
    
    img = double( img ); % make variable a double by default
    niftiwrite( img, output_filepath ,'Compressed',true); % save out image to specified path
    
end


end