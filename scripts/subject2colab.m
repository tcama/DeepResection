function [] = subject2colab(folder, outDir, whichSet)
% This fucntion takes in a subjects folder and outputs the data to a
% training, testings, or validation folder with the subject ID prefixed to
% each image.

switch whichSet
    case 0
        whichSet = 'train';
    case 1
        whichSet = 'test';
    case 2
        whichSet = 'validation';
end
mkdir( [outDir filesep whichSet 'images'] );
mkdir( [outDir filesep whichSet 'masks'] );

% get list of all images and masks files
imgs = dir( [folder filesep 'img' filesep '*.png'] );
masks = dir( [folder filesep 'mask' filesep '*.png'] );

% error if inconsistent images and masks
if length(imgs) ~= length(masks)
    error( ['Different number of masks (N=', nums2tr(length(masks)) ,') and images (N=', nums2tr(length(imgs)) ,')'] );
end

% loop through files, pad if necessary
for i = 1:length(imgs)
    
    
    %% images 
    % read in file and pad
    file = [imgs(i).folder filesep imgs(i).name];
    img = imread( file );
    img = img_square_pad( img );
    
    % set output name and write
    C = strsplit(imgs(i).folder, filesep); % split folder name to get subjectID
    outFilename = [outDir filesep whichSet 'images' filesep C{end-1} '_' imgs(i).name];
    imwrite( img, outFilename);
    
    %% masks
    % read in file and pad
    file = [masks(i).folder filesep imgs(i).name];
    mask = imread( file );
    mask = img_square_pad( mask );
    
    % set output name and write
    C = strsplit(masks(i).folder, filesep); % split folder name to get subjectID
    outFilename = [outDir filesep whichSet 'masks' filesep C{end-1} '_' masks(i).name];
    imwrite( mask, outFilename);
    
end

end