function [img_padded] = img_square_pad( img )
% this function takes in an image matrix and pads to make the image square

[x,y] = size( img );

if x == y % if dimensions equal, set output to input
    % disp('Padding not necessary'); % commented out to reduce output
    img_padded = img;
    
else % if dimensions differ, pad.
    
    % get difference and padding for each side
    pad = abs( x - y );
    pre = ceil(pad/2);
    post = floor(pad/2);
    
    % add padding to apporpriate dimension
    if x > y
        img_padded = cast( zeros(x,x), class(img)); % preallocate matrix
        img_padded(:, pre+1:end-post) = img; % input image
    else
        img_padded = cast( zeros(y,y), class(img)); % preallocate matrix
        img_padded(pre+1:end-post, :) = img; % input image
    end
    
end

end