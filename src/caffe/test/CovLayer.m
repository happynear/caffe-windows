function [output, grad] = CovLayer(input, dzdy)
[num,channel,height,width] = size(input);
output = zeros(num,channel,channel);
grad = zeros(size(input));
for i=1:num
    slice = reshape(input(i,:,:,:),channel,height*width);
    output(i,:,:) = slice * slice' / (height * width);
    if nargin==2
        grad(i,:,:,:) = reshape(squeeze(dzdy(i,:,:)) * slice,[channel,height,width]) / (height * width) ;
    end;
end;