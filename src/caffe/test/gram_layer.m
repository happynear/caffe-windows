function [output, grad] = gram_layer(input, dzdy)
[num,channel,height,width] = size(input);
output = zeros(num,channel,channel);
grad = zeros(size(input));
for i=1:num
    slice = reshape(input(i,:,:,:),channel,height*width);
    output(i,:,:) = slice * slice';
    if nargin==2
        grad(i,:,:,:) = reshape((squeeze(dzdy(i,:,:)) + squeeze(dzdy(i,:,:))') * slice,[channel,height,width]);
    end;
end;