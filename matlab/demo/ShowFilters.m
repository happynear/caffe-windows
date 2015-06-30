caffe.reset_all();

net_model = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\googlenet_deploy_prob.prototxt';
net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\thinned_net.caffemodel';
net = caffe.Net(net_model,net_weights,'test');
nth_layer = net.layer_vec(net.name2layer_index('conv1'));
nth_layer_blob1_data = nth_layer.params(1).get_data();

sizeB = size(nth_layer_blob1_data);
GridL = ceil(sqrt(sizeB(4)));

scale = 4;
border = 2;
sizeB(1) = sizeB(1) * scale;
sizeB(2) = sizeB(2) * scale;

background = zeros((border+sizeB(1))*GridL+border,(border+sizeB(2))*GridL+border,sizeB(3));
minV = min(nth_layer_blob1_data(:));
maxV = max(nth_layer_blob1_data(:));
nth_layer_blob1_data = (nth_layer_blob1_data - minV) / (maxV - minV);

for i = 1:sizeB(4)
    x = ceil(i / GridL);
    y = mod(i - 1,GridL) + 1;
    patch = imresize(nth_layer_blob1_data(:,:,:,i),[sizeB(1) sizeB(2)],'nearest');
    background(border + (x-1)*(border+sizeB(1)) + 1 : x*(border+sizeB(1)),border + (y-1)*(border+sizeB(2)) + 1 : y*(border+sizeB(2)),:) = patch;
end;

figure(3);
imshow(background);