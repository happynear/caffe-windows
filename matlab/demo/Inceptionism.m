caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% warning: You must change all the file_path to their absolute path in the
% prototxt file!
% for example:
%   transform_param {
%     mean_file: "D:/deeplearning/caffe-windows/examples/cifar10/mean.binaryproto"
%   }
%   data_param {
%     source: "D:/deeplearning/caffe-windows/examples/cifar10/cifar10_train_leveldb"
%     batch_size: 100
%     backend: LEVELDB
%   }
net_model = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\googlenet_deploy_prob.prototxt';
net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\thinned_net.caffemodel';
mean_file = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_mean.binaryproto';
% net_model = 'D:\deeplearning\caffe-windows\examples\PlaceCNN\hybridCNN_deploy_upgraded.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\PlaceCNN\hybridCNN_iter_700000_upgraded.caffemodel';
% mean_file = 'D:\deeplearning\caffe-windows\examples\PlaceCNN\hybridCNN_mean.binaryproto';

%%%%%%%%%extract the train features
train_net = caffe.Net(net_model,net_weights,'test');
train_net.need_backward();
mean_image = caffe.read_mean(mean_file);
mean_image = mean_image(17:240,17:240,:);
% mean_image = mean_image + randn(size(mean_image));
input_data = zeros(size(mean_image,1), size(mean_image,2), 3, 1, 'single');

% trans_image = single(imread('c:\\cloud.jpg'));
trans_image = imresize(trans_image,[size(mean_image,1), size(mean_image,2)]);
% input_data(:,:,:,1) = trans_image - mean_image;%randn(size(mean_image));
input_data(:,:,:,1) = randn(size(mean_image)) * 30;
% output_blob_index = train_net.name2blob_index('pool3');%feature blob
% output_blob = train_net.blob_vec(output_blob_index);
% output_label_index = train_net.name2blob_index('label');
% output_label = train_net.blob_vec(output_label_index);
prob = train_net.forward({input_data});
[max_prob,max_idx] = max(prob{1});
max_idx = 249;
this_prob = prob{1}(max_idx);
back_data = zeros(size(prob{1}),'single');
back_data(max_idx) = 1;
base_lr = 100000000000000;
lambda = 0.001;
last_prob = 0;
while this_prob<0.9999
%     lr = base_lr;
    lr = base_lr;% * sqrt(this_prob / (1 - this_prob));
%     if this_prob>0.9
%         lr = base_lr * 10;
%     end;
%     if this_prob>0.99
%         lr = base_lr * 100;
%     end;
%     if this_prob>0.999
%         lr = base_lr * 1000;
%     end;
    
    res = train_net.backward({back_data});
%     nth_layer = train_net.layer_vec(train_net.name2layer_index('ip1'));
%     nth_layer_blob1_diff = nth_layer.params(1).get_diff();
%     nth_layer_blob1_data = nth_layer.params(1).get_data();
%     nth_blob = train_net.blob_vec(train_net.name2blob_index('ip1'));
%     nth_blob_diff = nth_blob.get_diff();
%     nth_blob_data = nth_blob.get_data();
    
    bak_data = input_data;
    input_data(:,:,:,1) = input_data(:,:,:,1) + lr * res{1};
%     if this_prob > 0.1
        I = input_data(:,:,:,1) + mean_image;
        Gx = sign(I(2:end-1,2:end-1,:) - I(1:end-2,2:end-1,:)) - sign(I(3:end,2:end-1,:) - I(2:end-1,2:end-1,:));
        Gy = sign(I(2:end-1,2:end-1,:) - I(2:end-1,1:end-2,:)) - sign(I(2:end-1,3:end,:) - I(2:end-1,2:end-1,:));
%         Gx = (I(2:end-1,2:end-1,:) - I(1:end-2,2:end-1,:)) - (I(3:end,2:end-1,:) - I(2:end-1,2:end-1,:));
%         Gy = (I(2:end-1,2:end-1,:) - I(2:end-1,1:end-2,:)) - (I(2:end-1,3:end,:) - I(2:end-1,2:end-1,:));
        input_data(2:end-1,2:end-1,:,1) = input_data(2:end-1,2:end-1,:,1) + lambda * (Gx + Gy);
%         input_data(:,:,:,1) = input_data(:,:,:,1) - lambda * I.^2;
%     end;
    prob = train_net.forward({input_data});
    this_prob = prob{1}(max_idx);
    fprintf('lr=%f,prob=%f,last_prob=%f\n',lr,this_prob,last_prob);
    if this_prob<last_prob
        base_lr = base_lr * 0.9;
        input_data = bak_data;
    end;
    if this_prob>last_prob%&& (this_prob-last_prob) / this_prob < 0.001
        base_lr = base_lr * 1.1;
%         input_data = bak_data;
    end;
    if this_prob>last_prob
        last_prob = this_prob;
    end;
    if lr<0.000001
        break;
    end;
end;
figure(1)
output = reshape(input_data,[size(input_data,1)*size(input_data,2) size(input_data,3)]);
output = zscore(output);
output = reshape(output,[size(input_data,1),size(input_data,2),size(input_data,3)]);
imshow(uint8(output*255));
figure(2);
% imshow(uint8(mean_image + input_data));
output = mean_image + input_data(:,:,:,1);
imshow(uint8(output));