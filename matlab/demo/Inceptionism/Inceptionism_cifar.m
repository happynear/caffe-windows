caffe.reset_all();
% caffe.set_mode_gpu();
% gpu_id = 0;  % we will use the first gpu in this demo
% caffe.set_device(gpu_id);

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
net_model = 'D:\deeplearning\caffe-windows\examples\cifar10\cifar10_full.prototxt';
net_weights = 'D:\deeplearning\caffe-windows\examples\cifar10\cifar10_full_iter_90000.caffemodel';
mean_file = 'D:\deeplearning\caffe-windows\examples\cifar10\mean.binaryproto';

%%%%%%%%%extract the train features
train_net = caffe.Net(net_model,net_weights,'train');
train_net.need_backward();
mean_image = caffe.read_mean(mean_file);
% mean_image = mean_image + randn(size(mean_image));
input_data = zeros(size(mean_image,1), size(mean_image,2), 3, 1, 'single');
input_data(:,:,:,1) = randn(size(mean_image));
% output_blob_index = train_net.name2blob_index('pool3');%feature blob
% output_blob = train_net.blob_vec(output_blob_index);
% output_label_index = train_net.name2blob_index('label');
% output_label = train_net.blob_vec(output_label_index);
prob = train_net.forward({input_data});
[max_prob,max_idx] = max(prob{1});
max_idx = 1;
back_data = zeros(size(prob{1}),'single');
back_data(max_idx) = -1;
base_lr = 500;
while max_prob<0.9999
    lr = base_lr;
    if max_prob>0.9
        lr = base_lr * 10;
    end;
    if max_prob>0.99
        lr = base_lr * 100;
    end;
    if max_prob>0.999
        lr = base_lr * 1000;
    end;
    disp(max_prob);
    res = train_net.backward({back_data});
    nth_layer = train_net.layer_vec(train_net.name2layer_index('ip1'));
    nth_layer_blob1_diff = nth_layer.params(1).get_diff();
    nth_layer_blob1_data = nth_layer.params(1).get_data();
    nth_blob = train_net.blob_vec(train_net.name2blob_index('ip1'));
    nth_blob_diff = nth_blob.get_diff();
    nth_blob_data = nth_blob.get_data();
    input_data(:,:,:,1) = input_data(:,:,:,1) - lr * res{1};
    prob = train_net.forward({input_data});
    [max_prob,max_idx] = max(prob{1});
end;
imshow(uint8(mean_image + input_data));