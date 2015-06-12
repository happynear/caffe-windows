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
net_model = 'D:\deeplearning\caffe-windows\examples\cifar10\cifar10_full_train_test.prototxt';
net_weights = 'D:\deeplearning\caffe-windows\examples\cifar10\cifar10_full_iter_80000.caffemodel';

%%%%%%%%%extract the train features
train_net = caffe.Net(net_model,net_weights,'train');
output_blob_index = train_net.name2blob_index('pool3');%feature blob
output_blob = train_net.blob_vec(output_blob_index);
output_label_index = train_net.name2blob_index('label');
output_label = train_net.blob_vec(output_label_index);
feature_train = [];
label_train = [];
num = 500;%data_num / batch_size
for i = 1 : num
    disp(i);
    train_net.forward_prefilled();
    output = output_blob.get_data();
    output = reshape(output,size(output,1)*size(output,2)*size(output,3),size(output,4));
    if isempty(feature_train)
        feature_train = zeros(size(output,1),size(output,2)*num);
        feature_train(:,1:size(output,2)) = output;
        label_train = zeros(1,size(output,2)*num);
        label_train(1:size(output,2)) = output_label.get_data();
    else
        feature_train(:,(i-1)*size(output,2)+1:i*size(output,2)) = output;
        label_train((i-1)*size(output,2)+1:i*size(output,2)) = output_label.get_data();
    end;
end;
%check if has traversed all the data
train_net.forward_prefilled();
output = output_blob.get_data();
output = reshape(output,size(output,1)*size(output,2)*size(output,3),size(output,4));
assert(sum(sum(abs(output - feature_train(:,1:size(output,2)))))==0);

%%%%%%%%%extract the test features
test_net = caffe.Net(net_model,net_weights,'test');
output_blob_index = test_net.name2blob_index('pool3');%feature blob
output_blob = test_net.blob_vec(output_blob_index);
output_label_index = test_net.name2blob_index('label');
output_label = test_net.blob_vec(output_label_index);
feature_test = [];
label_test = [];
num = 100;%data_num / batch_size
for i = 1 : num
    disp(i);
    test_net.forward_prefilled();
    output = output_blob.get_data();
    output = reshape(output,size(output,1)*size(output,2)*size(output,3),size(output,4));
    if isempty(feature_test)
        feature_test = zeros(size(output,1),size(output,2)*num);
        feature_test(:,1:size(output,2)) = output;
        label_test = zeros(1,size(output,2)*num);
        label_test(1:size(output,2)) = output_label.get_data();
    else
        feature_test(:,(i-1)*size(output,2)+1:i*size(output,2)) = output;
        label_test((i-1)*size(output,2)+1:i*size(output,2)) = output_label.get_data();
    end;
end;
%check if has traversed all the data
test_net.forward_prefilled();
output = output_blob.get_data();
output = reshape(output,size(output,1)*size(output,2)*size(output,3),size(output,4));
assert(sum(sum(abs(output - feature_test(:,1:size(output,2)))))==0);