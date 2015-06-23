caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
net_model = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\googlenet_deploy_conv1.prototxt';
net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\thinned_net.caffemodel';
mean_file = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_mean.binaryproto';

train_net = caffe.Net(net_model,net_weights,'test');
train_net.need_backward();
mean_image = caffe.read_mean(mean_file);
mean_image = mean_image(17:240,17:240,:);

trans_image = single(imread('c:\\lena.png'));
trans_image = imresize(trans_image,[size(mean_image,1), size(mean_image,2)]);
input_data(:,:,:,1) = trans_image - mean_image;
prob = train_net.forward({input_data});
nth_blob = train_net.blob_vec(train_net.name2blob_index('conv1'));
nth_blob_data = nth_blob.get_data();

map1 = nth_blob_data(:,:,2);
