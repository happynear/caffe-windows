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
net_model = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\googlenet_deploy_prob3.prototxt';
net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_googlenet.caffemodel';
mean_file = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_mean.binaryproto';
% net_model = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\googlenet_deploy_prob.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\thinned_net.caffemodel';
% mean_file = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_mean.binaryproto';
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
% input_data(120,120,:) = 1;
% input_data(60:180,60,:) =1;
% input_data(60,60:180,:) = 1;
% input_data(180,60:180,:) = 1;
% input_data(60:180,180,:) = 1;

% trans_image = single(imread('e:\\banana.png'));
% trans_image = imresize(trans_image,[size(mean_image,1), size(mean_image,2)]);
% input_data(:,:,:,1) = trans_image - mean_image;%randn(size(mean_image));
input_data(:,:,:,1) = randn(size(mean_image)) * 50;
% input_data(:,:,:,1) = zeros(size(mean_image));
% H = fspecial('gaussian',[20 20],10);
% input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
% input_data = input_data / std(input_data(:)) * 30;

use_clip = false;
use_cv_norm = true;
use_weight_decay = false;
use_image_blur = true;
use_gradient_blur = false;
use_dropout = false;

H = fspecial('gaussian',[7 7],1.2);
% nth_layer = train_net.layer_vec(train_net.name2layer_index('blur'));
% blur_kernel = zeros(7,7,3,3);
% blur_kernel(:,:,1,1) = H;
% blur_kernel(:,:,2,2) = H;
% blur_kernel(:,:,3,3) = H;
% nth_layer.params(1).set_data(blur_kernel);

prob = train_net.forward({input_data});
[max_prob,max_idx] = max(prob{3});
max_idx = 473;
this_prob = prob{3}(max_idx);
back_data = ones(size(prob{3}),'single') * -1;
back_data(max_idx) = 1;
back_cell = prob;
back_cell{1} = back_data;
back_cell{2} = back_data;
back_cell{3} = back_data;
blur_data = zeros(size(input_data));
base_lr = 1;
max_lr = 2000;
lambda1 = 0.00001;
lambda2 = 0.0001;
last_prob = -999;
momentum = 0.8;
lastgrad = zeros(size(mean_image));
mask = ones(size(mean_image,1), size(mean_image,2));
iter = 1;
dropout = 0.5;

while 1
    lr = base_lr;% * sqrt(this_prob / (1 - this_prob));
    res = train_net.backward(back_cell);
    
    bak_data = input_data;
    
    if use_gradient_blur
        res{1} = imfilter(res{1},H,'same');
    end;
    
    if use_clip
        app_gradient = sum(abs(res{1} .* input_data(:,:,:,1)),3);
        app_gradient = app_gradient < mean(app_gradient(:)) * 0.5;
        grad = reshape(res{1},[size(mean_image,1)*size(mean_image,2) 3]);
        grad(app_gradient==1,:) = 0;
        grad = reshape(grad,size(input_data));
        res{1} = grad;
    end;
    
    
    lastgrad = (1 - momentum) * lr * res{1}   + momentum * lastgrad;%/ norm(res{1}(:))
%     lastgrad = reshape(lastgrad,[size(mean_image,1)*size(mean_image,2) 3]);
%     lastgrad(mask==1,:) = 0;
%     lastgrad = reshape(lastgrad,size(input_data));
    input_data(:,:,:,1) = input_data(:,:,:,1) + lastgrad;
    
    if use_cv_norm
        I = input_data(:,:,:,1);
%         Gx = sign(I(2:end-1,2:end-1,:) - I(1:end-2,2:end-1,:)) - sign(I(3:end,2:end-1,:) - I(2:end-1,2:end-1,:));
%         Gy = sign(I(2:end-1,2:end-1,:) - I(2:end-1,1:end-2,:)) - sign(I(2:end-1,3:end,:) - I(2:end-1,2:end-1,:));
        Gx = smoothL1(I(2:end-1,:,:) - I(1:end-2,:,:)) - smoothL1(I(3:end,:,:) - I(2:end-1,:,:));
        Gx = [smoothL1(I(1,:,:) - I(2,:,:)); Gx; smoothL1(I(end,:,:) - I(end-1,:,:))];
        Gy = smoothL1(I(:,2:end-1,:) - I(:,1:end-2,:)) - smoothL1(I(:,3:end,:) - I(:,2:end-1,:));
        Gy = [smoothL1(I(:,1,:) - I(:,2,:)) Gy smoothL1(I(:,end,:) - I(:,end-1,:))];
%         Gx = sign(I(2:end-1,:,:) - I(1:end-2,:,:)) - sign(I(3:end,:,:) - I(2:end-1,:,:));
%         Gx = [sign(I(1,:,:) - I(2,:,:)); Gx; sign(I(end,:,:) - I(end-1,:,:))];
%         Gy = sign(I(:,2:end-1,:) - I(:,1:end-2,:)) - sign(I(:,3:end,:) - I(:,2:end-1,:));
%         Gy = [sign(I(:,1,:) - I(:,2,:)) Gy sign(I(:,end,:) - I(:,end-1,:))];
        input_data(:,:,:,1) = input_data(:,:,:,1) - lr * lambda2 * (Gx + Gy);
    end;
    if use_weight_decay
        input_data(:,:,:,1) = input_data(:,:,:,1) - lr * lambda1 * I;
    end;
%         input_data = (input_data -mean(input_data(:))) / std(input_data(:)) * 30;
%     end;

%     for_forward = reshape(input_data,[size(mean_image,1)*size(mean_image,2) 3]);
%     mask = rand(size(mean_image,1), size(mean_image,2)) < dropout;
%     for_forward(mask==1,:) = 0;
%     for_forward = reshape(for_forward,size(input_data));
    
    if mod(iter,10) ==0&&use_image_blur&&iter<2000
%         blur_data = input_data;
%         blur_data(:,:,:,1) = imfilter(blur_data(:,:,:,1),H,'same');
        input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
    end;
    prob = train_net.forward({input_data});
    
    this_prob = prob{3}(max_idx);
    fprintf('iter=%d,lr=%f,prob1=%f,prob2=%f,prob3=%f,last_prob=%f\n',iter,lr,prob{1}(max_idx),prob{2}(max_idx),this_prob,last_prob);
    iter = iter + 1;
    
    if mod(iter,100)==0
        figure(1)
        output = reshape(input_data,[size(input_data,1)*size(input_data,2) size(input_data,3)]);
        output = zscore(output);
        output = reshape(output,[size(input_data,1),size(input_data,2),size(input_data,3)]);
        output = output(:, :, [3, 2, 1]);
        output = permute(output, [2 1 3]);
        imshow(uint8(output*255));
        figure(2);
        % imshow(uint8(mean_image + input_data));
        output = mean_image + input_data(:,:,:,1);
        output = output(:, :, [3, 2, 1]);
        output = permute(output, [2 1 3]);
        imshow(uint8(output));
        I = output;
        Gx = abs(I(2:end-1,2:end-1,:) - I(1:end-2,2:end-1,:)) + abs(I(3:end,2:end-1,:) - I(2:end-1,2:end-1,:));
        Gy = abs(I(2:end-1,2:end-1,:) - I(2:end-1,1:end-2,:)) + abs(I(2:end-1,3:end,:) - I(2:end-1,2:end-1,:));
        figure(3);hist(Gx(:),1000);
        figure(4);hist(Gy(:),1000);
    end;
    if this_prob<last_prob
        base_lr = base_lr * 0.99;
%         input_data = bak_data;
    end;
    if this_prob>last_prob&&base_lr<max_lr%&& (this_prob-last_prob) / this_prob < 0.001
        base_lr = base_lr * 1.01;
%         input_data = bak_data;
    end;
%     if this_prob>last_prob
        last_prob = this_prob;
%     end;
    if lr<0.000001
        break;
    end;
end;
figure(1)
output = reshape(input_data,[size(input_data,1)*size(input_data,2) size(input_data,3)]);
output = zscore(output);
output = reshape(output,[size(input_data,1),size(input_data,2),size(input_data,3)]);
output = output(:, :, [3, 2, 1]);
imshow(uint8(output*255));
figure(2);
% imshow(uint8(mean_image + input_data));
output = mean_image + input_data(:,:,:,1);
output = output(:, :, [3, 2, 1]);
imshow(uint8(output));
I = output;
Gx = abs(I(2:end-1,2:end-1,:) - I(1:end-2,2:end-1,:)) + abs(I(3:end,2:end-1,:) - I(2:end-1,2:end-1,:));
Gy = abs(I(2:end-1,2:end-1,:) - I(2:end-1,1:end-2,:)) + abs(I(2:end-1,3:end,:) - I(2:end-1,2:end-1,:));
figure(3);hist(Gx(:),1000);
figure(4);hist(Gy(:),1000);