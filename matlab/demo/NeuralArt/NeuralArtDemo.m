%% A messy code for Neural Art
%% It's not very good now.
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% original_file = 'D:\deepLearning\caffe-windows\matlab\demo\Inceptionism\googlenet_neuralart.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_googlenet.caffemodel';
original_file = 'D:\deepLearning\caffe-windows\matlab\demo\NeuralArt\VGG_16_nueralart.prototxt';
net_weights = 'D:\deepLearning\caffe-windows\matlab\demo\NeuralArt\VGG16_thinned_net.caffemodel';
% style_layer = {'icp2_in','icp3_out','icp5_out','icp7_out','icp9_out'};
style_layer = {'conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'};
style_weights = [1 1 1 1 1 0.1];
content_layer = {'conv4_2'};
style_image = imread('d:\starry_night.jpg');
content_image = imread('d:\tubingen.jpg');
content_image = imresize(content_image,0.5,'bilinear');
% content_image = content_image(end-199:end,1:200,:);
[style_generate_prototxt, style_pattern, content_pattern] = MakeStylePrototxt(original_file, net_weights, style_layer, style_weights, content_layer, style_image, content_image);

forward_input = cell(length(style_pattern)+2,1);
forward_input(3:end) = style_pattern;
forward_input(2) = {content_pattern};

mean_file = [];
vgg_mean =  [103.939, 116.779, 123.68];
[height, width, channel] = size(content_image);

%%%%%%%%%extract the train features
stylegen_net = caffe.Net(style_generate_prototxt,net_weights,'test');
mean_image = permute(repmat(vgg_mean',[1,size(content_image,2),size(content_image,1)]),[2,3,1]);
im_data = content_image(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = reshape(im_data,[width, height, 3, 1]);
for c = 1:3
    im_data(:, :, c, :) = im_data(:, :, c, :) - vgg_mean(c);
end
input_data = randn(size(mean_image,1), size(mean_image,2), 3, 1, 'single')*50;
% input_data = im_data;

use_clip = false;
use_cv_norm = true;
use_weight_decay = false;
use_gradient_blur = false;
use_dropout = false;

H = fspecial('gaussian',[17 17],5);
input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
forward_input(1) = {input_data};
prob = stylegen_net.forward(forward_input);
stylegen_content_pattern = stylegen_net.blob_vec(stylegen_net.name2blob_index(content_layer{1})).get_data();

c1 = content_pattern(:,:,1);
c2 = stylegen_content_pattern(:,:,1);
blur_data = zeros(size(input_data));
base_lr = 10;
max_lr = 100;
lambda1 = 0.00001;
lambda2 = 0.0001;
last_prob = -999999;
momentum = 0.8;
lastgrad = zeros(size(mean_image));
mask = ones(size(mean_image,1), size(mean_image,2));
iter = 1;
dropout = 0.5;

while 1
    lr = base_lr;% * sqrt(this_prob / (1 - this_prob));
    stylegen_net.backward_prefilled();
    res = stylegen_net.blob_vec(stylegen_net.name2blob_index('data')).get_diff();
    
    bak_data = input_data;
    
    if use_gradient_blur
        res = imfilter(res,H,'same');
    end;
    
    if use_clip
        app_gradient = sum(abs(res .* input_data(:,:,:,1)),3);
        app_gradient = app_gradient < mean(app_gradient(:)) * 0.5;
        grad = reshape(res,[size(mean_image,1)*size(mean_image,2) 3]);
        grad(app_gradient==1,:) = 0;
        grad = reshape(grad,size(input_data));
        res = grad;
    end;
    
    
    lastgrad = (1 - momentum) * lr * res   + momentum * lastgrad;%/ norm(res(:))
    input_data(:,:,:,1) = input_data(:,:,:,1) - lastgrad;
    
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
        if rand()>0.5
            input_data(:,:,:,1) = input_data(:,:,:,1) - lr * lambda2 * Gx;
        else
            input_data(:,:,:,1) = input_data(:,:,:,1) - lr * lambda2 * Gy;
        end;
    end;
    if use_weight_decay
        input_data(:,:,:,1) = input_data(:,:,:,1) - lr * lambda1 * I;
    end;
   
    forward_input(1) = {input_data};
    prob = stylegen_net.forward(forward_input);
    for i=1:length(prob)
        if length(prob{i})>1
            prob{i}=[];
        end;
    end;
    
    this_prob = [prob{2:length(style_pattern)+2}] * style_weights';
    fprintf('iter=%d,lr=%f,prob=',iter,lr);
    for p = 2:length(style_pattern) + 2
        fprintf('%f,',prob{p});
    end;
    fprintf('last_prob=%f\n',last_prob);
    iter = iter + 1;
    
    if mod(iter,10)==0
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
    end;
    if this_prob>last_prob
        base_lr = base_lr * 0.99;
%         input_data = bak_data;
    end;
    if this_prob<last_prob&&base_lr<max_lr%&& (this_prob-last_prob) / this_prob < 0.001
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
