function CNNComplexitySave(layers,savefolder,image_size,save_num)

net_model = 'D:\project\NNComplexity\net_define.prototxt';
inception_file = 'inception.prototxt';
inception_content = fileread(inception_file);
conv1x1_file = '1x1conv.prototxt';
conv1x1_content = fileread(conv1x1_file);
pooling_file = 'pooling.prototxt';
pooling_content = fileread(pooling_file);
maxout_file = 'maxout.prototxt';
maxout_content = fileread(maxout_file);
activation_file = 'activation.prototxt';
activation_content = fileread(activation_file);
output_file = 'output.prototxt';
output_content = fileread(output_file);

for sss=1:save_num
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

activation = 'ReLU';

width = image_size(1);
height = image_size(2);
border = 5;

fid = fopen(net_model,'w');
proto_file{1} = 'name: "mnist_siamese_train_test"';
proto_file{2} = 'input: "data"';
proto_file{3} = 'input_dim: 1';
proto_file{4} = 'input_dim: 2';
proto_file{5} = ['input_dim: ' num2str(width)];
proto_file{6} = ['input_dim: ' num2str(height)];
for i=1:6
    fprintf(fid,'%s\r\n',proto_file{i});
end;
top_layer = 'data';
top_layer_exp = 'top: "(.+?)"';
for i=1:length(layers)
    if strcmp(layers{i}.type,'convolution')
        this_layer = strrep(conv1x1_content,'{num}',num2str(i));
        this_layer = strrep(this_layer,'{node_num}',num2str(layers{i}.outputmaps));
        this_layer = strrep(this_layer,'{bottom_name}',top_layer);
        this_layer = strrep(this_layer,'{kernel_size}',num2str(layers{i}.kernelsize));
        if strcmp(layers{i}.activation,'maxout')
            top_layer = regexp(this_layer,top_layer_exp,'tokens');
            top_layer = top_layer{end}{1};
            fprintf(fid,'%s\r\n',this_layer);
            this_layer = strrep(maxout_content,'{num}',num2str(i));
            this_layer = strrep(this_layer,'{bottom_name}',top_layer);
        elseif strcmp(layers{i}.activation,'LReLU')
            top_layer = regexp(this_layer,top_layer_exp,'tokens');
            top_layer = top_layer{end}{1};
            fprintf(fid,'%s\r\n',this_layer);
            this_layer = strrep(activation_content,'{num}',num2str(i));
            this_layer = strrep(this_layer,'{bottom_name}',top_layer);
            this_layer = strrep(this_layer,'{activation}','ReLU');
            this_layer = strrep(this_layer,'{negative_slope}',num2str(rand() / 2));
        else
            top_layer = regexp(this_layer,top_layer_exp,'tokens');
            top_layer = top_layer{end}{1};
            fprintf(fid,'%s\r\n',this_layer);
            this_layer = strrep(activation_content,'{num}',num2str(i));
            this_layer = strrep(this_layer,'{bottom_name}',top_layer);
            this_layer = strrep(this_layer,'{activation}',layers{i}.activation);
            this_layer = strrep(this_layer,'{negative_slope}',num2str(0));
        end;
    elseif strcmp(layers{i}.type,'pooling')
        this_layer = strrep(pooling_content,'{num}',num2str(i));
        this_layer = strrep(this_layer,'{bottom_name}',top_layer);
        this_layer = strrep(this_layer,'{method}',layers{i}.method);
        this_layer = strrep(this_layer,'{scale}',num2str(layers{i}.scale));
    elseif strcmp(layers{i}.type,'inception')
        this_layer = strrep(inception_content,'{num}',num2str(i));
        this_layer = strrep(this_layer,'{bottom_name}',top_layer);
        this_layer = strrep(this_layer,'{1x1node}',num2str(layers{i}.node1x1));
        this_layer = strrep(this_layer,'{3x3reduce}',num2str(layers{i}.reduce3x3));
        this_layer = strrep(this_layer,'{3x3node}',num2str(layers{i}.node3x3));
        this_layer = strrep(this_layer,'{5x5reduce}',num2str(layers{i}.reduce5x5));
        this_layer = strrep(this_layer,'{5x5node}',num2str(layers{i}.node5x5));
        this_layer = strrep(this_layer,'{poolconv}',num2str(layers{i}.poolconv));
    elseif strcmp(layers{i}.type,'maxout')
        this_layer = strrep(maxout_content,'{num}',num2str(i));
        this_layer = strrep(this_layer,'{bottom_name}',top_layer);
    end;
    top_layer = regexp(this_layer,top_layer_exp,'tokens');
    top_layer = top_layer{end}{1};
    fprintf(fid,'%s\r\n',this_layer);
end;
this_layer = strrep(output_content,'{bottom_name}',top_layer);
fprintf(fid,'%s\r\n',this_layer);
fclose(fid);

train_net = caffe.Net(net_model,'train');

input_data = zeros(height, width, 2, 1, 'single');
input_data(:,:,1,1) = repmat((1:height)',1,width) / height;
input_data(:,:,2,1) = repmat((1:width),height,1) / width;
input_data = input_data - 0.5;

output_data = train_net.forward({input_data});

output = output_data{1};
output = output(border+1:end-border,border+1:end-border,:);
output = bsxfun(@minus,output,mean(mean(output,1),2));
output_std = std(reshape(output,[size(output,1)*size(output,2) size(output,3)]));
output_std = reshape(output_std,[1 1 3]);
output = bsxfun(@rdivide,output,output_std);
output = uint8(output * 100 + 128);
% output = Lab2RGB(output);
if ~exist(savefolder,'dir')
    mkdir(savefolder);
end;
imwrite(output,[savefolder '/' num2str(sss) '.bmp']);
end;
% imshow(output);
end