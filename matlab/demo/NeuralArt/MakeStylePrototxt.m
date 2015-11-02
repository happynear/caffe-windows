function [style_generate_prototxt, style_pattern, content_pattern] = MakeStylePrototxt( original_file, weight_file, style_layer, style_weights, content_layer, style_image, content_image )
%MAKESTYLEPROTOTXT 此处显示有关此函数的摘要
%   此处显示详细说明
    
    addpath('../PrototxtGen');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%get style pattern
    style_pattern_prototxt = strrep(original_file,'.prototxt','_style.prototxt');
    image_size = size(style_image);
    height = image_size(1);
    width = image_size(2);
    border = 5;

    fid = fopen(style_pattern_prototxt,'w');
    proto_txt{1} = 'name: "StylePatternGen"';
    proto_txt{2} = 'input: "data"';
    proto_txt{3} = 'input_dim: 1';
    proto_txt{4} = 'input_dim: 3';
    proto_txt{5} = ['input_dim: ' num2str(height)];
    proto_txt{6} = ['input_dim: ' num2str(width)];
    for i=1:6
        fprintf(fid,'%s\r\n',proto_txt{i});
    end;
    
    original_net_model = fileread(original_file);
    fprintf(fid,'%s\r\n',original_net_model);
    
    covariance_file = '../PrototxtGen/covariance.prototxt';
    covariance_content = fileread(covariance_file);
    euclideanloss_file = '../PrototxtGen/smoothL1Loss.prototxt';
    euclideanloss_content = fileread(euclideanloss_file);
    for i = 1 : length(style_layer)
        covariance_layer = strrep(covariance_content,'{num}',num2str(i));
        covariance_layer = strrep(covariance_layer,'{bottom_name}',style_layer{i});
        fprintf(fid,'%s\r\n',covariance_layer);
    end;
    fclose(fid);
    
    style_net = caffe.Net(style_pattern_prototxt,weight_file,'test');
    vgg_mean =  [103.939, 116.779, 123.68];
    im_data = style_image(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = reshape(im_data,[width, height, 3, 1]);
    for c = 1:3
        im_data(:, :, c, :) = im_data(:, :, c, :) - vgg_mean(c);
    end

    output_data = style_net.forward({im_data});
    
    style_pattern = cell(length(style_layer),1);
    for i = 1:length(style_layer)
        style_pattern{i} = style_net.blob_vec(style_net.name2blob_index(['cov' num2str(i)])).get_data();
    end;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%get content pattern
    content_prototxt = strrep(original_file,'.prototxt','_content.prototxt');
    image_size = size(content_image);
    height = image_size(1);
    width = image_size(2);
    
    fid = fopen(content_prototxt,'w');
    proto_txt{1} = 'name: "ContentNet"';
    proto_txt{2} = 'input: "data"';
    proto_txt{3} = 'input_dim: 1';
    proto_txt{4} = 'input_dim: 3';
    proto_txt{5} = ['input_dim: ' num2str(height)];
    proto_txt{6} = ['input_dim: ' num2str(width)];
    for i=1:6
        fprintf(fid,'%s\r\n',proto_txt{i});
    end;
    
    fprintf(fid,'%s\r\n',original_net_model);
    fclose(fid);
    content_net = caffe.Net(content_prototxt,weight_file,'test');
    vgg_mean =  [103.939, 116.779, 123.68];
    im_data = content_image(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = reshape(im_data,[width, height, 3, 1]);
    for c = 1:3
        im_data(:, :, c, :) = im_data(:, :, c, :) - vgg_mean(c);
    end
    
    content_net.forward({im_data});
    content_pattern = content_net.blob_vec(content_net.name2blob_index(content_layer{1})).get_data();
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%style generate net
    style_generate_prototxt = strrep(original_file,'.prototxt','_style_gen.prototxt');
    fid = fopen(style_generate_prototxt,'w');
    proto_txt{1} = 'name: "StyleGen"';
    proto_txt{2} = 'input: "data"';
    proto_txt{3} = 'input_dim: 1';
    proto_txt{4} = 'input_dim: 3';
    proto_txt{5} = ['input_dim: ' num2str(height)];
    proto_txt{6} = ['input_dim: ' num2str(width)];
    for i=1:6
        fprintf(fid,'%s\r\n',proto_txt{i});
    end;
    
    fprintf(fid,'\r\ninput: \"content\"\r\n');
    fprintf(fid,'input_dim: 1\r\n');
    fprintf(fid,'input_dim: %d\r\n',size(content_pattern,3));
    fprintf(fid,'input_dim: %d\r\n',size(content_pattern,2));
    fprintf(fid,'input_dim: %d\r\n',size(content_pattern,1));
    
    for i=1:length(style_layer)
        fprintf(fid,'\r\ninput: \"style_pattern%d\"\r\n',i);
        fprintf(fid,'input_dim: 1\r\n');
        fprintf(fid,'input_dim: %d\r\n',size(style_pattern{i},2));
        fprintf(fid,'input_dim: %d\r\n',size(style_pattern{i},2));
        fprintf(fid,'input_dim: 1\r\n');
    end;
    
    fprintf(fid,'%s\r\n',original_net_model);
    
    for i = 1 : length(style_layer)
        covariance_layer = strrep(covariance_content,'{num}',num2str(i));
        covariance_layer = strrep(covariance_layer,'{bottom_name}',style_layer{i});
        fprintf(fid,'%s\r\n',covariance_layer);
        euclidean_layer = strrep(euclideanloss_content,'{num}',num2str(i));
        euclidean_layer = strrep(euclidean_layer,'{bottom1}',['cov' num2str(i)]);
        euclidean_layer = strrep(euclidean_layer,'{bottom2}',['style_pattern' num2str(i)]);
        euclidean_layer = strrep(euclidean_layer,'{loss_weight}',num2str(style_weights(i)));
        fprintf(fid,'%s\r\n',euclidean_layer);
    end;
    
    euclidean_layer = strrep(euclideanloss_content,'{num}','_content');
    euclidean_layer = strrep(euclidean_layer,'{bottom1}',content_layer{1});
    euclidean_layer = strrep(euclidean_layer,'{bottom2}','content');
    euclidean_layer = strrep(euclidean_layer,'{loss_weight}',num2str(style_weights(end)));
    fprintf(fid,'%s\r\n',euclidean_layer);
    fclose(fid);
    caffe.reset_all();
    
% end

