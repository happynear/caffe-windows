function init_glog(base_dir)
% image_mean = read_mean(mean_file)
%   Read binary proto file to mat

CHECK(ischar(base_dir), 'mean_file must be a string');
caffe_('init_log', base_dir);

end

