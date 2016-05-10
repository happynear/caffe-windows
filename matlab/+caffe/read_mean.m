function image_mean = read_mean(mean_file)
% image_mean = read_mean(mean_file)
%   Read binary proto file to mat

CHECK(ischar(mean_file), 'mean_file must be a string');
CHECK_FILE_EXIST(mean_file);
image_mean = caffe_('read_mean', mean_file);

end

