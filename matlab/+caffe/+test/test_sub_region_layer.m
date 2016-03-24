caffe.reset_all();
caffe.set_mode_gpu();
net = caffe.Net('sub_region_pure.prototxt', 'test');

net.forward_prefilled();
image_blob = net.blob_vec(1);
image_data = image_blob.get_data();
landmark_blob = net.blobs('landmark_label');
landmark_data = landmark_blob.get_data();
% landmark_ip2 = net.blobs('landmark_ip2').get_data();
sub_regions = net.blobs('sub_region').get_data();
ground_offset = net.blobs('ground_offset').get_data();
region_offset = net.blobs('region_offset').get_data();
check_landmark = region_offset + ground_offset;
% sub_landmark_ip2 = net.blobs('sub_landmark_ip2').get_data();
% sub_conv41_bn = net.blobs('sub_conv41_bn').get_data();
% inception_3b = net.blobs('inception_3b/output').get_data();


for i=14:14%size(image_data,4)
    image_slice = uint8(image_data(:,:,:,i)+128);
    image_slice = image_slice(:, :, [3, 2, 1]);
    image_slice = permute(image_slice, [2 1 3]);
%     image_slice = image_slice';
%     image_slice = repmat(image_slice,[1 1 3]);
    figure(1);
    image(image_slice);
    hold on;
    disp(landmark_data(:,i));
    for j=1:5
        plot(landmark_data(j*2-1,i)+30+0.5, landmark_data(j*2,i)+30+0.5,'r.');
        plot(check_landmark(j*2-1,i)+0.5, check_landmark(j*2,i)+0.5,'gx');
%         plot(landmark_ip2(j*2-1,i)+30, landmark_ip2(j*2,i)+30,'gx');
    end;
    for j=1:5
        figure(j+1);
        sub_image = uint8(sub_regions(:,:,3*(j-1)+(1:3),i)+128);
        sub_image = sub_image(:, :, [3, 2, 1]);
        sub_image = permute(sub_image, [2 1 3]);
        imshow(imresize(sub_image,4,'nearest'));
    end;
end;