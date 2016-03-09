inited = true;
if inited
    caffe.reset_all();
    caffe.set_mode_gpu();
    align_net = caffe.Net('CASIA_train_test_landmark_nobn.prototxt', 'landmark_without_bn.caffemodel', 'train');
end;

align_net.forward_prefilled();
image_blob = align_net.blob_vec(1);
image_data = image_blob.get_data();
landmark_blob = align_net.blob_vec(3);
landmark_data = landmark_blob.get_data();
attribute_blob = align_net.blob_vec(4);
attribute_data = attribute_blob.get_data();
landmark_ip2 = align_net.blobs('landmark_ip2').get_data();

for i=14:14%size(image_data,4)
    image_slice = uint8(image_data(:,:,:,i)+128);
%     image_slice = image_slice(:, :, [3, 2, 1]);
%     image_slice = permute(image_slice, [2 1 3]);
    image_slice = image_slice';
    image_slice = repmat(image_slice,[1 1 3]);
    figure(1);
    image(image_slice);
    hold on;
    disp(landmark_data(:,i));
    for j=1:5
        plot(landmark_data(j*2-1,i)+30, landmark_data(j*2,i)+30,'r.');
        plot(landmark_ip2(j*2-1,i)+30, landmark_ip2(j*2,i)+30,'gx');
    end;
    [res, eyec2, cropped, resize_scale] = align_face(image_slice, reshape(landmark_ip2(:,i), 2, 5)'+30, 128, 48, 40);
    figure(2);
    imshow(cropped);
end;