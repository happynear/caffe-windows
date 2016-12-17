caffe.reset_all();
caffe.set_mode_gpu();
net = caffe.Net('hotspot_pure.prototxt', 'test');

net.forward_prefilled();
image_blob = net.blob_vec(1);
image_data = image_blob.get_data();
landmark_blob = net.blobs('landmark_label');
landmark_data = landmark_blob.get_data();
hot_blob = net.blobs('hotspot');
hot_data = hot_blob.get_data();
size(hot_data)
figure(1);
for i = 1:5
    subplot(5,1,i);
    imshow(hot_data(:,:,i,1)');
end;
figure(2);
image(uint8(permute(squeeze(image_data(:,:,:,1))+128,[2 1 3])));
hold on;
disp(landmark_data(:,1));
for j=1:5
    plot(landmark_data(j*2-1,1)+30+0.5, landmark_data(j*2,1)+30+0.5,'r.');
end;
figure(3);
imshow(imresize(sum(hot_data(:,:,:,1),3)',4));