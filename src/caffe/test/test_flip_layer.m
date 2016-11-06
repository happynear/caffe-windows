caffe.set_mode_gpu();
test_net = caffe.Net('flip_layer.prototxt','test');
f = test_net.forward({});
data = test_net.blobs('data').get_data();
flip_data = test_net.blobs('flip_data').get_data();

subplot(1,2,1);
data1 = permute(data(:,:,:,1),[2 1 3]);
imshow(uint8(data1(:,:,[3 2 1])));
subplot(1,2,2);
flip_data1 = permute(flip_data(:,:,:,1),[2 1 3]);
imshow(uint8(flip_data1(:,:,[3 2 1])));