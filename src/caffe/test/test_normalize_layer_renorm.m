caffe.set_mode_gpu();
test_net = caffe.Net('normalize_layer_renorm.prototxt','train');
random_data = single(randn(1,1,128,10));
f = test_net.forward({random_data});

fprintf('sum(abs(x - x/norm(x)*norm(x)))=%g\n',sum(sum(abs(f{1} - random_data))));
assert(sum(sum(abs(f{1} - random_data))) < 1e-4);

g = test_net.backward({ones(size(f{1}))});
fprintf('sum(abs(x_diff - 1))=%g\n',sum(sum(abs(g{1}-1))));
assert(sum(sum(abs(g{1}-1))) < 1e-4);