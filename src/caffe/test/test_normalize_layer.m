caffe.reset_all();
caffe.set_mode_gpu();
test_net = caffe.Net('normalize_layer.prototxt','train');
random_data = single(randn(16,16,512,1));
f = test_net.forward({random_data});
normed_data = reshape(random_data, [9,100]);
normed_data = bsxfun(@rdivide, normed_data, sqrt(sum(normed_data.^2,2)));
normed_data_calc = f{1}(:);
abs(mean(abs(normed_data(:) - normed_data_calc(:))))

delta = 1e-4;
epsilon = 1e-3;

for o = 4:4%length(normed_data_calc)
    f = test_net.forward({random_data});
    bp_blob = zeros(size(f{1}));
    bp_blob(o) = 1;
    data_grad = test_net.backward({bp_blob});
    data_grad = squeeze(data_grad{1});
    num_grad = zeros(size(data_grad));
    for i=1:length(random_data(:))
       P1p = random_data;
       P1m = random_data;
       P1p(i) = P1p(i) + delta;
       P1m(i) = P1m(i) - delta;
       fp = test_net.forward({P1p});
       fm = test_net.forward({P1m});
       numerical_grad = (fp{1}(o) - fm{1}(o)) / 2 / delta;
       num_grad(i) = numerical_grad;
       calculated_grad = data_grad(i);
       assert(abs(calculated_grad - numerical_grad) < epsilon)
    end;
end;
data_grad = reshape(data_grad,[9,100]);
num_grad = reshape(num_grad,[9,100]);