caffe.reset_all();
caffe.set_mode_gpu();
test_net = caffe.Net('inner_distance_layer.prototxt','train');
P1 = single(randn(1,1,100,2));
f = test_net.forward({P1});
ip1_w = test_net.layers('ip1').params(1).get_data();
dis_gt = pdist2(reshape(P1,[100,2])', ip1_w')';
dis_calc = f{1}(:);
abs(mean(abs(dis_calc - dis_gt(:))))

delta = 1e-3;
epsilon = 1e-1;

for o = 1:length(dis_calc)
    f = test_net.forward({P1});
    bp_blob = zeros(size(f{1}));
    bp_blob(o) = 1;
    data_grad = test_net.backward({bp_blob});
    data_grad = squeeze(data_grad{1});
    num_grad = zeros(size(data_grad));
    for i=1:length(P1(:))
       P1p = P1;
       P1m = P1;
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