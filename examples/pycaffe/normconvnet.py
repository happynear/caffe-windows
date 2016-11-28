from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

# helper function for common structures

sigmoid_scale = 0.3144#0.3673#0.2084
sigmoid_bias = 0.1837#0.1042

def ip_factory(bottom, nout):
    ip = L.InnerProduct(bottom, num_output=nout, normalize_scale=2.0, weight_filler=dict(type='xavier'))
    sigmoid = L.Sigmoid(ip, in_place = True);
    scale = L.Scale(sigmoid, bias_term=True, 
                    filler=dict(type='constant',value= 1.0 / sigmoid_scale),bias_filler=dict(type='constant', value=-0.5 / sigmoid_scale),
                    param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]);
    return scale

def conv_factory(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, normalize_scale=2.0,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    sigmoid = L.Sigmoid(conv, in_place = True);
    scale = L.Scale(sigmoid, bias_term=True, 
                    filler=dict(type='constant',value=1.0 / sigmoid_scale),bias_filler=dict(type='constant', value=-0.5 / sigmoid_scale),
                    param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]);
    return scale

def max_pool_3x3(bottom, stride=1):
    pool= L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=3, stride=stride)
    scale = L.Scale(pool, bias_term=True, 
                    filler=dict(type='constant',value=1.6725),bias_filler=dict(type='constant', value=-2.4834),
                    param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]);
    return scale

def max_pool_2x2(bottom, stride=1):
    pool= L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=2, stride=stride)
    scale = L.Scale(pool, bias_term=True, 
                    filler=dict(type='constant',value=1.4263),bias_filler=dict(type='constant', value=-1.4675),
                    param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]);
    return scale

def avg_pool(bottom, ks, stride=1):
    pool= L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)
    scale = L.Scale(pool, filler=dict(type='constant',value=ks),
                    param=[dict(lr_mult=0, decay_mult=0)]);
    return scale

def SimpleFactory(bottom, ch1x1, ch3x3):
    conv1x1 = conv_factory(bottom, 1, ch1x1, 1, 0)
    conv3x3 = conv_factory(bottom, 3, ch3x3, 1, 1)
    concat = L.Concat(conv1x1, conv3x3)
    return concat

def DownsampleFactory(bottom, ch3x3):
    conv3x3 = conv_factory(bottom, 3, ch3x3, 2, 1)
    pool = max_pool_3x3(bottom, 2)
    concat = L.Concat(conv3x3, pool)
    return concat

def normnet(train_lmdb, test_lmdb, batch_size=256, stages=[2, 2, 2, 2], input_size=32, first_output=32, include_acc=False):
    # now, this code can't recognize include phase, so there will only be a TEST phase data layer
    data, label = L.Data(source=train_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TEST')))
    data, label = L.MemoryData(batch_size=batch_size, height=input_size, width=input_size, channels=3, ntop=2,
        transform_param=dict(mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TEST')))

    # the net itself
    
    conv1 = conv_factory(data, 3, 96, 1, 1)
    in3a = SimpleFactory(conv1, 32, 32)
    in3b = SimpleFactory(in3a, 32, 48)
    in3c = DownsampleFactory(in3b, 80)
    in4a = SimpleFactory(in3c, 112, 48)
    in4b = SimpleFactory(in4a, 96, 64)
    in4c = SimpleFactory(in4b, 80, 80)
    in4d = SimpleFactory(in4c, 48, 96)
#    for i in range(25):
#        in4d = SimpleFactory(in4d, 48, 96)
    in4e = DownsampleFactory(in4d, 96)
    in5a = SimpleFactory(in4e, 176, 160)
    in5b = SimpleFactory(in5a, 176, 160)

    pool = avg_pool(in5b, 8)

    fc = L.InnerProduct(pool, num_output=10, weight_filler=dict(type='xavier'))
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)

def make_net(output_file):
    with open(output_file, 'w') as f:
        print(normnet('/path/to/caffe-train-lmdb', '/path/to/caffe-val-lmdb'), file=f)

if __name__ == '__main__':
    train_test_file = 'normconvnet_train_test.prototxt'
    make_net(train_test_file)
    caffe.Net(train_test_file, caffe.TEST)  # test loading the net
