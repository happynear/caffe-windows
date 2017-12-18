from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# helper function for common structures

def conv_factory(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

def Conv(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def Inception7A(data, num_1x1, num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5, pool, proj):
    tower_1x1 = Conv(data, 1, num_1x1)
    tower_5x5 = Conv(data, 1, num_5x5_red)
    tower_5x5 = Conv(tower_5x5, 5, num_5x5, 1, 2)
    tower_3x3 = Conv(data, 1, num_3x3_red)
    tower_3x3 = Conv(tower_3x3, 3, num_3x3_1, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_2')
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(pooling, proj, name=('%s_tower_2' %  name), suffix='_conv')
    concat = mx.sym.Concat(*[tower_1x1, tower_5x5, tower_3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def residual_factory1(bottom, num_filter):
    conv1 = conv_factory_relu(bottom, 3, num_filter, 1, 1);
    conv2 = conv_factory(conv1, 3, num_filter, 1, 1);
    residual = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    relu = L.ReLU(residual, in_place=True)
    return relu

def residual_factory_proj(bottom, num_filter, stride=2):
    conv1 = conv_factory_relu(bottom, 3, num_filter, stride, 1);
    conv2 = conv_factory(conv1, 3, num_filter, 1, 1);
    proj = conv_factory(bottom, 1, num_filter, stride, 0);
    residual = L.Eltwise(conv2, proj, operation=P.Eltwise.SUM)
    relu = L.ReLU(residual, in_place=True)
    return relu

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def resnet(train_lmdb, test_lmdb, batch_size=256, stages=[2, 2, 2, 2], first_output=32, include_acc=False):
    # now, this code can't recognize include phase, so there will only be a TEST phase data layer
    data, label = L.Data(source=train_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TEST')))

    # the net itself
    relu1 = conv_factory_relu(data, 3, first_output, stride=1, pad=1)
    relu2 = conv_factory_relu(relu1, 3, first_output, stride=1, pad=1)
    residual = max_pool(relu2, 3, stride=2)
    
    for i in stages[1:]:
        first_output *= 2
        for j in range(i):
            if j==0:
                if i==0:
                    residual = residual_factory_proj(residual, first_output, 1)
                else:
                    residual = residual_factory_proj(residual, first_output, 2)
            else:
                residual = residual_factory1(residual, first_output)

    glb_pool = L.Pooling(residual, pool=P.Pooling.AVE, global_pooling=True);
    fc = L.InnerProduct(glb_pool, num_output=1000)
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)

def make_net():
    with open('residual_train_test.prototxt', 'w') as f:
        print(resnet('/path/to/caffe-train-lmdb', '/path/to/caffe-val-lmdb'), file=f)

if __name__ == '__main__':
    make_net()
