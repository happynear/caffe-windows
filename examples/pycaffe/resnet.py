from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

# helper function for common structures

def conv_factory(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

def conv_factory_relu(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def conv_factory_relu_inverse(bottom, ks, nout, stride=1, pad=0):
    batch_norm = L.BatchNorm(bottom, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, weight_filler=dict(type='msra'))
    return conv

def conv_factory_relu_inverse_no_inplace(bottom, ks, nout, stride=1, pad=0):
    batch_norm = L.BatchNorm(bottom, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, weight_filler=dict(type='msra'))
    return conv

def residual_factory1(bottom, num_filter):
    conv1 = conv_factory_relu_inverse_no_inplace(bottom, 3, num_filter, 1, 1)
    conv2 = conv_factory_relu_inverse(conv1, 3, num_filter, 1, 1)
    addition = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    return addition

def residual_factory_proj(bottom, num_filter, stride=2):
    batch_norm = L.BatchNorm(bottom, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    conv1 = conv_factory_relu(scale, 3, num_filter, stride, 1)
    conv2 = L.Convolution(conv1, kernel_size=3, stride=1,
                                num_output=num_filter, pad=1, weight_filler=dict(type='msra'))
    proj = L.Convolution(scale, kernel_size=1, stride=stride,
                                num_output=num_filter, pad=0, weight_filler=dict(type='msra'))
    addition = L.Eltwise(conv2, proj, operation=P.Eltwise.SUM)
    return addition

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def resnet(train_lmdb, test_lmdb, batch_size=256, stages=[2, 2, 2, 2], input_size=128, first_output=32, include_acc=False):
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
    caffe.Net('residual_train_test.prototxt', caffe.TEST)  # test loading the net
