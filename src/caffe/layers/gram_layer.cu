#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {
  template <typename Dtype>
  void GramLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int num = bottom[0]->shape(0);
    int channel = bottom[0]->shape(1);
    int spatial_dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    
    for (int n=0; n < num; n++){
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel, channel, spatial_dim,
        1 / (Dtype)spatial_dim / (Dtype)channel, bottom_data + n * spatial_dim * channel, bottom_data + n * spatial_dim * channel, Dtype(0), top_data + n * channel * channel);
    }
  }

  template <typename Dtype>
  void GramLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int num = bottom[0]->shape(0);
    int channel = bottom[0]->shape(1);
    int spatial_dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    
    for (int n=0; n < num; n++){
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channel, spatial_dim, channel,
        1 / (Dtype)spatial_dim / (Dtype)channel, top_diff + n * channel * channel, bottom_data + n * spatial_dim * channel,
                            Dtype(0), bottom_diff + n * spatial_dim * channel);
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channel, spatial_dim, channel,
                            1 / (Dtype)spatial_dim / (Dtype)channel, top_diff + n * channel * channel, bottom_data + n * spatial_dim * channel,
                            Dtype(1), bottom_diff + n * spatial_dim * channel);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(GramLayer);


}  // namespace caffe
