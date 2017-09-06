#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/channel_scale_layer.hpp"

namespace caffe {

#define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))

template <typename Dtype>
void ChannelScaleLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num() == bottom[1]->num());
  CHECK(bottom[0]->count() / bottom[0]->channels() == bottom[1]->count());
  do_forward_ = this->layer_param_.channel_scale_param().do_forward();
  do_backward_feature_ = this->layer_param_.channel_scale_param().do_backward_feature();
  do_backward_scale_ = this->layer_param_.channel_scale_param().do_backward_scale();
}

template <typename Dtype>
void ChannelScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ChannelScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  if (do_forward_) {
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    int spatial_dim = bottom[0]->height() * bottom[0]->width();
    for (int n = 0; n < num; n++) {
      for (int s = 0; s < spatial_dim; s++) {
        for (int c = 0; c < channels; c++) {
          top_data[(n * channels + c) * spatial_dim + s] = scale_data[n*spatial_dim + s] * bottom_data[(n * channels + c) * spatial_dim + s];
        }
      }
    }
  }
  else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void ChannelScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_diff =bottom[1]->mutable_cpu_diff();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();

  if (propagate_down[0]) {
    if (do_backward_feature_) {
      for (int n = 0; n < num; n++) {
        for (int s = 0; s < spatial_dim; s++) {
          for (int c = 0; c < channels; c++) {
            bottom_diff[(n * channels + c) * spatial_dim + s] = scale_data[n*spatial_dim + s] * top_diff[(n * channels + c) * spatial_dim + s];
          }
        }
      }
    }
    else {
      caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
    }
  }

  caffe_set(num*spatial_dim, Dtype(0), scale_diff);
  if (propagate_down[1] && do_backward_scale_) {
    for (int n = 0; n < num; n++) {
      for (int s = 0; s < spatial_dim; s++) {
        for (int c = 0; c < channels; c++) {
          scale_diff[n*spatial_dim + s] += bottom_data[(n * channels + c) * spatial_dim + s] * top_diff[(n * channels + c) * spatial_dim + s];
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ChannelScaleLayer);
#endif

INSTANTIATE_CLASS(ChannelScaleLayer);
REGISTER_LAYER_CLASS(ChannelScale);

}  // namespace caffe