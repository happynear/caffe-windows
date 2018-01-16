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
  global_scale_ = this->layer_param_.channel_scale_param().global_scale();
  max_global_scale_ = this->layer_param_.channel_scale_param().max_global_scale();
  min_global_scale_ = this->layer_param_.channel_scale_param().min_global_scale();
  if (global_scale_) {
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    }
    else {
      this->blobs_.resize(1);
      this->blobs_[0].reset(new Blob<Dtype>({ 1 }));
      this->blobs_[0]->mutable_cpu_data()[0] = -1;
    }
    if (this->layer_param_.param_size() == 0) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
      fixed_param_spec->set_decay_mult(0.f);
    }
    else {
      CHECK_EQ(this->layer_param_.param(0).lr_mult(), 0.f)
        << "Cannot configure statistics as layer parameters.";
    }
  }
}

template <typename Dtype>
void ChannelScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  if (global_scale_ && top.size() == 2) {
    top[1]->Reshape({ 1 });
  }
}

template <typename Dtype>
void ChannelScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  if (do_forward_) {
    if (global_scale_) {
      int count = bottom[0]->count();
      Dtype* scale = this->blobs_[0]->mutable_cpu_data();
      scale[0] = std::min(scale[0], max_global_scale_);
      scale[0] = std::max(scale[0], min_global_scale_);
      if (top.size() == 2) {
        top[1]->mutable_cpu_data()[0] = scale[0];
      }
      caffe_cpu_scale(count, scale[0], bottom_data, top_data);
    }
    else {
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

  //if (global_scale_ && this->param_propagate_down_[0]) {
  //  this->blobs_[0]->mutable_cpu_diff()[0] = caffe_cpu_dot(bottom[0]->count(), top_diff, bottom_data);
  //}
}


#ifdef CPU_ONLY
STUB_GPU(ChannelScaleLayer);
#endif

INSTANTIATE_CLASS(ChannelScaleLayer);
REGISTER_LAYER_CLASS(ChannelScale);

}  // namespace caffe