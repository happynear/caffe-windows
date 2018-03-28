#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

#define sign(x) ((Dtype(0) < (x)) - ((x) < Dtype(0)))

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  normalize_type_ =
    this->layer_param_.normalize_param().normalize_type();
  fix_gradient_ =
    this->layer_param_.normalize_param().fix_gradient();
  bp_norm_ = this->layer_param_.normalize_param().bp_norm() && top.size() == 2;
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                  bottom[0]->height(), bottom[0]->width());
  squared_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  if (top.size() == 2) {
    top[1]->Reshape(bottom[0]->num(), 1,
                    bottom[0]->height(), bottom[0]->width());
  }
  norm_.Reshape(bottom[0]->num(), 1,
                bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* square_data = squared_.mutable_cpu_data();
  Dtype* norm_data = (top.size() == 2) ? top[1]->mutable_cpu_data() : norm_.mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (normalize_type_ == "L2") {
    caffe_sqr<Dtype>(num*channels*spatial_dim, bottom_data, square_data);
    for (int n = 0; n < num; n++) {
      for (int s = 0; s < spatial_dim; s++) {
        norm_data[n*spatial_dim + s] = Dtype(0);
        for (int c = 0; c < channels; c++) {
          norm_data[n*spatial_dim + s] += square_data[(n * channels + c) * spatial_dim + s];
        }
        norm_data[n*spatial_dim + s] += 1e-6;
        norm_data[n*spatial_dim + s] = sqrt(norm_data[n*spatial_dim + s]);
        for (int c = 0; c < channels; c++) {
          top_data[(n * channels + c) * spatial_dim + s] = bottom_data[(n * channels + c) * spatial_dim + s] / norm_data[n*spatial_dim + s];
        }
      }
    }
  }
  else if (normalize_type_ == "L1") {
    caffe_abs<Dtype>(num*channels*spatial_dim, bottom_data, square_data);
    for (int n = 0; n < num; n++) {
      for (int s = 0; s < spatial_dim; s++) {
        norm_data[n*spatial_dim +s] = Dtype(0);
        for (int c = 0; c < channels; c++) {
          norm_data[n*spatial_dim + s] += square_data[(n * channels + c) * spatial_dim + s];
        }
        norm_data[n*spatial_dim + s] += 1e-6;
        norm_data[n*spatial_dim + s] = norm_data[n*spatial_dim + s];
        for (int c = 0; c < channels; c++) {
          top_data[(n * channels + c) * spatial_dim + s] = bottom_data[(n * channels + c) * spatial_dim + s] / norm_data[n*spatial_dim + s];
        }
      }
    }
  }
  else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* square_data = squared_.cpu_data();
  const Dtype* norm_data = (top.size() == 2) ? top[1]->cpu_data() : norm_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (propagate_down[0]) {
    if (normalize_type_ == "L2") {
      for (int n = 0; n < num; ++n) {
        for (int s = 0; s < spatial_dim; s++) {
          Dtype a = caffe_cpu_strided_dot(channels, top_data + n*channels*spatial_dim + s, spatial_dim, top_diff + n*channels*spatial_dim + s, spatial_dim);
          for (int c = 0; c < channels; c++) {
            bottom_diff[(n * channels + c) * spatial_dim + s] =
              (top_diff[(n * channels + c) * spatial_dim + s] - top_data[(n * channels + c) * spatial_dim + s] * a) / norm_data[n*spatial_dim + s];
          }
        }
      }
    }
    else if (normalize_type_ == "L1") {
      for (int n = 0; n < num; ++n) {
        for (int s = 0; s < spatial_dim; s++) {
          Dtype a = caffe_cpu_strided_dot(channels, top_data + n*channels*spatial_dim + s, spatial_dim, top_diff + n*channels*spatial_dim + s, spatial_dim);
          for (int c = 0; c < channels; c++) {
            bottom_diff[(n * channels + c) * spatial_dim + s] =
              (top_diff[(n * channels + c) * spatial_dim + s] - sign(bottom_data[(n * channels + c) * spatial_dim + s]) * a) / norm_data[n*spatial_dim + s];
          }
        }
      }
    }
    else {
      NOT_IMPLEMENTED;
    }
  }

  if (bp_norm_) {
    const Dtype* norm_diff =top[1]->cpu_diff();
    if (normalize_type_ == "L2") {
      for (int n = 0; n < num; ++n) {
        for (int s = 0; s < spatial_dim; s++) {
          for (int c = 0; c < channels; c++) {
            bottom_diff[(n * channels + c) * spatial_dim + s] += norm_diff[n*spatial_dim + s] * top_data[(n * channels + c) * spatial_dim + s];
          }
        }
      }
    }
    else if (normalize_type_ == "L1") {
      for (int n = 0; n < num; ++n) {
        for (int s = 0; s < spatial_dim; s++) {
          for (int c = 0; c < channels; c++) {
            bottom_diff[(n * channels + c) * spatial_dim + s] += norm_diff[n*spatial_dim + s] * sign(bottom_data[(n * channels + c) * spatial_dim + s]);
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
