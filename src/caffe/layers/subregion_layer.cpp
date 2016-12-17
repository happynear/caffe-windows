#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

template <typename Dtype>
void SubRegionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  SubRegionParameter sub_region_param = this->layer_param_.sub_region_param();
  height_ = sub_region_param.region_height();
  width_ = sub_region_param.region_width();
  data_height_ = sub_region_param.data_height();
  data_width_ = sub_region_param.data_width();
  as_dim_ = sub_region_param.as_dim();
}

template <typename Dtype>
void SubRegionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(bottom[1]->channels() % 2, 0) << "The coordinate blob's size must be able to divided by 2!";
  int num_point = bottom[1]->channels() / 2;
  if (as_dim_ == 0) {
    top[0]->Reshape({ bottom[0]->num() * num_point, bottom[0]->channels(), height_, width_ });
  }
  else {
    top[0]->Reshape({ bottom[0]->num(), bottom[0]->channels() * num_point, height_, width_ });
  }
  if (top.size() == 3) {
    top[1]->ReshapeLike(*bottom[1]);
    top[2]->ReshapeLike(*bottom[1]);
  }
}

template <typename Dtype>
void SubRegionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* point_data = bottom[1]->cpu_data();
  const Dtype* ground_truth_point_data = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const int num_point = bottom[1]->shape(1) / 2;
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  for (int n = 0; n < num; n++) {
    for (int i = 0; i < num_point; i++) {
      Dtype center_x = (point_data[n * bottom[1]->shape(1) + i * 2] / data_width_ + 0.5)  * bottom[0]->width();
      Dtype center_y = (point_data[n * bottom[1]->shape(1) + i * 2 + 1] / data_height_ + 0.5) * bottom[0]->height();
      int x0 = floor(center_x - width_ / 2);
      int y0 = floor(center_y - height_ / 2);
      if (top.size() == 3) {
        top[2]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2] = (Dtype)(x0 + (Dtype)width_ / 2 + 0.5) / (Dtype)bottom[0]->width() * data_width_;
        top[2]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2 + 1] = (Dtype)(y0 + (Dtype)height_ / 2 + 0.5) / (Dtype)bottom[0]->height() * data_height_;
        top[1]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2] = 
          (ground_truth_point_data[n * bottom[1]->shape(1) + i * 2] + data_width_ / 2) - top[2]->cpu_data()[n * bottom[1]->shape(1) + i * 2];
        top[1]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2 + 1] = 
          (ground_truth_point_data[n * bottom[1]->shape(1) + i * 2 + 1] + data_height_ / 2) - top[2]->cpu_data()[n * bottom[1]->shape(1) + i * 2 + 1];

      }

      for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height_; h++) {
          for (int w = 0; w < width_; w++) {
            if (y0 + h >= 0 && y0 + h <= bottom[0]->height() - 1
                && x0 + w >= 0 && x0 + w <= bottom[0]->width() - 1) {
              if (as_dim_ == 0) {
                top_data[top[0]->offset(num * i + n, c, h, w)] = bottom_data[bottom[0]->offset(n, c, y0 + h, x0 + w)];
              }
              else {
                top_data[top[0]->offset(n, channels * i + c, h, w)] = bottom_data[bottom[0]->offset(n, c, y0 + h, x0 + w)];
              }
            }
            else {
              if (as_dim_ == 0) {
                top_data[top[0]->offset(num * i + n, c, h, w)] = 0;
              }
              else {
                top_data[top[0]->offset(n, channels * i + c, h, w)] = 0;
              }
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void SubRegionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* point_data = bottom[1]->cpu_data();
  Dtype* bottom_diff = top[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  const int num_point = bottom[1]->shape(1) / 2;
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  for (int n = 0; n < num; n++) {
    for (int i = 0; i < num_point; i++) {
      Dtype center_x = (point_data[n * bottom[1]->shape(1) + i * 2] / data_width_ + 0.5)  * bottom[0]->width();
      Dtype center_y = (point_data[n * bottom[1]->shape(1) + i * 2 + 1] / data_height_ + 0.5) * bottom[0]->height();
      int x0 = floor(center_x - width_ / 2);
      int y0 = floor(center_y - height_ / 2);

      for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height_; h++) {
          for (int w = 0; w < width_; w++) {
            if (y0 + h >= 0 && y0 + h <= bottom[0]->height() - 1
                && x0 + w >= 0 && x0 + w <= bottom[0]->width() - 1) {
              if (as_dim_ == 0) {
                bottom_diff[bottom[0]->offset(n, c, y0 + h, x0 + w)] += top_diff[top[0]->offset(num * i + n, c, h, w)];
              }
              else {
                bottom_diff[bottom[0]->offset(n, c, y0 + h, x0 + w)] += top_diff[top[0]->offset(n, channels * i + c, h, w)];
              }
            }
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SubRegionLayer);
#endif

INSTANTIATE_CLASS(SubRegionLayer);
REGISTER_LAYER_CLASS(SubRegion);

}  // namespace caffe
