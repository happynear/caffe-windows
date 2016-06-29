#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void TransformPoint(const int num, const int bottom_height, const int bottom_width,
                                 const int target_height, const int target_width, const int num_point,
                                 const int data_height, const int data_width,
                                 const Dtype*  point_data, const Dtype* ground_truth,
                                 Dtype* region_bias, Dtype* ground_truth_bias) {
    CUDA_KERNEL_LOOP(index, num * num_point) {
      int n = index / num_point;
      int p = index % num_point;
      Dtype center_x = (point_data[n * num_point * 2 + p * 2] / data_width + 0.5)  * bottom_width;
      Dtype center_y = (point_data[n * num_point * 2 + p * 2 + 1] / data_height + 0.5) * bottom_height;
      int x0 = floor(center_x - target_width / 2);
      int y0 = floor(center_y - target_height / 2);
      region_bias[n * num_point * 2 + p * 2] = (Dtype)(x0 + (Dtype)target_width / 2 + 0.5) / (Dtype)bottom_width * data_width;
      region_bias[n * num_point * 2 + p * 2 + 1] = (Dtype)(y0 + (Dtype)target_height / 2 + 0.5) / (Dtype)bottom_height * data_height;
      ground_truth_bias[n * num_point * 2 + p * 2] = ground_truth[n * num_point * 2 + p * 2] + data_width / 2 - region_bias[n * num_point * 2 + p * 2];
      ground_truth_bias[n * num_point * 2 + p * 2 + 1] = ground_truth[n * num_point * 2 + p * 2 + 1] + data_height / 2 - region_bias[n * num_point * 2 + p * 2 + 1];
    }
  }

  template <typename Dtype>
  __global__ void SubRegionFoward(const int num, const int channels, const int num_point,
                                  const int bottom_height, const int bottom_width, bool as_dim,
                                  const int data_height, const int data_width,
                                  const int target_height, const int target_width,
                                  const Dtype* bottom_data, const Dtype*  point_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, num * channels * target_height * target_width * num_point) {
      int n = index / (channels * target_height * target_width * num_point);
      int csp = index % (channels * target_height * target_width * num_point);
      int c = csp / (target_height * target_width * num_point);
      int s = csp % (target_height * target_width * num_point);
      int h = s / (target_width * num_point);
      int pw = s % (target_width * num_point);
      int w = pw / num_point;
      int p = pw % num_point;
      Dtype center_x = (point_data[n * num_point * 2 + p * 2] / data_width + 0.5)  * bottom_width;
      Dtype center_y = (point_data[n * num_point * 2 + p * 2 + 1] / data_height + 0.5) * bottom_height;
      int x0 = floor(center_x - target_width / 2);
      int y0 = floor(center_y - target_height / 2);
      if (y0 + h >= 0 && y0 + h <= bottom_height - 1
          && x0 + w >= 0 && x0 + w <= bottom_width - 1) {
        if (as_dim == 0) {
          top_data[((((num * p + n) * channels + c) * target_height + h) * target_width + w)] = bottom_data[(((n*channels + c)*bottom_height + y0 + h)*bottom_width + x0 + w)];
        }
        else {
          top_data[((((n * num_point + p) * channels + c) * target_height + h) * target_width + w)] = bottom_data[(((n*channels + c)*bottom_height + y0 + h)*bottom_width + x0 + w)];
        }
      }
      else {
        if (as_dim == 0) {
          top_data[((((num * p + n) * channels + c) * target_height + h) * target_width + w)] = 0;
        }
        else {
          top_data[((((n * num_point + p) * channels + c) * target_height + h) * target_width + w)] = 0;
        }
      }
    }
  }

  template <typename Dtype>
  __global__ void SubRegionBackward(const int num, const int channels, const int num_point,
                                    const int bottom_height, const int bottom_width, bool as_dim,
                                    const int data_height, const int data_width,
                                    const int target_height, const int target_width,
                                    Dtype* bottom_diff, const Dtype*  point_data, const Dtype* top_diff);

  template <>
  __global__ void SubRegionBackward<float>(const int num, const int channels, const int num_point,
                                  const int bottom_height, const int bottom_width, bool as_dim,
                                  const int data_height, const int data_width,
                                  const int target_height, const int target_width,
                                  float* bottom_diff, const float*  point_data, const float* top_diff) {
    CUDA_KERNEL_LOOP(index, num * channels * target_height * target_width * num_point) {
      int n = index / (channels * target_height * target_width * num_point);
      int csp = index % (channels * target_height * target_width * num_point);
      int c = csp / (target_height * target_width * num_point);
      int s = csp % (target_height * target_width * num_point);
      int h = s / (target_width * num_point);
      int pw = s % (target_width * num_point);
      int w = pw / num_point;
      int p = pw % num_point;
      float center_x = (point_data[n * num_point * 2 + p * 2] / data_width + 0.5)  * bottom_width;
      float center_y = (point_data[n * num_point * 2 + p * 2 + 1] / data_height + 0.5) * bottom_height;
      int x0 = floor(center_x - target_width / 2);
      int y0 = floor(center_y - target_height / 2);
      if (y0 + h >= 0 && y0 + h <= bottom_height - 1
          && x0 + w >= 0 && x0 + w <= bottom_width - 1) {
        if (as_dim == 0) {
          atomicAdd(&bottom_diff[(((n*channels + c)*bottom_height + y0 + h)*bottom_width + x0 + w)], top_diff[((((num * p + n) * channels + c) * target_height + h) * target_width + w)]);
        }
        else {
          atomicAdd(&bottom_diff[(((n*channels + c)*bottom_height + y0 + h)*bottom_width + x0 + w)], top_diff[((((n * num_point + p) * channels + c) * target_height + h) * target_width + w)]);
        }
      }
    }
  }

  template <>
  __global__ void SubRegionBackward<double>(const int num, const int channels, const int num_point,
                                    const int bottom_height, const int bottom_width, bool as_dim,
                                    const int data_height, const int data_width,
                                    const int target_height, const int target_width,
                                    double* bottom_diff, const double*  point_data, const double* top_diff) {
    CUDA_KERNEL_LOOP(index, num * channels * target_height * target_width * num_point) {
      //NOT_IMPLEMENTED
    }
  }

template <typename Dtype>
void SubRegionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* point_data = bottom[1]->gpu_data();
  const Dtype* ground_truth_point_data = bottom[2]->gpu_data();
  Dtype* ground_truth_bias = top[1]->mutable_gpu_data();
  Dtype* region_bias = top[2]->mutable_gpu_data();
  int count = top[0]->count();
  const int num_point = bottom[1]->shape(1) / 2;
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int target_height = top[0]->height();
  const int target_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  TransformPoint<Dtype> << <CAFFE_GET_BLOCKS(num * num_point),
    CAFFE_CUDA_NUM_THREADS >> >(num, bottom_height, bottom_width,
    target_height, target_width, num_point,
    data_height_, data_width_,
    point_data, ground_truth_point_data,
    region_bias, ground_truth_bias);
  CUDA_POST_KERNEL_CHECK;
  
  SubRegionFoward<Dtype> << <CAFFE_GET_BLOCKS(num * channels * target_height * target_width * num_point),
    CAFFE_CUDA_NUM_THREADS >> >(num, channels, num_point,
    bottom_height, bottom_width, as_dim_,
    data_height_, data_width_,
    target_height, target_width,
    bottom_data, point_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SubRegionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* point_data = bottom[1]->gpu_data();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int num_point = bottom[1]->shape(1) / 2;
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int target_height = top[0]->height();
  const int target_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();

  SubRegionBackward<Dtype> << <CAFFE_GET_BLOCKS(num * channels * target_height * target_width * num_point),
    CAFFE_CUDA_NUM_THREADS >> >(num, channels, num_point,
    bottom_height, bottom_width, as_dim_,
    data_height_, data_width_,
    target_height, target_width,
    bottom_diff, point_data, top_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(SubRegionLayer);


}  // namespace caffe
