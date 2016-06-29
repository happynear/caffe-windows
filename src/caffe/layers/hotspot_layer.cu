#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/hotspot_layer.hpp"
#define CV_PI 3.1415926535897932384626433832795
#define GAUSSIAN(x0,y0,x,y) 0.5 / gaussian_std / gaussian_std / CV_PI * exp(-0.5 * (((x0)-(x)) * ((x0)-(x)) + ((y0)-(y)) * ((y0)-(y))) / gaussian_std / gaussian_std)

namespace caffe {

  __device__ __constant__  float kEps= 1e-4;

  template <typename Dtype>
  __global__ void HotspotFoward(const int num, const int num_point, const Dtype gaussian_std,
                                const int data_height, const int data_width, const bool mean_removed,
                                  const int target_height, const int target_width,
                                  const Dtype*  point_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, num * target_height * target_width * num_point) {
      int n = index / (target_height * target_width * num_point);
      int sp = index % (target_height * target_width * num_point);
      int h = sp / (target_width * num_point);
      int pw = sp % (target_width * num_point);
      int w = pw / num_point;
      int p = pw % num_point;
      Dtype p1 = (point_data[n * num_point * 2 + p * 2] / data_width + (mean_removed ? 0.5 : 0))  * target_width;
      Dtype p2 = (point_data[n * num_point * 2 + p * 2 + 1] / data_height + (mean_removed ? 0.5 : 0)) * target_height;
      Dtype temp = GAUSSIAN(p1, p2, w, h);
      if (temp > kEps) {
        top_data[(((n * num_point + p) * target_height + h) * target_width + w)] = temp;
      }
      else {
        top_data[(((n * num_point + p) * target_height + h) * target_width + w)] = 0;
      }
    }
  }

template <typename Dtype>
void HotspotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* point_data = bottom[0]->gpu_data();
  const int num_point = bottom[0]->shape(1) / 2;
  const int num = bottom[0]->num();

  HotspotFoward<Dtype> << <CAFFE_GET_BLOCKS(num * num_point * height_ * width_),
  CAFFE_CUDA_NUM_THREADS >> >(num, num_point, gaussian_std_,
    data_height_, data_width_, mean_removed_,
    height_, width_,
    point_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void HotspotLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


INSTANTIATE_LAYER_GPU_FUNCS(HotspotLayer);


}  // namespace caffe
