#include <vector>

#include "caffe/layers/slope_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void ConcatSlopeForward(const int num, const int channel, const int spatial_dim,
                                     const Dtype* bottom_data, const Dtype* slope_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(n, num) {
      memcpy(top_data + (channel + 2) * spatial_dim * n, bottom_data + channel * spatial_dim * n, sizeof(Dtype) * channel * spatial_dim);
      memcpy(top_data + (channel + 2) * spatial_dim * n + channel * spatial_dim, slope_data, sizeof(Dtype) * 2 * spatial_dim);
    }
  }

  template <typename Dtype>
  __global__ void ConcatSlopeBackward(const int num, const int channel, const int spatial_dim,
                                      const Dtype* top_diff, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(n, num) {
      memcpy(bottom_diff + channel * spatial_dim * n, top_diff + (channel + 2) * spatial_dim * n, sizeof(Dtype) * channel * spatial_dim);
    }
  }

  template <typename Dtype>
  void SlopeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* slope_data = concat_blob_.gpu_data();
    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int spatial_dim = height * width;

    //ConcatSlopeForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    //  << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
    //    num, channel, spatial_dim,
    //    bottom_data, slope_data, top_data);
    for (int n = 0; n < num; n++) {
      caffe_copy(channel * spatial_dim, bottom_data + channel * spatial_dim * n, top_data + (channel + 2) * spatial_dim * n);
      caffe_copy(2 * spatial_dim, slope_data, top_data + (channel + 2) * spatial_dim * n + channel * spatial_dim);
    }
  }

  template <typename Dtype>
  void SlopeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int spatial_dim = height * width;

    //ConcatSlopeBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    //  << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
    //    num, channel, spatial_dim,
    //    top_diff, bottom_diff);

    for (int n = 0; n < num; n++) {
      caffe_copy(channel * spatial_dim, top_diff + (channel + 2) * spatial_dim * n, bottom_diff + channel * spatial_dim * n);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(SlopeLayer);

}  // namespace caffe
