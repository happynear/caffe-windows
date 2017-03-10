#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/predict_box_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void PredictBoxForward(const int num, const int spatial_dim, const int height, const int width,
                                    const Dtype* score_data, Dtype* bb_data, Dtype positive_thresh_,
                                    int stride_, int receptive_field_, Dtype* counter_,
                                    bool bounding_box_regression_, const Dtype* bbr_data, bool bounding_box_exp_,
                                    bool use_stitch_, const Dtype* stitch_data) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      int h = s / width;
      int w = s % width;
      if (score_data[((n * 2 + 1) * height + h) * width + w] > positive_thresh_ &&
          score_data[((n * 2 + 1) * height + h) * width + w] < 1 + 1e-6 &&
          !(use_stitch_ && stitch_data[((n * 2 + 2) * height + h) * width + w] == 0)) {
        Dtype bias_x = use_stitch_ ? stitch_data[((n * 2 + 0) * height + h) * width + w] : 0;
        Dtype bias_y = use_stitch_ ? stitch_data[((n * 2 + 1) * height + h) * width + w] : 0;
        Dtype real_receptive_field = use_stitch_ ? stitch_data[((n * 2 + 2) * height + h) * width + w] : receptive_field_;
        bb_data[((n * 5 + 0) * height + h) * width + w] = (Dtype(w * stride_) - bias_x) / Dtype(12) * real_receptive_field;
        bb_data[((n * 5 + 1) * height + h) * width + w] = (Dtype(h * stride_) - bias_y) / Dtype(12) * real_receptive_field;
        bb_data[((n * 5 + 2) * height + h) * width + w] = real_receptive_field;
        bb_data[((n * 5 + 3) * height + h) * width + w] = real_receptive_field;
        bb_data[((n * 5 + 4) * height + h) * width + w] = score_data[((n * 2 + 1) * height + h) * width + w];
        if (bounding_box_regression_) {
          if (bounding_box_exp_) {
            bb_data[((n * 5 + 0) * height + h) * width + w] += bbr_data[((n * 4 + 0) * height + h) * width + w] * real_receptive_field;
            bb_data[((n * 5 + 1) * height + h) * width + w] += bbr_data[((n * 4 + 1) * height + h) * width + w] * real_receptive_field;
            bb_data[((n * 5 + 2) * height + h) * width + w] *= exp(bbr_data[((n * 4 + 2) * height + h) * width + w]);
            bb_data[((n * 5 + 3) * height + h) * width + w] *= exp(bbr_data[((n * 4 + 3) * height + h) * width + w]);
          }
          else {
            bb_data[((n * 5 + 0) * height + h) * width + w] += bbr_data[((n * 4 + 1) * height + h) * width + w] * real_receptive_field;
            bb_data[((n * 5 + 1) * height + h) * width + w] += bbr_data[((n * 4 + 0) * height + h) * width + w] * real_receptive_field;
            bb_data[((n * 5 + 2) * height + h) * width + w] +=
              (bbr_data[((n * 4 + 3) * height + h) * width + w] - bbr_data[((n * 4 + 1) * height + h) * width + w]) * real_receptive_field;
            bb_data[((n * 5 + 3) * height + h) * width + w] +=
              (bbr_data[((n * 4 + 2) * height + h) * width + w] - bbr_data[((n * 4 + 0) * height + h) * width + w]) * real_receptive_field;
          }
        }
        counter_[(n * height + h) * width + w] = 1;
      }
      else {
        bb_data[((n * 5 + 0) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 1) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 2) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 3) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 4) * height + h) * width + w] = 0;
        counter_[(n * height + h) * width + w] = 0;
      }
    }
  }

  template <typename Dtype>
  __global__ void PredictBoxForwardWithNMS(const int num, const int spatial_dim, const int height, const int width,
                                           const Dtype* score_data, Dtype* bb_data, Dtype positive_thresh_,
                                           int stride_, int receptive_field_, Dtype* counter_, const Dtype* nms_data,
                                           bool bounding_box_regression_, const Dtype* bbr_data, bool bounding_box_exp_) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      int h = s / width;
      int w = s % width;
      if (score_data[((n * 2 + 1) * height + h) * width + w] > positive_thresh_ &&
          score_data[((n * 2 + 1) * height + h) * width + w] < 1 + 1e-6 &&
          score_data[((n * 2 + 1) * height + h) * width + w] > nms_data[((n * 2 + 1) * height + h) * width + w] - 1e-6) {
        bb_data[((n * 5 + 0) * height + h) * width + w] = w * stride_;
        bb_data[((n * 5 + 1) * height + h) * width + w] = h * stride_;
        bb_data[((n * 5 + 2) * height + h) * width + w] = receptive_field_;
        bb_data[((n * 5 + 3) * height + h) * width + w] = receptive_field_;
        bb_data[((n * 5 + 4) * height + h) * width + w] = score_data[((n * 2 + 1) * height + h) * width + w];
        if (bounding_box_regression_) {
          if (bounding_box_exp_) {
            bb_data[((n * 5 + 0) * height + h) * width + w] += bbr_data[((n * 4 + 0) * height + h) * width + w] * receptive_field_;
            bb_data[((n * 5 + 1) * height + h) * width + w] += bbr_data[((n * 4 + 1) * height + h) * width + w] * receptive_field_;
            bb_data[((n * 5 + 2) * height + h) * width + w] *= exp(bbr_data[((n * 4 + 2) * height + h) * width + w]);
            bb_data[((n * 5 + 3) * height + h) * width + w] *= exp(bbr_data[((n * 4 + 3) * height + h) * width + w]);
          }
          else {
            bb_data[((n * 5 + 0) * height + h) * width + w] += bbr_data[((n * 4 + 1) * height + h) * width + w] * receptive_field_;
            bb_data[((n * 5 + 1) * height + h) * width + w] += bbr_data[((n * 4 + 0) * height + h) * width + w] * receptive_field_;
            bb_data[((n * 5 + 2) * height + h) * width + w] +=
              (bbr_data[((n * 4 + 3) * height + h) * width + w] - bbr_data[((n * 4 + 1) * height + h) * width + w]) * receptive_field_;
            bb_data[((n * 5 + 3) * height + h) * width + w] +=
              (bbr_data[((n * 4 + 2) * height + h) * width + w] - bbr_data[((n * 4 + 0) * height + h) * width + w]) * receptive_field_;
          }
        }
        counter_[(n * height + h) * width + w] = 1;
      }
      else {
        bb_data[((n * 5 + 0) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 1) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 2) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 3) * height + h) * width + w] = 0;
        bb_data[((n * 5 + 4) * height + h) * width + w] = 0;
        counter_[(n * height + h) * width + w] = 0;
      }
    }
  }

template <typename Dtype>
void PredictBoxLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* score_data = bottom[0]->gpu_data();
  Dtype* bb_data = top[0]->mutable_gpu_data();
  const Dtype* bbr_data = NULL;
  if (bounding_box_regression_) bbr_data = bottom[1]->gpu_data();
  int num = bottom[0]->num();
  int output_height = bottom[0]->height();
  int output_width = bottom[0]->width();
  int spatial_dim = output_height * output_width;
  Dtype count = Dtype(0.0);

  if (nms_) {
    PredictBoxForwardWithNMS<Dtype> << <CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> > (num, spatial_dim, output_height, output_width,
                                   score_data, bb_data, positive_thresh_,
                                   stride_, receptive_field_, counter_.mutable_gpu_data(), bottom[2]->gpu_data(),
                                   bounding_box_regression_, bbr_data, bounding_box_exp_);
  }
  else {
    PredictBoxForward<Dtype> << <CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> > (num, spatial_dim, output_height, output_width,
                                   score_data, bb_data, positive_thresh_,
                                   stride_, receptive_field_, counter_.mutable_gpu_data(),
                                   bounding_box_regression_, bbr_data, bounding_box_exp_,
                                   use_stitch_, use_stitch_ ? bottom[2]->gpu_data() : NULL);
  }

  if (output_vector_) {
    caffe_gpu_asum(num*spatial_dim, counter_.gpu_data(), &count);
    const Dtype* score_data_cpu = bottom[0]->cpu_data();
    const Dtype* bb_data_cpu = top[0]->cpu_data();
    if (num == 1 && count > 0) {
      top[1]->Reshape({ bottom[0]->num(), (int)count, 5 });
      int i = 0;
      for (int x = 0; x < output_width; x++) {
        for (int y = 0; y < output_height; y++) {
          if (bb_data_cpu[(4 * output_height + y) * output_width + x] > positive_thresh_) {
            top[1]->mutable_cpu_data()[i * 5 + 0] = bb_data_cpu[(0 * output_height + y) * output_width + x];
            top[1]->mutable_cpu_data()[i * 5 + 1] = bb_data_cpu[(1 * output_height + y) * output_width + x];
            top[1]->mutable_cpu_data()[i * 5 + 2] = bb_data_cpu[(2 * output_height + y) * output_width + x];
            top[1]->mutable_cpu_data()[i * 5 + 3] = bb_data_cpu[(3 * output_height + y) * output_width + x];
            top[1]->mutable_cpu_data()[i * 5 + 4] = bb_data_cpu[(4 * output_height + y) * output_width + x];
            i++;
          }
        }
      }
    }
    else {
      top[1]->Reshape({ bottom[0]->num(), 1, 5 });
      caffe_gpu_set<Dtype>(top[1]->count(), 0, top[1]->mutable_gpu_data());
    }
  }
}

template <typename Dtype>
void PredictBoxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(PredictBoxLayer);

}  // namespace caffe
