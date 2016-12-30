#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void Compute_distance_data_gpu(int nthreads, const int K,
                                            const Dtype* bottom,
                                            const Dtype* label,
                                            const Dtype* center,
                                            Dtype* distance) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int m = index / K;
      int k = index % K;
      const int label_value = static_cast<int>(label[m]);
      // distance(i) = x(i) - c_{y(i)}
      distance[index] = bottom[index] - center[label_value * K + k];
    }
  }
  template <typename Dtype>
  __global__ void Compute_variation_sum_gpu(int nthreads, const int K,
                                            const Dtype* label,
                                            const Dtype* distance,
                                            Dtype* variation_sum, int * count) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int m = index / K;
      int k = index % K;
      const int label_value = static_cast<int>(label[m]);
      variation_sum[label_value * K + k] -= distance[m * K + k];
      count[label_value] += ((k == 0) ? 1 : 0);
    }
  }
  template <typename Dtype>
  __global__ void Compute_center_diff_gpu(int nthreads, const int K,
                                          const Dtype* label,
                                          Dtype* variation_sum,
                                          int * count, Dtype* center_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int m = index / K;
      int k = index % K;
      const int n = static_cast<int>(label[m]);
      center_diff[n * K + k] += variation_sum[n * K + k]
        / (count[n] + (Dtype)1.);
    }
  }
  template <typename Dtype>
  void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
    int nthreads = M_ * K_;
    Compute_distance_data_gpu<Dtype> << < CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS >> >(nthreads, K_, bottom[0]->gpu_data(),
                                  bottom[1]->gpu_data(),
                                  this->blobs_[0]->gpu_data(),
                                  distance_.mutable_gpu_data());
    Dtype dot;
    Dtype loss;
    if (distance_type_ == "L1") {
      caffe_gpu_asum(M_ * K_, distance_.gpu_data(), &dot);
      loss = dot / M_;
    }
    else if(distance_type_ == "L2"){
      caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);
      loss = dot / M_ / Dtype(2);
    }
    else {
      LOG(FATAL) << "distance_type must be L1 or L2!";
    }
    top[0]->mutable_cpu_data()[0] = loss;
  }
  template <typename Dtype>
  void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom) {
    caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_gpu_data());
    caffe_gpu_set(N_, 0, count_.mutable_gpu_data());
    int nthreads = M_ * K_;
    if (distance_type_ == "L1") {
      caffe_gpu_sign(M_ * K_, distance_.gpu_data(), distance_.mutable_gpu_data());
    }
    Compute_variation_sum_gpu<Dtype> << < CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS >> >(nthreads, K_, bottom[1]->gpu_data(),
                                  distance_.gpu_data(),
                                  variation_sum_.mutable_gpu_data(),
                                  count_.mutable_gpu_data());
    Compute_center_diff_gpu<Dtype> << < CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS >> >(nthreads, K_, bottom[1]->gpu_data(),
                                  variation_sum_.mutable_gpu_data(),
                                  count_.mutable_gpu_data(),
                                  this->blobs_[0]->mutable_gpu_diff());
    if (propagate_down[0]) {
      caffe_gpu_scale(M_ * K_,
                      top[0]->cpu_diff()[0] / M_,
                      distance_.gpu_data(),
                      bottom[0]->mutable_gpu_diff());
    }
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe