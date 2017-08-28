/*
Based on https://github.com/zimenglan-sysu-512/Focal-Loss.
Paper https://arxiv.org/abs/1708.02002.
NOTICE: This layer is NOT the original focal loss layer.
I changed 1-p to 1+p for face verification. If you want the original version,
replace the cpp and cu files from https://github.com/zimenglan-sysu-512/Focal-Loss.
*/
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


  template <typename Dtype>
  __global__ void LogOpGPU(const int nthreads,
                           const Dtype* in, Dtype* out, const Dtype eps) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      out[index] = log(max(in[index], eps));
    }
  }

  template <typename Dtype>
  void FocalLossLayer<Dtype>::compute_intermediate_values_of_gpu() {
    // compute the corresponding variables
    const int count = prob_.count();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* ones_data = ones_.gpu_data();
    Dtype* log_prob_data = log_prob_.mutable_gpu_data();
    Dtype* power_prob_data = power_prob_.mutable_gpu_data();

    /// log(p_t)
    const int nthreads = prob_.count();
    const Dtype eps = Dtype(FLT_MIN); // where FLT_MIN = 1.17549e-38, here u can change it
                                      // more stable
                                      // NOLINT_NEXT_LINE(whitespace/operators)
    LogOpGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, log_prob_data, eps);
    /// caffe_gpu_log(count,  prob_data, log_prob_data);

    if (type_ == FocalLossParameter::ONEADDP) {
      /// (1 + p_t) ^ gamma
      caffe_gpu_add(count, ones_data, prob_data, power_prob_data);
    }
    else {
      /// (1 - p_t) ^ gamma
      caffe_gpu_sub(count, ones_data, prob_data, power_prob_data);
    }
    caffe_gpu_powx(count, power_prob_.gpu_data(), gamma_, power_prob_data);
    //caffe_gpu_scal(count, alpha_, power_prob_data);
  }

  template <typename Dtype>
  __global__ void FocalLossForwardGPU(const int nthreads,
                                      const Dtype* log_prob_data,
                                      const Dtype* power_prob_data,
                                      const Dtype* label,
                                      Dtype* loss,
                                      const int num,
                                      const int dim,
                                      const int spatial_dim,
                                      const bool has_ignore_label_,
                                      const int ignore_label_,
                                      Dtype* counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        loss[index] = 0;
        counts[index] = 0;
      }
      else {
        int ind = n * dim + label_value * spatial_dim + s;
        // loss[index]   = -max(power_prob_data[ind] * log_prob_data[ind], Dtype(log(Dtype(FLT_MIN))));
        loss[index] = -power_prob_data[ind] * log_prob_data[ind];
        counts[index] = 1;
      }
    }
  }

  template <typename Dtype>
  void FocalLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the softmax prob values.
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

    // compute all needed values
    compute_intermediate_values_of_gpu();

    // const Dtype* prob_data       = prob_.gpu_data();
    const Dtype* log_prob_data = log_prob_.gpu_data();
    const Dtype* power_prob_data = power_prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;

    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();

    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();

    // NOLINT_NEXT_LINE(whitespace/operators)
    FocalLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS >> >(nthreads, log_prob_data, power_prob_data,
                                  label, loss_data, outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);

    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;

    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                          valid_count);
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    }
  }

  template <typename Dtype>
  __global__ void FocalLossBackwardGPU(const int nthreads,
                                       const Dtype* top,
                                       const Dtype* label,
                                       const Dtype* prob_data,
                                       const Dtype* log_prob_data,
                                       const Dtype* power_prob_data,
                                       Dtype* bottom_diff,
                                       const int num,
                                       const int dim,
                                       const int spatial_dim,
                                       const Dtype gamma,
                                       const bool has_ignore_label_,
                                       const int ignore_label_,
                                       const Dtype eps,
                                       Dtype* counts,
                                       bool one_add_p) {
    const int channels = dim / spatial_dim;

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);

      if (has_ignore_label_ && label_value == ignore_label_) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[n * dim + c * spatial_dim + s] = 0;
        }
        counts[index] = 0;
      }
      else {
        // the gradient from FL w.r.t p_t, here ignore the `sign`
        int ind_i = n * dim + label_value * spatial_dim + s; // index of ground-truth label
        Dtype grad;
        if (one_add_p) {
          grad = gamma * (power_prob_data[ind_i] / max(1 + prob_data[ind_i], eps))
            * log_prob_data[ind_i] * prob_data[ind_i]
            + power_prob_data[ind_i];
        }
        else {
          grad = -gamma * (power_prob_data[ind_i] / max(1 - prob_data[ind_i], eps))
            * log_prob_data[ind_i] * prob_data[ind_i]
            + power_prob_data[ind_i];
        }
        // the gradient w.r.t input data x
        for (int c = 0; c < channels; ++c) {
          int ind_j = n * dim + c * spatial_dim + s;
          if (c == label_value) {
            // if i == j, (here i,j are refered for derivative of softmax)
            bottom_diff[ind_j] = grad * (prob_data[ind_i] - 1);
          }
          else {
            // if i != j, (here i,j are refered for derivative of softmax)
            bottom_diff[ind_j] = grad * prob_data[ind_j];
          }
        }
        // count
        counts[index] = 1;
      }
    }
  }

  template <typename Dtype>
  void FocalLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.gpu_data();
      const Dtype* top_data = top[0]->gpu_data();
      const Dtype* label = bottom[1]->gpu_data();
      const int dim = prob_.count() / outer_num_;
      const int nthreads = outer_num_ * inner_num_;
      const Dtype eps = 1e-10;

      // intermidiate  
      const Dtype* log_prob_data = log_prob_.gpu_data();
      const Dtype* power_prob_data = power_prob_.gpu_data();

      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      Dtype* counts = prob_.mutable_gpu_diff();

      // NOLINT_NEXT_LINE(whitespace/operators)
      FocalLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS >> >(nthreads, top_data, label, prob_data, log_prob_data, power_prob_data,
                                    bottom_diff, outer_num_, dim, inner_num_, gamma_, has_ignore_label_, ignore_label_, eps, counts,
                                    type_ == FocalLossParameter::ONEADDP);

      // Only launch another CUDA kernel if we actually need the count of valid outputs.
      Dtype valid_count = -1;
      if (normalization_ == LossParameter_NormalizationMode_VALID &&
          has_ignore_label_) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
      }
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
      caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(FocalLossLayer);

}  // namespace caffe
