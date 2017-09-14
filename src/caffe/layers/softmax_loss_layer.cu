#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "thrust/functional.h"
#include "thrust/sort.h"

namespace caffe {

  template <typename Dtype>
  __global__ void SoftmaxLossForwardGPU(const int nthreads,
                                        Dtype* prob_data, const Dtype* label, Dtype* loss,
                                        const int num, const int dim, const int spatial_dim,
                                        const bool has_ignore_label_, const int ignore_label_,
                                        const bool has_hard_mining_label_, const int hard_mining_label_,
                                        const bool has_cutting_point_, Dtype cutting_point_,
                                        Dtype label_smooth_factor, Dtype* counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);
      const int channels = dim / spatial_dim;
      if (has_cutting_point_ && prob_data[n * dim + label_value * spatial_dim + s] > cutting_point_
          && (!has_hard_mining_label_ || hard_mining_label_ == label_value)) {
        for (int c = 0; c < channels; ++c) {
          prob_data[n * dim + c * spatial_dim + s] = 0;
        }
        prob_data[n * dim + label_value * spatial_dim + s] = 1;
      }
      if ((has_ignore_label_ && label_value == ignore_label_)) {
        loss[index] = 0;
        counts[index] = 0;
      }
      else {
        loss[index] = Dtype(0);
        for (int c = 0; c < channels; ++c) {
          if (c == label_value) {
            loss[index] -= (1 - label_smooth_factor) * log(max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN)));
          }
          else {
            if (label_smooth_factor > Dtype(1e-6)) {
              loss[index] -= label_smooth_factor / (channels - 1) * log(max(prob_data[n * dim + c * spatial_dim + s], Dtype(FLT_MIN)));
            }
          }
        }
        counts[index] = 1;
      }
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxLossForwardWithWeightsGPU(const int nthreads,
                                                   Dtype* prob_data, const Dtype* label, Dtype* loss,
                                                   const Dtype* weights, const Dtype* class_weights,
                                                   const int num, const int dim, const int spatial_dim,
                                                   const bool has_ignore_label_, const int ignore_label_,
                                                   const bool has_hard_mining_label_, const int hard_mining_label_,
                                                   const bool has_cutting_point_, Dtype cutting_point_, 
                                                   Dtype label_smooth_factor, Dtype* counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);
      const Dtype weight_value = weights[n * spatial_dim + s] * class_weights[label_value];
      const int channels = dim / spatial_dim;
      if (has_cutting_point_ && prob_data[n * dim + label_value * spatial_dim + s] > cutting_point_
          && (!has_hard_mining_label_ || hard_mining_label_ == label_value)) {
        for (int c = 0; c < channels; ++c) {
          prob_data[n * dim + c * spatial_dim + s] = 0;
        }
        prob_data[n * dim + label_value * spatial_dim + s] = 1;
      }
      if ((weight_value == 0) || (has_ignore_label_ && label_value == ignore_label_)) {
        loss[index] = 0;
        counts[index] = 0;
      }
      else {
        loss[index] = Dtype(0);
        for (int c = 0; c < channels; ++c) {
          if (c == label_value) {
            loss[index] -= weight_value * (1 - label_smooth_factor) * log(max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN)));
          }
          else {
            if (label_smooth_factor > Dtype(1e-6)) {
              loss[index] -= weight_value * label_smooth_factor / (channels - 1) * log(max(prob_data[n * dim + c * spatial_dim + s], Dtype(FLT_MIN)));
            }
          }
        }
        counts[index] = weight_value;
      }
    }
  }

  template <typename Dtype>
  void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    Dtype* prob_data = prob_.mutable_gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data = loss_.mutable_gpu_data();
    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = counts_.mutable_gpu_data();
    if (bottom.size() == 2) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, label, loss_data,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
          has_hard_mining_label_, hard_mining_label_,
          has_cutting_point_, cutting_point_, label_smooth_factor_, counts);
    }
    else if (bottom.size() == 3) {
      const Dtype* weights = bottom[2]->gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossForwardWithWeightsGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, label, loss_data,
          weights, class_weight_.gpu_data(),
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
          has_hard_mining_label_, hard_mining_label_,
          has_cutting_point_, cutting_point_, label_smooth_factor_, counts);
    }
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if ((normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_)
        || (bottom.size() == 3)) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                          valid_count);
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
                                         const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
                                         const int spatial_dim, const bool has_ignore_label_,
                                         const int ignore_label_, Dtype label_smooth_factor, Dtype* counts) {
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
        for (int c = 0; c < channels; ++c) {
          if (c == label_value) {
            bottom_diff[n * dim + label_value * spatial_dim + s] -= 1 - label_smooth_factor;
          }
          else {
            if (label_smooth_factor > Dtype(1e-6)) {
              bottom_diff[n * dim + c * spatial_dim + s] -= label_smooth_factor / (channels - 1);
            }
          }
        }
        counts[index] = 1;
      }
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxLossBackwardWithWeightsGPU(const int nthreads, const Dtype* top,
                                                    const Dtype* weights, const Dtype* class_weight,
                                                    const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
                                                    const int spatial_dim, const bool has_ignore_label_,
                                                    const int ignore_label_, Dtype label_smooth_factor, Dtype* counts) {
    const int channels = dim / spatial_dim;

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);
      const Dtype weight_value = weights[n * spatial_dim + s];
      if ((has_ignore_label_ && label_value == ignore_label_) || (weight_value == 0)) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[n * dim + c * spatial_dim + s] = 0;
        }
        counts[index] = 0;
      }
      else {
        for (int c = 0; c < channels; ++c) {
          if (c == label_value) {
            bottom_diff[n * dim + label_value * spatial_dim + s] -= 1 - label_smooth_factor;
          }
          else {
            if (label_smooth_factor > Dtype(1e-6)) {
              bottom_diff[n * dim + c * spatial_dim + s] -= label_smooth_factor / (channels - 1);
            }
          }
          bottom_diff[n * dim + c * spatial_dim + s] *= weight_value * class_weight[c];
        }
        counts[index] = weight_value;
      }
    }
  }

  template <typename Dtype>
  __global__ void Threshold(const int n, const Dtype* loss, Dtype threshold, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = loss[index] < threshold ? 0 : out[index];
    }
  }

  template <typename Dtype>
  __global__ void ThresholdWithLabel(const int n, const Dtype* loss, Dtype threshold, 
    const Dtype* label, Dtype hard_mining_label, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = (label[index] == hard_mining_label &&loss[index] < threshold) ? 0 : out[index];
    }
  }

  template <typename Dtype>
  void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      if (has_hard_ratio_ && bottom.size() == 3) {
        caffe_copy(outer_num_ * inner_num_, loss_.cpu_data(), loss_.mutable_cpu_diff());
        std::sort(loss_.mutable_cpu_diff(), loss_.mutable_cpu_diff() + outer_num_ * inner_num_);//thrust::sort
        Dtype loss_threshold = loss_.cpu_diff()[(int)(outer_num_ * inner_num_ * (1 - hard_ratio_))];
        if (has_hard_mining_label_) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          ThresholdWithLabel<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_), CAFFE_CUDA_NUM_THREADS >> >(
            outer_num_ * inner_num_, loss_.gpu_data(), loss_threshold, 
            bottom[1]->gpu_data(), hard_mining_label_, bottom[2]->mutable_gpu_data());
        }
        else {
          // NOLINT_NEXT_LINE(whitespace/operators)
          Threshold<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_), CAFFE_CUDA_NUM_THREADS >> >(
            outer_num_ * inner_num_, loss_.gpu_data(), loss_threshold, bottom[2]->mutable_gpu_data());
        }
      }

      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.gpu_data();
      const Dtype* top_data = top[0]->gpu_data();
      caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->gpu_data();
      const int dim = prob_.count() / outer_num_;
      const int nthreads = outer_num_ * inner_num_;
      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      Dtype* counts = counts_.mutable_gpu_data();
      if (bottom.size() == 2) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        SoftmaxLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS >> >(nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, label_smooth_factor_, counts);
      }
      else if (bottom.size() == 3) {
        const Dtype* weights = bottom[2]->gpu_data();
        // NOLINT_NEXT_LINE(whitespace/operators)
        SoftmaxLossBackwardWithWeightsGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS >> >(nthreads, top_data, 
          weights, class_weight_.gpu_data(), label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, label_smooth_factor_, counts);
      }

      Dtype valid_count = -1;
      // Only launch another CUDA kernel if we actually need the count of valid
      // outputs.
      if ((normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_)
          || (bottom.size() == 3)) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] /
        get_normalizer(normalization_, valid_count);
      caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
