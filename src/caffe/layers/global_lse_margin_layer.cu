#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/global_lse_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void GlobalLSEMarginForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                         Dtype* top_data, Dtype margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      if (margin > 0) {
        top_data[index * dim + gt] = bottom_data[index * dim + gt] - margin;
      }
    }
  }

  template <typename Dtype>
  __global__ void select_target_logits(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                       Dtype* selected_value) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      selected_value[index] = bottom_data[index * dim + gt];
    }
  }

  template <typename Dtype>
  __global__ void max_negative_logit(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                     Dtype* max_negative) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      max_negative[index] = Dtype(-1.0);
      for (int i = 0; i < dim; i++) {
        if (i != gt && bottom_data[index * dim + i] > max_negative[index]) {
          max_negative[index] = bottom_data[index * dim + i];
        }
      }
    }
  }

  template <typename Dtype>
  __global__ void kernel_channel_remove_target(const int n, const int dim, const double* bottom_data, const Dtype* label,
                                               double* LSE) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      LSE[index] -= bottom_data[index * dim + gt];
    }
  }

  template <typename Dtype>
  __global__ void kernel_exp(const int num, const int dim, const Dtype* data, const Dtype* label, Dtype* out, const Dtype* max_negative, Dtype scale) {
    CUDA_KERNEL_LOOP(index, num * dim) {
      int n = index / dim;
      int d = index % dim;
      if (d == static_cast<int>(label[n])) {
        out[index] = Dtype(0);
      }
      else {
        out[index] = exp(scale * (data[index] - max_negative[n]));
      }
    }
  }

  template <typename Dtype>
  __global__ void kernel_exp_norm(const int num, const int dim, const Dtype* data, const Dtype* label, Dtype* out, const Dtype* max_negative, const Dtype* scale) {
    CUDA_KERNEL_LOOP(index, num * dim) {
      int n = index / dim;
      int d = index % dim;
      if (d == static_cast<int>(label[n])) {
        out[index] = Dtype(0);
      }
      else {
        out[index] = exp(scale[n] * (data[index] - max_negative[n]));
      }
    }
  }

  template <typename Dtype>
  __global__ void kernel_log(const int count, Dtype* in_out_data, const Dtype* max_negative, Dtype scale) {
    CUDA_KERNEL_LOOP(index, count) {
      in_out_data[index] = log(in_out_data[index]) / scale + max_negative[index];
    }
  }

  template <typename Dtype>
  __global__ void kernel_log_norm(const int count, Dtype* in_out_data, const Dtype* max_negative, const Dtype* scale) {
    CUDA_KERNEL_LOOP(index, count) {
      in_out_data[index] = log(in_out_data[index]) / scale[index] + max_negative[index];
    }
  }

  template <typename Dtype>
  __global__ void kernel_num_scale(const int num, const int dim,
                                   const Dtype* data, const Dtype* norm_data,
                                   Dtype* output_data) {
    CUDA_KERNEL_LOOP(index, num * dim) {
      int n = index / dim;
      //int d = index % dim;
      output_data[index] = data[index] * norm_data[n];
    }
  }

  template <typename Dtype>
  __global__ void BackwardNorm(const int num, const int dim,
                               const Dtype* top_diff, const Dtype* top_data, const Dtype* norm_data, Dtype* norm_diff) {
    CUDA_KERNEL_LOOP(index, num) {
      Dtype sum = 0;
      for (int c = 0; c < dim; ++c) {
        sum += top_diff[index * dim + c] * top_data[index * dim + c] / norm_data[index];
      }
      norm_diff[index] = sum;
    }
  }

  template <typename Dtype>
  Dtype GlobalLSEMarginLayer<Dtype>::MeanShift(Blob<Dtype>* margins, Dtype mu, int loop) {
    int num = margins->count();
    const Dtype* margin_data = margins->cpu_data();
    Dtype last_mu = mu;
    while (loop--) {
      int non_zero_count = 0;
      Dtype sum_margin = Dtype(0);
      for (int i = 0; i < num; i++) {
        if (abs(margin_data[i] - mu) < 0.1) {
          sum_margin += margin_data[i];
          non_zero_count++;
        }
      }
      if (non_zero_count == 0) { // re-initialize
        sum_margin = Dtype(0);
        for (int i = 0; i < num; i++) {
          sum_margin += margin_data[i];
          non_zero_count++;
        }
        mu = sum_margin / non_zero_count;
        last_mu = mu;
        loop++;
        continue;
      }
      mu = sum_margin / non_zero_count;
      if (abs(last_mu - mu) < 0.001) break;
      last_mu = mu;
    }
    return mu;
  }

  template double GlobalLSEMarginLayer<double>::MeanShift(Blob<double>* margins, double mu, int loop);
  template float GlobalLSEMarginLayer<float>::MeanShift(Blob<float>* margins, float mu, int loop);

  template <typename Dtype>
  Dtype GlobalLSEMarginLayer<Dtype>::MeanMaxNegativeLogit(const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    // NOLINT_NEXT_LINE(whitespace/operators)
    max_negative_logit<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, max_negative_logits_.mutable_gpu_data());

    Dtype mean_target;
    caffe_gpu_dot(num, max_negative_logits_.gpu_data(), sum_multiplier_.gpu_data(), &mean_target);
    mean_target /= num;

    return mean_target;
  }

  template double GlobalLSEMarginLayer<double>::MeanMaxNegativeLogit(const vector<Blob<double>*>& bottom);
  template float GlobalLSEMarginLayer<float>::MeanMaxNegativeLogit(const vector<Blob<float>*>& bottom);

  template <typename Dtype>
  Dtype GlobalLSEMarginLayer<Dtype>::CalcLSE(const vector<Blob<Dtype>*>& bottom, Blob<Dtype>* LSE) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* scale = this->blobs_[0]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Blob<Dtype> exp_bottom_data;
    exp_bottom_data.Reshape({ num, dim });
    Blob<Dtype> sum_exp_bottom_data;
    sum_exp_bottom_data.Reshape({ num });

    if (original_norm_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_exp_norm<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, bottom_data, label_data, exp_bottom_data.mutable_gpu_data(), max_negative_logits_.gpu_data(), bottom[2]->gpu_data());
    }
    else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_exp<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, bottom_data, label_data, exp_bottom_data.mutable_gpu_data(), max_negative_logits_.gpu_data(), scale[0]);
    }

    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1.0,
                          exp_bottom_data.gpu_data(), sum_multiplier_channel_.gpu_data(), 0.0, LSE->mutable_gpu_data());

    if (original_norm_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_log_norm<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, LSE->mutable_gpu_data(), max_negative_logits_.gpu_data(), bottom[2]->gpu_data());
    }
    else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_log<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, LSE->mutable_gpu_data(), max_negative_logits_.gpu_data(), scale[0]);
    }

    Dtype average_LSE;
    caffe_gpu_dot(num, LSE->gpu_data(), sum_multiplier_.gpu_data(), &average_LSE);
    average_LSE /= num;
    return average_LSE;
  }

  template double GlobalLSEMarginLayer<double>::CalcLSE(const vector<Blob<double>*>& bottom, Blob<double>* LSE);
  template float GlobalLSEMarginLayer<float>::CalcLSE(const vector<Blob<float>*>& bottom, Blob<float>* LSE);

  template <typename Dtype>
  void GlobalLSEMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    const Dtype* cos_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* scale = this->blobs_[0]->mutable_cpu_data();
    Dtype* margin = this->blobs_[1]->mutable_cpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (scale[0] < min_scale_) scale[0] = min_scale_;
    if (scale[0] > max_scale_) scale[0] = max_scale_;

    // NOLINT_NEXT_LINE(whitespace/operators)
    select_target_logits<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, cos_data, label_data, target_logits_.mutable_gpu_data());

    Dtype mean_max_negative = MeanMaxNegativeLogit(bottom);
    Dtype mean_lse = CalcLSE(bottom, &lse_);
    caffe_gpu_sub(num, target_logits_.gpu_data(), lse_.gpu_data(), margins_.mutable_gpu_data());

    if (margin[0] == Dtype(0)) {//initialize
      Dtype mean_margin;
      caffe_gpu_dot(num, margins_.gpu_data(), sum_multiplier_.gpu_data(), &mean_margin);
      mean_margin /= num;
      Dtype batch_margin = MeanShift(&margins_, mean_margin, 20);
      if (!isnan(batch_margin)) margin[0] = batch_margin;
    }
    else {
      Dtype batch_margin = MeanShift(&margins_, margin[0]);
      if (!isnan(batch_margin)) margin[0] = 0.9 * margin[0] + 0.1 * batch_margin;
    }

    caffe_copy(count, cos_data, top_data);

    // NOLINT_NEXT_LINE(whitespace/operators)
    GlobalLSEMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, cos_data, label_data, top_data, margin[0]);
    CUDA_POST_KERNEL_CHECK;

    if (original_norm_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_num_scale<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_data, bottom[2]->gpu_data(), top_data);
    }
    else {
      caffe_gpu_scal(count, scale[0], top_data);
    }

    if (top.size() >= 2) {
      Dtype* statistics = top[1]->mutable_cpu_data();
      if (original_norm_) {
        caffe_gpu_dot(num, bottom[2]->gpu_data(), sum_multiplier_.gpu_data(), statistics);
        statistics[0] /= num;
      }
      else {
        statistics[0] = scale[0];
      }
      Dtype mean_target;
      caffe_gpu_dot(num, target_logits_.gpu_data(), sum_multiplier_.gpu_data(), &mean_target);
      mean_target /= num;
      statistics[1] = mean_target;
      statistics[2] = mean_lse;
      statistics[3] = mean_max_negative;
      statistics[4] = margin[0];
    }
  }

  template <typename Dtype>
  void GlobalLSEMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down,
                                                 const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    Dtype scale = this->blobs_[0]->cpu_data()[0];

    if (original_norm_) {
      if (propagate_down[2]) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        BackwardNorm<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
          num, dim, top_diff, top_data, bottom[2]->gpu_data(), bottom[2]->mutable_gpu_diff());
      }
    }
    else {
      caffe_gpu_dot(count, top_diff, top_data, this->blobs_[0]->mutable_cpu_diff());
      this->blobs_[0]->mutable_cpu_diff()[0] /= scale;
    }


    //caffe_set(bottom[2]->count(), Dtype(0), bottom[2]->mutable_gpu_diff());
    if (propagate_down[0]) {
      if (original_norm_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        kernel_num_scale<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
          num, dim, top_diff, bottom[2]->gpu_data(), bottom_diff);
      }
      else {
        caffe_gpu_scale(count, scale, top_diff, bottom_diff);
      }
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(GlobalLSEMarginLayer);


}  // namespace caffe
