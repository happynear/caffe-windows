#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_affine_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificAffineForwardWithBound(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                             const Dtype* lower_bound, Dtype* top_data, Dtype scale, Dtype bias, Dtype power) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      if (bottom_data[index * dim + gt] > 0) {
        top_data[index * dim + gt] = pow(bottom_data[index * dim + gt], power) * scale;
        if (top_data[index * dim + gt] > lower_bound[index]) top_data[index * dim + gt] += bias;
      }
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificAffineForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                                 Dtype* top_data, Dtype scale, Dtype bias, Dtype power) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      if (bottom_data[index * dim + gt] > 0) {
        top_data[index * dim + gt] = pow(bottom_data[index * dim + gt], power) * scale;
        /*if (top_data[index * dim + gt] > -bias)*/ top_data[index * dim + gt] += bias;
      }
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificAffineBackward(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                              const Dtype* bottom_data, Dtype* bottom_diff, Dtype scale, Dtype power) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      if (bottom_data[index * dim + gt] > 0) {
        if (power == Dtype(1)) {
          bottom_diff[index * dim + gt] = top_diff[index * dim + gt] * scale;
        }
        else {
          bottom_diff[index * dim + gt] = top_diff[index * dim + gt] * scale * power * pow(bottom_data[index * dim + gt], power - 1);
        }
      }
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificAffineBackwardScale(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                   const Dtype* bottom_data, Dtype* selected_value) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      selected_value[index] = top_diff[index * dim + gt] * bottom_data[index * dim + gt];
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificAffineBackwardBias(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                  const Dtype* bottom_data, Dtype* selected_value) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      selected_value[index] = top_diff[index * dim + gt];
    }
  }

  template <typename Dtype>
  __global__ void kernel_exp(const int count, const Dtype* data, double* out, Dtype scale) {
    CUDA_KERNEL_LOOP(index, count) {
      out[index] = exp((double)scale * (double)data[index]);
    }
  }

  template <typename Dtype>
  __global__ void kernel_log(const int count, const double* data, Dtype* out, Dtype scale) {
    CUDA_KERNEL_LOOP(index, count) {
      out[index] = (Dtype)log(data[index]) / scale;
    }
  }

  template <typename Dtype>
  __global__ void double2float(const int count, const double* data, Dtype* out) {
    CUDA_KERNEL_LOOP(index, count) {
      out[index] = (Dtype)data[index];
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
  Dtype LabelSpecificAffineLayer<Dtype>::CalcLSE(const vector<Blob<Dtype>*>& bottom, Blob<Dtype>* LSE) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Blob<double> exp_bottom_data;
    exp_bottom_data.Reshape({ num, dim });
    Blob<double> sum_exp_bottom_data;
    sum_exp_bottom_data.Reshape({ num });
    Blob<double> LSE_double;
    LSE_double.Reshape({ num });

    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_exp<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
      count, bottom_data, exp_bottom_data.mutable_gpu_data(), scale_factor_);

    caffe_gpu_gemv<double>(CblasNoTrans, num, dim, 1.0,
                   exp_bottom_data.gpu_data(), sum_multiplier_channel_.gpu_data(), 0.0, LSE_double.mutable_gpu_data());

    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_remove_target<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, exp_bottom_data.gpu_data(), label_data, LSE_double.mutable_gpu_data());

    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_log<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, LSE_double.gpu_data(), LSE->mutable_gpu_data(), scale_factor_);

    Dtype average_LSE;
    caffe_gpu_dot(num, LSE->gpu_data(), sum_multiplier_.gpu_data(), &average_LSE);
    average_LSE /= num;
    return average_LSE;
  }

  template double LabelSpecificAffineLayer<double>::CalcLSE(const vector<Blob<double>*>& bottom, Blob<double>* LSE);
  template float LabelSpecificAffineLayer<float>::CalcLSE(const vector<Blob<float>*>& bottom, Blob<float>* LSE);

  template <typename Dtype>
  Dtype LabelSpecificAffineLayer<Dtype>::MeanTargetLogit(const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    // NOLINT_NEXT_LINE(whitespace/operators)
    select_target_logits<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, selected_value_.mutable_gpu_data());

    Dtype mean_target;
    caffe_gpu_dot(num, selected_value_.gpu_data(), sum_multiplier_.gpu_data(), &mean_target);
    mean_target /= num;
    
    return mean_target;
  }

  template double LabelSpecificAffineLayer<double>::MeanTargetLogit(const vector<Blob<double>*>& bottom);
  template float LabelSpecificAffineLayer<float>::MeanTargetLogit(const vector<Blob<float>*>& bottom);

  template <typename Dtype>
  Dtype LabelSpecificAffineLayer<Dtype>::MeanMaxNegativeLogit(const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    // NOLINT_NEXT_LINE(whitespace/operators)
    max_negative_logit<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, selected_value_.mutable_gpu_data());

    Dtype mean_target;
    caffe_gpu_dot(num, selected_value_.gpu_data(), sum_multiplier_.gpu_data(), &mean_target);
    mean_target /= num;

    return mean_target;
  }

  template double LabelSpecificAffineLayer<double>::MeanMaxNegativeLogit(const vector<Blob<double>*>& bottom);
  template float LabelSpecificAffineLayer<float>::MeanMaxNegativeLogit(const vector<Blob<float>*>& bottom);

  template <typename Dtype>
  void LabelSpecificAffineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* scale_bias = this->blobs_[0]->mutable_cpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (this->phase_ == TEST) {
      scale = Dtype(1);
      bias = Dtype(0);
      power = Dtype(1);
    }
    else {
      if (auto_tune_) {//Now, only support auto-tuning the bias.
        scale = Dtype(1.0);
        scale_bias[0] = scale;
        power = Dtype(1.0);
        scale_bias[2] = power;

        Dtype mean_target = MeanTargetLogit(bottom);
        Dtype mean_max_negative = CalcLSE(bottom, &lower_bound_);
        scale_bias[1] = 0.9 * scale_bias[1] + 0.1 * (mean_max_negative - mean_target);
        bias = scale_bias[1] < Dtype(0) ? scale_bias[1] : Dtype(0);
        if (top.size() >= 2) {
          top[1]->mutable_cpu_data()[3] = mean_target;
          top[1]->mutable_cpu_data()[4] = mean_max_negative;
        }
      }
      else {
        scale = scale_base_ * pow(((Dtype)1. + scale_gamma_ * iteration_), scale_power_);
        bias = bias_base_ + pow(((Dtype)1. + bias_gamma_ * iteration_), bias_power_) - (Dtype)1.;
        power = power_base_ * pow(((Dtype)1. + power_gamma_ * iteration_), power_power_);
        scale = std::max(scale, scale_min_);
        scale = std::min(scale, scale_max_);
        bias = std::max(bias, bias_min_);
        bias = std::min(bias, bias_max_);
        power = std::max(power, power_min_);
        power = std::min(power, power_max_);
        iteration_++;
      }
    }

    caffe_copy(count, bottom_data, top_data);
    if (top.size() >= 2) {
      top[1]->mutable_cpu_data()[0] = scale;
      top[1]->mutable_cpu_data()[1] = bias;
      top[1]->mutable_cpu_data()[2] = power;
    }
    if (!transform_test_ && this->phase_ == TEST) return;

    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificAffineForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, top_data, scale, bias, power);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void LabelSpecificAffineLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* scale_bias_diff = bottom.size() == 3 ? bottom[2]->mutable_cpu_diff() : this->blobs_[0]->mutable_cpu_diff();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    if (propagate_down[0]) {
      caffe_copy(count, top_diff, bottom_diff);
      if (!transform_test_ && this->phase_ == TEST) return;
      if (original_bp_) return;
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificAffineBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_data, bottom_diff, scale, power);
      CUDA_POST_KERNEL_CHECK;
    }

    //if (auto_tune_ || (bottom.size() == 3 && propagate_down[2])) {
    //  // NOLINT_NEXT_LINE(whitespace/operators)
    //  LabelSpecificAffineBackwardScale<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
    //    num, dim, top_diff, label_data, bottom_data, selected_value_.mutable_gpu_data());
    //  caffe_gpu_dot(num, selected_value_.gpu_data(), sum_multiplier_.gpu_data(), scale_bias_diff);

    //  // NOLINT_NEXT_LINE(whitespace/operators)
    //  LabelSpecificAffineBackwardBias<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
    //    num, dim, top_diff, label_data, bottom_data, selected_value_.mutable_gpu_data());
    //  caffe_gpu_dot(num, selected_value_.gpu_data(), sum_multiplier_.gpu_data(), scale_bias_diff+1);
    //}
  }


  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificAffineLayer);


}  // namespace caffe
