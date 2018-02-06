#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/radial_removal_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void RadialRemovalMarginForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                                 Dtype* top_data, const Dtype* margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      //if (margin[index] > 0) {
        top_data[index * dim + gt] = bottom_data[index * dim + gt] - margin[index];
      //}
    }
  }

  template <typename Dtype>
  __global__ void kernel_exp(const int num, const int dim, 
                             const Dtype* data, const Dtype* label, const Dtype* norm, const Dtype* max_negative_logits,
                             Dtype* out) {
    CUDA_KERNEL_LOOP(index, num*dim) {
      int n = index / dim;
      int d = index % dim;
      if (d == static_cast<int>(label[n])) {
        out[index] = Dtype(0);
      }
      else {
        out[index] = exp(norm[n] * (data[index] - max_negative_logits[n]));
      }
    }
  }

  template <typename Dtype>
  __global__ void kernel_log(const int count, 
                             const Dtype* data, const Dtype* norm, const Dtype* max_negative_logits,
                             Dtype* out) {
    CUDA_KERNEL_LOOP(index, count) {
      out[index] = log(data[index]) / norm[index] + max_negative_logits[index];
    }
  }

  template <typename Dtype>
  __global__ void kernel_channel_remove_target(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                               Dtype* LSE) {
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
  Dtype RadialRemovalMarginLayer<Dtype>::CalcSoftNegativeMean(const vector<Blob<Dtype>*>& bottom) {
    const Dtype* cos_data = bottom[0]->gpu_data();
    const Dtype* norm_data = original_norm_ ? bottom[2]->gpu_data() : fake_feature_norm_.gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Blob<Dtype> exp_cos_data;
    exp_cos_data.Reshape({ num, dim });
    Blob<Dtype> exp_cos_data_mul_cos;
    exp_cos_data_mul_cos.Reshape({ num, dim });
    Blob<Dtype> sum_exp_cos_data_mul_cos;
    sum_exp_cos_data_mul_cos.Reshape({ num });

    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_exp<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, cos_data, label_data, norm_data, max_negative_logits_.gpu_data(), exp_cos_data.mutable_gpu_data());

    caffe_gpu_mul(count, exp_cos_data.gpu_data(), cos_data, exp_cos_data_mul_cos.mutable_gpu_data());

    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1.0,
                          exp_cos_data.gpu_data(), sum_multiplier_channel_.gpu_data(), 0.0, sum_exp_negative_logits_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1.0,
                          exp_cos_data_mul_cos.gpu_data(), sum_multiplier_channel_.gpu_data(), 0.0, sum_exp_cos_data_mul_cos.mutable_gpu_data());

    caffe_gpu_div<Dtype>(num, sum_exp_cos_data_mul_cos.gpu_data(), sum_exp_negative_logits_.gpu_data(), margins_.mutable_gpu_data());
    caffe_gpu_axpby<Dtype>(num, Dtype(1), target_logits_.gpu_data(), Dtype(-1), margins_.mutable_gpu_data());

    Dtype average_margin;
    caffe_gpu_dot(num, margins_.gpu_data(), sum_multiplier_.gpu_data(), &average_margin);
    average_margin /= num;
    return average_margin;
  }

  template double RadialRemovalMarginLayer<double>::CalcSoftNegativeMean(const vector<Blob<double>*>& bottom);
  template float RadialRemovalMarginLayer<float>::CalcSoftNegativeMean(const vector<Blob<float>*>& bottom);

  template <typename Dtype>
  Dtype RadialRemovalMarginLayer<Dtype>::MeanTargetLogit(const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    // NOLINT_NEXT_LINE(whitespace/operators)
    select_target_logits<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, target_logits_.mutable_gpu_data());

    Dtype mean_target;
    caffe_gpu_dot(num, target_logits_.gpu_data(), sum_multiplier_.gpu_data(), &mean_target);
    mean_target /= num;
    
    return mean_target;
  }

  template double RadialRemovalMarginLayer<double>::MeanTargetLogit(const vector<Blob<double>*>& bottom);
  template float RadialRemovalMarginLayer<float>::MeanTargetLogit(const vector<Blob<float>*>& bottom);

  template <typename Dtype>
  Dtype RadialRemovalMarginLayer<Dtype>::MeanMaxNegativeLogit(const vector<Blob<Dtype>*>& bottom) {
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

  template double RadialRemovalMarginLayer<double>::MeanMaxNegativeLogit(const vector<Blob<double>*>& bottom);
  template float RadialRemovalMarginLayer<float>::MeanMaxNegativeLogit(const vector<Blob<float>*>& bottom);

  template <typename Dtype>
  void RadialRemovalMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* cos_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* norm_data = original_norm_ ? bottom[2]->gpu_data() : fake_feature_norm_.gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Dtype mean_target = MeanTargetLogit(bottom);
    Dtype mean_max_negative = MeanMaxNegativeLogit(bottom);
    Dtype mean_margin = CalcSoftNegativeMean(bottom);
    Dtype mean_LSE;
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_log<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, sum_exp_negative_logits_.gpu_data(), norm_data, max_negative_logits_.gpu_data(), log_sum_exp_negative_logits_.mutable_gpu_data());
    caffe_gpu_dot(num, log_sum_exp_negative_logits_.gpu_data(), sum_multiplier_.gpu_data(), &mean_LSE);
    mean_LSE /= num;
    if (top.size() >= 2) {
      //caffe_copy(num, target_logits_.gpu_data(), top[1]->mutable_gpu_data());
      //caffe_copy(num, max_negative_logits_.gpu_data(), top[1]->mutable_gpu_data()+num);
      //caffe_copy(num, margins_.gpu_data(), top[1]->mutable_gpu_data()+num*2);
      //caffe_copy(num, sum_exp_negative_logits_.gpu_data(), top[1]->mutable_gpu_data()+num*3);
      top[1]->mutable_cpu_data()[0] = mean_target;
      top[1]->mutable_cpu_data()[1] = mean_max_negative;
      top[1]->mutable_cpu_data()[2] = mean_margin;
      top[1]->mutable_cpu_data()[3] = mean_LSE;
    }

    caffe_copy(count, cos_data, top_data);

    // NOLINT_NEXT_LINE(whitespace/operators)
    RadialRemovalMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, cos_data, label_data, top_data, margins_.gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void RadialRemovalMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    int num = bottom[0]->num();
    int count = bottom[0]->count();

    //caffe_set(bottom[2]->count(), Dtype(0), bottom[2]->mutable_gpu_diff());
    if (propagate_down[0]) {
      caffe_copy(count, top_diff, bottom_diff);
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(RadialRemovalMarginLayer);


}  // namespace caffe
