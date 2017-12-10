#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void ArcCosDegree(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      Dtype fixed_in_data = min(in[index], Dtype(1.0) - Dtype(1e-4));
      fixed_in_data = max(fixed_in_data, Dtype(-1.0) + Dtype(1e-4));
      out[index] = Dtype(acos(in[index]) / M_PI * 180.0);
    }
  }

  template <typename Dtype>
  __global__ void CreateMask(const int num, const int dim, const Dtype* label, Dtype* positive_mask, Dtype* negative_mask) {
    CUDA_KERNEL_LOOP(index, num) {
      int gt = static_cast<int>(label[index]);
      positive_mask[index*dim + gt] = Dtype(1);
      negative_mask[index*dim + gt] = Dtype(0);
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificSoftMarginForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                                 Dtype* top_data, Dtype* theta, Dtype margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      theta[index * dim + gt] = acos(bottom_data[index * dim + gt]);
      if (margin * theta[index * dim + gt] > M_PI - 1e-4) {
        theta[index * dim + gt] = M_PI - 1e-4;
      }
      top_data[index * dim + gt] = cos(margin * theta[index * dim + gt]);
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificSoftMarginBackward(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                  Dtype* bottom_diff, const Dtype* bottom_data, const Dtype* theta, const Dtype* top_data, Dtype margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      Dtype fixed_bottom_data = min(bottom_data[index * dim + gt], Dtype(1.0) - Dtype(1e-4));
      fixed_bottom_data = max(fixed_bottom_data, Dtype(-1.0) + Dtype(1e-4));
      Dtype gradient = margin * sin(margin * theta[index * dim + gt]) / sqrt(1 - fixed_bottom_data * fixed_bottom_data);
      gradient = gradient > 2 ? 2 : gradient;//bound the gradient.
      gradient = gradient < 0 ? 0 : gradient;
      bottom_diff[index * dim + gt] = top_diff[index * dim + gt] * gradient;
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                                 Dtype* top_data, Dtype cos_margin, Dtype sin_margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      Dtype fixed_bottom_data = min(bottom_data[index * dim + gt], Dtype(1.0) - Dtype(1e-4));
      fixed_bottom_data = max(fixed_bottom_data, Dtype(-1.0) + Dtype(1e-4));
      top_data[index * dim + gt] = fixed_bottom_data * cos_margin -
        sqrt(1 - fixed_bottom_data * fixed_bottom_data) * sin_margin;
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginBackward(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                  Dtype* bottom_diff, const Dtype* bottom_data, Dtype cos_margin, Dtype sin_margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      Dtype fixed_bottom_data = min(bottom_data[index * dim + gt], Dtype(1.0) - Dtype(1e-4));
      fixed_bottom_data = max(fixed_bottom_data, Dtype(-1.0) + Dtype(1e-4));
      Dtype gradient = cos_margin - fixed_bottom_data / sqrt(1 - fixed_bottom_data * fixed_bottom_data) * sin_margin;
      gradient = gradient > 2 ? 2 : gradient;//bound the gradient.
      gradient = gradient < 0 ? 0 : gradient;
      bottom_diff[index * dim + gt] = top_diff[index * dim + gt] * gradient;
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginBackwardToMargin(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                          Dtype* margin_diff, const Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      margin_diff[index] = top_diff[index * dim + gt] * sqrt(1 - top_data[index * dim + gt] * top_data[index * dim + gt]);
    }
  }

  template <typename Dtype>
  void LabelSpecificMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* margin = this->blobs_[0]->mutable_cpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (has_margin_base_ && this->phase_ == TRAIN) {
      margin[0] = margin_base_ + pow(((Dtype)1. + gamma_ * iter_), power_) - 1;
      iter_++;
    }
    if (has_margin_max_ && this->phase_ == TRAIN) {
      margin[0] = std::min(margin[0], margin_max_);
    }

    if (top.size() == 2 && auto_tune_) {
      Dtype *positive_mask_data = positive_mask.mutable_gpu_data();
      Dtype *negative_mask_data = negative_mask.mutable_gpu_data();
      caffe_gpu_set(count, Dtype(0), positive_mask_data);
      caffe_gpu_set(count, Dtype(1), negative_mask_data);
      // NOLINT_NEXT_LINE(whitespace/operators)
      CreateMask<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, label_data, positive_mask.mutable_gpu_data(), negative_mask.mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK;

      Dtype positive_mean;
      //Dtype positive_std;
      Dtype negative_mean;
      //Dtype negative_std;

      // NOLINT_NEXT_LINE(whitespace/operators)
      ArcCosDegree<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
        count, bottom_data, bottom_angle.mutable_gpu_data());
      //caffe_gpu_powx(count, bottom_angle.gpu_data(), Dtype(2), bottom_square.mutable_gpu_data());
      caffe_gpu_dot(count, bottom_angle.gpu_data(), positive_mask.gpu_data(), &positive_mean);
      //caffe_gpu_dot(count, bottom_square.gpu_data(), positive_mask.gpu_data(), &positive_std);
      caffe_gpu_dot(count, bottom_angle.gpu_data(), negative_mask.gpu_data(), &negative_mean);
      //caffe_gpu_dot(count, bottom_square.gpu_data(), negative_mask.gpu_data(), &negative_std);

      positive_mean /= num;
      //positive_std = sqrt(positive_std / num - positive_mean * positive_mean);
      negative_mean /= num * (dim - 1);
      //negative_std = sqrt(negative_std / num / (dim - 1) - negative_mean * negative_mean);

      if (this->phase_ == TEST) {
        top[1]->mutable_cpu_data()[0] = margin[0];
        top[1]->mutable_cpu_data()[1] = positive_mean;
        //top[1]->mutable_cpu_data()[2] = positive_std;
        top[1]->mutable_cpu_data()[2] = negative_mean;
        //top[1]->mutable_cpu_data()[4] = negative_std;
      }
      else {
        if (iter_ == 1) {
          margin[1] = positive_mean;
          //margin[2] = positive_std;
          margin[2] = negative_mean;
          //margin[4] = negative_std;
        }
        else {
          margin[1] = 0.99 * margin[1] + 0.01 * positive_mean;
          //margin[2] = 0.99 * margin[2] + 0.01 * positive_std;
          margin[2] = 0.99 * margin[2] + 0.01 * negative_mean;
          //margin[4] = 0.99 * margin[4] + 0.01 * negative_std;
        }

        //margin[0] = (margin[3] - margin[1]) / (margin[2] + margin[4]) * margin[2];
        margin[0] = (margin[2] - margin[1]) / 2;
        caffe_copy(3, this->blobs_[0]->cpu_data(), top[1]->mutable_cpu_data());
      }
    }
    if (bottom.size() == 3) {
      margin[0] = bottom[2]->cpu_data()[0];
    }
    if (top.size() >= 2) {
      top[1]->mutable_cpu_data()[0] = margin[0];
    }

    caffe_copy(count, bottom_data, top_data);
    if (!margin_on_test_ && this->phase_ == TEST) return;

    if (margin[0] != Dtype(0.0)) {
      if (type_ == LabelSpecificMarginParameter_MarginType_SOFT) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        LabelSpecificSoftMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
          num, dim, bottom_data, label_data, top_data, theta.mutable_gpu_data(), margin[0]);
        CUDA_POST_KERNEL_CHECK;
      }
      else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        LabelSpecificHardMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
          num, dim, bottom_data, label_data, top_data, cos(margin[0] / 180 * M_PI), sin(margin[0] / 180 * M_PI));
        CUDA_POST_KERNEL_CHECK;
      }
    }
  }

  template <typename Dtype>
  void LabelSpecificMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* label_data = bottom[1]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* top_data = top[0]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      Dtype* margin = this->blobs_[0]->mutable_cpu_data();

      int num = bottom[0]->num();
      int count = bottom[0]->count();
      int dim = count / num;

      caffe_copy(count, top_diff, bottom_diff);
      if (!margin_on_test_ && this->phase_ == TEST) return;

      if (margin[0] != Dtype(0.0)) {
        if (type_ == LabelSpecificMarginParameter_MarginType_SOFT) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          LabelSpecificSoftMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
            num, dim, top_diff, label_data, bottom_diff, bottom_data, theta.gpu_data(), top_data, margin[0]);
          CUDA_POST_KERNEL_CHECK;
        }
        else {
          // NOLINT_NEXT_LINE(whitespace/operators)
          LabelSpecificHardMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
            num, dim, top_diff, label_data, bottom_diff, bottom_data, cos(margin[0] / 180 * M_PI), sin(margin[0] / 180 * M_PI));
          CUDA_POST_KERNEL_CHECK;
          if (bottom.size() == 3 && propagate_down[3]) {
            // NOLINT_NEXT_LINE(whitespace/operators)
            LabelSpecificHardMarginBackwardToMargin<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
              num, dim, top_diff, label_data, positive_data.mutable_gpu_data(), top_data);
            CUDA_POST_KERNEL_CHECK;
            caffe_gpu_dot(num, positive_data.gpu_data(), sum_multiplier_.gpu_data(), bottom[3]->mutable_cpu_data());
          }
        }
      }
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificMarginLayer);


}  // namespace caffe
