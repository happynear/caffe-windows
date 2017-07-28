#include <vector>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/largemargin_inner_product_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void Compute_bottom_norm_gpu(int nthreads, const int K_,
          const Dtype* bottom, Dtype* x_norm) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype sum_sqaure = (Dtype)0.;
    for (int i = 0; i < K_; i++) {
      sum_sqaure += bottom[index * K_ + i] * bottom[index * K_ + i];
    }
    x_norm[index] = sqrt(sum_sqaure);
  }
}
template <typename Dtype>
__global__ void Compute_weight_norm_gpu(int nthreads, const int K_,
          const Dtype* weight, Dtype* w_norm) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype sum_sqaure = (Dtype)0.;
    for (int i = 0; i < K_; i++) {
      sum_sqaure += weight[index * K_ + i] * weight[index * K_ + i];
    }
    w_norm[index] = sqrt(sum_sqaure);
  }
}
template <typename Dtype>
__global__ void Compute_cos_theta_gpu(int nthreads, const int N_,
          const Dtype* x_norm, const Dtype* w_norm, Dtype* cos_theta) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / N_;
    const int j = index % N_;
    cos_theta[index] = cos_theta[index] / x_norm[i] / w_norm[j];
  }
}

template <typename Dtype>
__global__ void Compute_sign_1_gpu(int nthreads, const Dtype* cos_theta, Dtype* sign_1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_1[index] = abs(cos_theta[index]) - (Dtype)0.5;
  }
}

template <typename Dtype>
__global__ void Compute_sign_2_gpu(int nthreads, const Dtype* sign_0, 
	      const Dtype* sign_1, Dtype* sign_2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_2[index] = sign_0[index] * ((Dtype)1. + sign_1[index]) - (Dtype)2.;
  }
}

template <typename Dtype>
__global__ void Compute_sign_3_gpu(int nthreads, const Dtype* sign_0, 
	      const Dtype* cos_theta_quadratic, Dtype* sign_3) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_3[index] = sign_0[index] * ((Dtype)2. * cos_theta_quadratic[index] - (Dtype)1.);
  }
}

template <typename Dtype>
__global__ void Compute_sign_4_gpu(int nthreads, const Dtype* sign_0, 
	      const Dtype* sign_3, Dtype* sign_4) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_4[index] = (Dtype)2. * sign_0[index] + sign_3[index] - (Dtype)3.;
  }
}

template <typename Dtype>
__global__ void LargeMargin_double_forward_gpu(int nthreads, const int N_, Dtype lambda,
            const Dtype* label, const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_0, 
            const Dtype* cos_theta_quadratic, Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    const int i = index / N_;
    const int j = index % N_;
    const int label_value = static_cast<int>(label[i]);
    if (label_value == j) {
      top[index] *= lambda;
      top[index] += x_norm[i] * w_norm[j] * ((Dtype)2. * sign_0[index] * cos_theta_quadratic[index] - 
      	                         (Dtype)1.);
      top[index] /= ((Dtype)1. + lambda);
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_triple_forward_gpu(int nthreads, const int N_, Dtype lambda,
            const Dtype* label, const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_1, 
            const Dtype* sign_2, const Dtype* cos_theta, const Dtype* cos_theta_cubic,
            Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    const int i = index / N_;
    const int j = index % N_;
    const int label_value = static_cast<int>(label[i]);
    if (label_value == j) {
      top[index] *= lambda;
      top[index] += x_norm[i] * w_norm[j] * (sign_1[index] * ((Dtype)4. * cos_theta_cubic[index] - 
      	                        (Dtype)3. * cos_theta[index]) + sign_2[index]);
      top[index] /= ((Dtype)1. + lambda);
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_quadruple_forward_gpu(int nthreads, const int N_, Dtype lambda,
            const Dtype* label, const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_3, 
            const Dtype* sign_4, const Dtype* cos_theta_quadratic, const Dtype* cos_theta_quartic,
            Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    const int i = index / N_;
    const int j = index % N_;
    const int label_value = static_cast<int>(label[i]);
    if (label_value == j) {
      top[index] *= lambda;
      top[index] += x_norm[i] * w_norm[j] * (sign_3[index] * ((Dtype)8. * cos_theta_quartic[index] - 
      	            (Dtype)8. * cos_theta_quadratic[index] + (Dtype)1.) + sign_4[index]);
      top[index] /= ((Dtype)1. + lambda);
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_weight_double_backward_gpu(int nthreads, const int M_, 
            const int N_, const int K_, Dtype lambda, const Dtype* weight,
            const Dtype* bottom, const Dtype* top_diff, const Dtype* label,
            const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_0, const Dtype* cos_theta,
            const Dtype* cos_theta_quadratic, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / K_;
    const int j = index % K_;
    Dtype coeff_x = (Dtype)0.;
    Dtype coeff_w = (Dtype)0.;
    for (int m = 0; m < M_; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value != i) {
        weight_diff[index] += top_diff[m * N_ + i] * bottom[m * K_ + j];
      } else {
        coeff_x = (Dtype)4. * sign_0[m * N_ + i] * cos_theta[m * N_ + i];
        coeff_w = - x_norm[m] / w_norm[i] * ((Dtype)2. * sign_0[m * N_ + i] *  
                     cos_theta_quadratic[m * N_ + i] + (Dtype)1.);
        weight_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[m * N_ + i] * 
                              (coeff_x * bottom[m * K_ + j] + coeff_w * weight[index]);
        weight_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[m * N_ + i] * bottom[m * K_ + j]; 
      }
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_weight_triple_backward_gpu(int nthreads, const int M_, 
            const int N_, const int K_, Dtype lambda, const Dtype* weight,
            const Dtype* bottom, const Dtype* top_diff, const Dtype* label,
            const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_1, const Dtype* sign_2, 
            const Dtype* cos_theta_quadratic, const Dtype* cos_theta_cubic, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / K_;
    const int j = index % K_;
    Dtype coeff_x = (Dtype)0.;
    Dtype coeff_w = (Dtype)0.;
    for (int m = 0; m < M_; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value != i) {
        weight_diff[index] += top_diff[m * N_ + i] * bottom[m * K_ + j];
      } else {
        coeff_x = sign_1[m * N_ + i] * ((Dtype)12. * cos_theta_quadratic[m * N_ + i] - (Dtype)3.);
        coeff_w = - x_norm[m] / w_norm[i] * ((Dtype)8. * sign_1[m * N_ + i] * 
        	        cos_theta_cubic[m * N_ + i] - sign_2[m * N_ + i]);
        weight_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[m * N_ + i] * 
                              (coeff_x * bottom[m * K_ + j] + coeff_w * weight[index]);
        weight_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[m * N_ + i] * bottom[m * K_ + j];        
      }
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_weight_quadruple_backward_gpu(int nthreads, const int M_, 
            const int N_, const int K_, Dtype lambda, const Dtype* weight,
            const Dtype* bottom, const Dtype* top_diff, const Dtype* label,
            const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_3, 
            const Dtype* sign_4, const Dtype* cos_theta, const Dtype* cos_theta_quadratic, 
            const Dtype* cos_theta_cubic, const Dtype* cos_theta_quartic, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / K_;
    const int j = index % K_;
    Dtype coeff_x = (Dtype)0.;
    Dtype coeff_w = (Dtype)0.;
    for (int m = 0; m < M_; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value != i) {
        weight_diff[index] += top_diff[m * N_ + i] * bottom[m * K_ + j];
      } else {
        coeff_x = sign_3[m * N_ + i] * ((Dtype)32. * cos_theta_cubic[m * N_ + i] - (Dtype)16. * cos_theta[m * N_ + i]);
        coeff_w = - x_norm[m] / w_norm[i] * (sign_3[m * N_ + i] * ((Dtype)24. * cos_theta_quartic[m * N_ + i] - 
                    (Dtype)8. * cos_theta_quadratic[m * N_ + i] - 1) - sign_4[m * N_ + i]);
        weight_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[m * N_ + i] * 
                              (coeff_x * bottom[m * K_ + j] + coeff_w * weight[index]);
        weight_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[m * N_ + i] * bottom[m * K_ + j]; 
      }
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_bottom_double_backward_gpu(int nthreads, const int N_, const int K_, Dtype lambda,
            const Dtype* bottom, const Dtype* weight, const Dtype* top_diff, const Dtype* label,
            const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_0, const Dtype* cos_theta,
            const Dtype* cos_theta_quadratic, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / K_;
    const int j = index % K_;
    Dtype coeff_w = (Dtype)0.;
    Dtype coeff_x = (Dtype)0.;
    bottom_diff[index] = (Dtype)0.;
    const int label_value = static_cast<int>(label[i]);
    for (int n = 0; n < N_; n++) {
      if (label_value != n) {
        bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
      } else {
        coeff_w = (Dtype)4. * sign_0[i * N_ + n] * cos_theta[i * N_ + n];
        coeff_x = - w_norm[n] / x_norm[i] * ((Dtype)2. * sign_0[i * N_ + n] *  
                     cos_theta_quadratic[i * N_ + n] + (Dtype)1.);
        bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
                              (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
        bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
      }
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_bottom_triple_backward_gpu(int nthreads, const int N_, const int K_, Dtype lambda,
            const Dtype* bottom, const Dtype* weight, const Dtype* top_diff, const Dtype* label,
            const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_1, const Dtype* sign_2, 
            const Dtype* cos_theta_quadratic, const Dtype* cos_theta_cubic, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / K_;
    const int j = index % K_;
    Dtype coeff_w = (Dtype)0.;
    Dtype coeff_x = (Dtype)0.;
    bottom_diff[index] = (Dtype)0.;
    const int label_value = static_cast<int>(label[i]);
    for (int n = 0; n < N_; n++) {
      if (label_value != n) {
        bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
      } else {
        coeff_w = sign_1[i * N_ + n] * ((Dtype)12. * cos_theta_quadratic[i * N_ + n] - (Dtype)3.);
        coeff_x = - w_norm[n] / x_norm[i] * ((Dtype)8. * sign_1[i * N_ + n] * cos_theta_cubic[i * N_ + n] - 
                    sign_2[i * N_ + n]);
        bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
                              (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
        bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
      }
    }
  }
}

template <typename Dtype>
__global__ void LargeMargin_bottom_quadruple_backward_gpu(int nthreads, const int N_, const int K_, Dtype lambda,
            const Dtype* bottom, const Dtype* weight, const Dtype* top_diff, const Dtype* label,
            const Dtype* x_norm, const Dtype* w_norm, const Dtype* sign_3, 
            const Dtype* sign_4, const Dtype* cos_theta, const Dtype* cos_theta_quadratic, 
            const Dtype* cos_theta_cubic, const Dtype* cos_theta_quartic, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / K_;
    const int j = index % K_;
    Dtype coeff_w = (Dtype)0.;
    Dtype coeff_x = (Dtype)0.;
    bottom_diff[index] = (Dtype)0.;
    const int label_value = static_cast<int>(label[i]);
    for (int n = 0; n < N_; n++) {
      if (label_value != n) {
        bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
      } else {
        coeff_w = sign_3[i * N_ + n] * ((Dtype)32. * cos_theta_cubic[i * N_ + n] - (Dtype)16. * cos_theta[i * N_ + n]);
        coeff_x = - w_norm[n] / x_norm[i] * (sign_3[i * N_ + n] * ((Dtype)24. * cos_theta_quartic[i * N_ + n] - 
                    (Dtype)8. * cos_theta_quadratic[i * N_ + n] - 1) - sign_4[i * N_ + n]);
        bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
                              (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
        bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
      }
    }
  }
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  iter_ += (Dtype)1.;
  Dtype base_ = this->layer_param_.largemargin_inner_product_param().base();
  Dtype gamma_ = this->layer_param_.largemargin_inner_product_param().gamma();
  Dtype power_ = this->layer_param_.largemargin_inner_product_param().power();
  Dtype lambda_min_ = this->layer_param_.largemargin_inner_product_param().lambda_min();
  lambda_ = base_ * powf(((Dtype)1. + gamma_ * iter_), -power_);
  lambda_ = max(lambda_, lambda_min_);
  top[1]->mutable_cpu_data()[0] = lambda_;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = bottom.size() == 3 ? bottom[2]->gpu_data() : this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* label = bottom[1]->gpu_data();

  /************************* common variables *************************/
  // x_norm_ = |x|
  int nthreads = M_;
  Compute_bottom_norm_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom_data,
                                x_norm_.mutable_gpu_data());
  // w_norm = |w|
  nthreads = N_;
  Compute_weight_norm_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, weight,
                                w_norm_.mutable_gpu_data());

  nthreads = M_ * N_;
  // cos_theta = x'w / (|x|*|w|)
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., cos_theta_.mutable_gpu_data());
  Compute_cos_theta_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, x_norm_.gpu_data(), w_norm_.gpu_data(), 
      	                        cos_theta_.mutable_gpu_data());
  // sign_0
  caffe_gpu_sign(M_ * N_, cos_theta_.gpu_data(), sign_0_.mutable_gpu_data());
  
  /************************* optional variables *************************/
  switch (type_) {
  case LargeMarginInnerProductParameter_LargeMarginType_SINGLE:
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE:
    // cos_theta_quadratic
    caffe_gpu_powx(M_ * N_, cos_theta_.gpu_data(), (Dtype)2., cos_theta_quadratic_.mutable_gpu_data());
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE:
    // cos_theta_quadratic && cos_theta_cubic
    caffe_gpu_powx(M_ * N_, cos_theta_.gpu_data(), (Dtype)2., cos_theta_quadratic_.mutable_gpu_data());
    caffe_gpu_powx(M_ * N_, cos_theta_.gpu_data(), (Dtype)3., cos_theta_cubic_.mutable_gpu_data());
    // sign_1 = sign(abs(cos_theta) - 0.5)
    Compute_sign_1_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, cos_theta_.gpu_data(), sign_1_.mutable_gpu_data());
    caffe_gpu_sign(M_ * N_, sign_1_.gpu_data(), sign_1_.mutable_gpu_data());
    // sign_2 = sign_0 * (1 + sign_1) - 2
    Compute_sign_2_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, sign_0_.gpu_data(),
                                sign_1_.gpu_data(), sign_2_.mutable_gpu_data());
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE:
    // cos_theta_quadratic && cos_theta_cubic && cos_theta_quartic
    caffe_gpu_powx(M_ * N_, cos_theta_.gpu_data(), (Dtype)2., cos_theta_quadratic_.mutable_gpu_data());
    caffe_gpu_powx(M_ * N_, cos_theta_.gpu_data(), (Dtype)3., cos_theta_cubic_.mutable_gpu_data());
    caffe_gpu_powx(M_ * N_, cos_theta_.gpu_data(), (Dtype)4., cos_theta_quartic_.mutable_gpu_data());
    // sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
    Compute_sign_3_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, sign_0_.gpu_data(), cos_theta_quadratic_.gpu_data(),
                                sign_3_.mutable_gpu_data());
    caffe_gpu_sign(M_ * N_, sign_3_.gpu_data(), sign_3_.mutable_gpu_data());
    // sign_4 = 2 * sign_0 + sign_3 - 3
    Compute_sign_4_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, sign_0_.gpu_data(),
                                sign_3_.gpu_data(), sign_4_.mutable_gpu_data());
    break;
  default:
    LOG(FATAL) << "Unknown LargeMargin type.";
  }

  /************************* Forward *************************/
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  switch (type_) {
  case LargeMarginInnerProductParameter_LargeMarginType_SINGLE:
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE:
    // caffe_gpu_memcpy(M_ * N_, cos_theta_.gpu_data(), top_data);
    LargeMargin_double_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, lambda_, label, x_norm_.gpu_data(), w_norm_.gpu_data(),
      	                        sign_0_.gpu_data(), cos_theta_quadratic_.gpu_data(), top_data);
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE:
    LargeMargin_triple_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, lambda_, label, x_norm_.gpu_data(), w_norm_.gpu_data(),
                                sign_1_.gpu_data(), sign_2_.gpu_data(), cos_theta_.gpu_data(), 
                                cos_theta_cubic_.gpu_data(), top_data);
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE:
    LargeMargin_quadruple_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, lambda_, label, x_norm_.gpu_data(), w_norm_.gpu_data(),
      	                        sign_3_.gpu_data(), sign_4_.gpu_data(), cos_theta_quadratic_.gpu_data(), 
                                cos_theta_quartic_.gpu_data(), top_data);
    break;
  default:
    LOG(FATAL) << "Unknown LargeMargin type.";
  }
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* weight = bottom.size() == 3 ? bottom[2]->gpu_data() : this->blobs_[0]->gpu_data();

  if ((bottom.size() == 3 && propagate_down[2]) || this->param_propagate_down_[0]) {
    Dtype* weight_diff = bottom.size() == 3 ? bottom[2]->mutable_gpu_diff() : this->blobs_[0]->mutable_gpu_diff();
    // Gradient with respect to weight
    int nthreads = N_ * K_;
    switch (type_) {
    case LargeMarginInnerProductParameter_LargeMarginType_SINGLE:
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., weight_diff);
      break;
    case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE:
      LargeMargin_weight_double_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, N_, K_, lambda_, weight, bottom_data, top_diff, label,
                                  x_norm_.gpu_data(), w_norm_.gpu_data(), sign_0_.gpu_data(), 
                                  cos_theta_.gpu_data(), cos_theta_quadratic_.gpu_data(), weight_diff);
      break;
    case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE:
      LargeMargin_weight_triple_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, N_, K_, lambda_, weight, bottom_data, top_diff, label,
        	                      x_norm_.gpu_data(), w_norm_.gpu_data(), sign_1_.gpu_data(), sign_2_.gpu_data(),
                                  cos_theta_quadratic_.gpu_data(), cos_theta_cubic_.gpu_data(), weight_diff);
      break;
    case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE:
      LargeMargin_weight_quadruple_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, N_, K_, lambda_, weight, bottom_data, top_diff, label,
        	                      x_norm_.gpu_data(), w_norm_.gpu_data(), sign_3_.gpu_data(), sign_4_.gpu_data(),
                                  cos_theta_.gpu_data(), cos_theta_quadratic_.gpu_data(),
                                  cos_theta_cubic_.gpu_data(), cos_theta_quartic_.gpu_data(), weight_diff);
      break;
    default:
      LOG(FATAL) << "Unknown LargeMargin type.";
    }
  }

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    // Gradient with respect to bottom data
    int nthreads = M_ * K_;
    switch (type_) {
    case LargeMarginInnerProductParameter_LargeMarginType_SINGLE:
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weight, (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
      break;
    case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE:
      LargeMargin_bottom_double_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, K_, lambda_, bottom_data, weight, top_diff, label,
                                  x_norm_.gpu_data(), w_norm_.gpu_data(), sign_0_.gpu_data(), 
                                  cos_theta_.gpu_data(), cos_theta_quadratic_.gpu_data(),                                  
                                  bottom_diff);
      break;
    case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE:
      LargeMargin_bottom_triple_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, K_, lambda_, bottom_data, weight, top_diff, label,
                                  x_norm_.gpu_data(), w_norm_.gpu_data(), sign_1_.gpu_data(), sign_2_.gpu_data(),
                                  cos_theta_quadratic_.gpu_data(), cos_theta_cubic_.gpu_data(),
                                  bottom_diff);
      break;
    case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE:
      LargeMargin_bottom_quadruple_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, K_, lambda_, bottom_data, weight, top_diff, label,
                                  x_norm_.gpu_data(), w_norm_.gpu_data(), sign_3_.gpu_data(), sign_4_.gpu_data(),
                                  cos_theta_.gpu_data(), cos_theta_quadratic_.gpu_data(),
                                  cos_theta_cubic_.gpu_data(), cos_theta_quartic_.gpu_data(),
                                  bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown LargeMargin type.";
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LargeMarginInnerProductLayer);

}  // namespace caffe
