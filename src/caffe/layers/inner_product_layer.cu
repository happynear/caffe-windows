#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void kernel_channel_dot(const int num, const int dim,
                                     const Dtype* data_1, const Dtype* data_2,
                                     Dtype* channel_dot, Dtype epsilon) {
    CUDA_KERNEL_LOOP(index, num) {
      Dtype dot = 0;
      for (int d = 0; d < dim; ++d) {
        dot += data_1[index * dim + d] * data_2[index * dim + d];
      }
      channel_dot[index] = dot + epsilon;
    }
  }

  template <typename Dtype>
  __global__ void kernel_channel_scal(const int num, const int dim,
                                      const Dtype* norm_data,
                                      Dtype* input_output_data) {
    CUDA_KERNEL_LOOP(index, num * dim) {
      int n = index / dim;
      input_output_data[index] *= norm_data[n];
    }
  }

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = bottom.size() >= 2 ? bottom[1]->gpu_data() : this->blobs_[0]->gpu_data();

  if (normalize_ && bottom.size() == 1) {
    Dtype* mutable_weight = this->blobs_[0]->mutable_gpu_data();
    Dtype* weight_norm_data = weight_norm_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_dot<Dtype> << <CAFFE_GET_BLOCKS(N_),
      CAFFE_CUDA_NUM_THREADS >> >(N_, K_, weight, weight, weight_norm_data, 1e-12);
    caffe_gpu_powx(N_, weight_norm_data, Dtype(-0.5), weight_norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scal<Dtype> << <CAFFE_GET_BLOCKS(N_ * K_),
      CAFFE_CUDA_NUM_THREADS >> >(N_, K_, weight_norm_data, mutable_weight);
  }

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            bottom.size() == 3 ? bottom[2]->gpu_data() : this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            bottom.size() == 3 ? bottom[2]->gpu_data() : this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = bottom.size() >= 2 ? bottom[1]->gpu_data() : this->blobs_[0]->gpu_data();
  if ((bottom.size() == 1 && this->param_propagate_down_[0]) ||
    (bottom.size() >= 2 && propagate_down[1])) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff = bottom.size() >= 2 ? bottom[1]->mutable_gpu_diff() : this->blobs_[0]->mutable_gpu_diff();
    if (bottom.size() >= 2) {
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              K_, N_, M_,
                              (Dtype)1., bottom_data, top_diff,
                              (Dtype)0., weight_diff);
      }
      else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              N_, K_, M_,
                              (Dtype)1., top_diff, bottom_data,
                              (Dtype)0., weight_diff);
      }
    }
    else {
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              K_, N_, M_,
                              (Dtype)1., bottom_data, top_diff,
                              (Dtype)1., weight_diff);
      }
      else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              N_, K_, M_,
                              (Dtype)1., top_diff, bottom_data,
                              (Dtype)1., weight_diff);
      }
    }
    
  }
  if (bias_term_ && (this->param_propagate_down_[1] ||
    (bottom.size() == 3 && propagate_down[2]))) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.gpu_data(), (Dtype)1.,
                          bottom.size() == 3 ? bottom[2]->mutable_gpu_diff() : this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, weight,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
