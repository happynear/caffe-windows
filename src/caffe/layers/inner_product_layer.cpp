#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  normalize_ = this->layer_param_.inner_product_param().normalize();
  if (bottom.size() == 1) N_ = num_output;
  else N_ = bottom[1]->num();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0 || bottom.size() == 3 
      || (bottom.size() == 2 && !bias_term_)) {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else {
    int bias_index = 0;
    if (bias_term_) {
      if (bottom.size() == 2) {
        this->blobs_.resize(1);
      }
      else {
        this->blobs_.resize(2);
        bias_index = 1;
      }
    }
    else {
      this->blobs_.resize(1);
    }
    if (bottom.size() == 1) {
      // Initialize the weights
      vector<int> weight_shape(2);
      if (transpose_) {
        weight_shape[0] = K_;
        weight_shape[1] = N_;
      }
      else {
        weight_shape[0] = N_;
        weight_shape[1] = K_;
      }
      this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
      // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }
    
    // If necessary, intiialize and fill the bias term
    if (bias_term_ && bottom.size() <= 2) {
      vector<int> bias_shape(1, N_);
      this->blobs_[bias_index].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[bias_index].get());
    }
  }  // parameter initialization
  if (bottom.size() == 1) this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  if (bottom.size() >= 2) N_ = bottom[1]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_ && bottom.size() <= 2) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  if (bottom.size() == 1 && normalize_) {
    vector<int> weight_norm_shape(1, N_);
    weight_norm_.Reshape(weight_norm_shape);
    caffe_set(N_, Dtype(0), weight_norm_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = bottom.size() >= 2 ? bottom[1]->cpu_data() : this->blobs_[0]->cpu_data();
  if (normalize_ && bottom.size() == 1) {
    Dtype* mutable_weight = this->blobs_[0]->mutable_cpu_data();
    Dtype sum_sq;
    for (int n = 0; n < N_; n++) {
      sum_sq = caffe_cpu_dot(K_, weight + n*K_, weight + n*K_) + 1e-6;
      caffe_scal<Dtype>(K_, Dtype(1) / sqrt(sum_sq), mutable_weight + n*K_);
    }
  }
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        bottom.size() == 3 ? bottom[2]->cpu_data() : this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = bottom.size() >= 2 ? bottom[1]->cpu_data() : this->blobs_[0]->cpu_data();
  if ((bottom.size() == 1 && this->param_propagate_down_[0])||
    (bottom.size() >= 2 && propagate_down[1])){
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* weight_diff = bottom.size() >= 2 ? bottom[1]->mutable_cpu_diff() : this->blobs_[0]->mutable_cpu_diff();
    if (bottom.size() >= 2) {
      if (transpose_) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              K_, N_, M_,
                              (Dtype)1., bottom_data, top_diff,
                              (Dtype)0., weight_diff);
      }
      else {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              N_, K_, M_,
                              (Dtype)1., top_diff, bottom_data,
                              (Dtype)0., weight_diff);
      }
    }
    else {
      if (transpose_) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              K_, N_, M_,
                              (Dtype)1., bottom_data, top_diff,
                              (Dtype)1., weight_diff);
      }
      else {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              N_, K_, M_,
                              (Dtype)1., top_diff, bottom_data,
                              (Dtype)1., weight_diff);
      }
    }
  }
  if (bias_term_ && (this->param_propagate_down_[1] || 
                     (bottom.size() == 3 && propagate_down[2]))) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        bottom.size()==3? bottom[2]->mutable_cpu_diff() : this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
