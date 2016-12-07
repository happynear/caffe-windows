#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))

template <typename Dtype>
void InnerDistanceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_distance_param().num_output();
  bias_term_ = this->layer_param_.inner_distance_param().bias_term();
  transpose_ = this->layer_param_.inner_distance_param().transpose();
  distance_type_ = this->layer_param_.inner_distance_param().distance_type();
  normalize_ = this->layer_param_.inner_distance_param().normalize();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_distance_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_distance_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_distance_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerDistanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_distance_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  if (normalize_) {
    vector<int> weight_norm_shape(1, N_);
    weight_norm_.Reshape(weight_norm_shape);
    caffe_set(M_, Dtype(0), weight_norm_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerDistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  if (normalize_) {
    Dtype* mutable_weight = this->blobs_[0]->mutable_cpu_data();
    Dtype sum_sq;
    for (int n = 0; n < N_; n++) {
      sum_sq = caffe_cpu_dot(K_, weight + n*K_, weight + n*K_);
      caffe_scal<Dtype>(K_, Dtype(1) / sqrt(sum_sq), mutable_weight + n*K_);
    }
  }

  if (distance_type_ == "L2") {//tanspose = false, TODO: transpose = true
    for (int m = 0; m < M_; m++) {
      for (int n = 0; n < N_; n++) {
        top_data[m * N_ + n] = Dtype(0);
        for (int k = 0; k < K_; k++) {
          top_data[m * N_ + n] += (bottom_data[m * K_ + k] - weight[n * K_ + k]) * (bottom_data[m * K_ + k] - weight[n * K_ + k]);
        }
      }
    }
  }
  else if (distance_type_ == "L1") {
    for (int m = 0; m < M_; m++) {
      for (int n = 0; n < N_; n++) {
        top_data[m * N_ + n] = Dtype(0);
        for (int k = 0; k < K_; k++) {
          top_data[m * N_ + n] += abs(bottom_data[m * K_ + k] - weight[n * K_ + k]);
        }
      }
    }
  }
  else {
    NOT_IMPLEMENTED;
  }
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerDistanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  if (this->param_propagate_down_[0]) {
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    // Gradient with respect to weight
    if (distance_type_ == "L2") {
      for (int n = 0; n < N_; n++) {
        for (int k = 0; k < K_; k++) {
          for (int m = 0; m < M_; m++) {
            weight_diff[n * K_ + k] += top_diff[m * N_ + n] * (weight[n * K_ + k] - bottom_data[m * K_ + k]) * Dtype(2);
          }
        }
      }
    }
    else if (distance_type_ == "L1") {
      for (int n = 0; n < N_; n++) {
        for (int k = 0; k < K_; k++) {
          for (int m = 0; m < M_; m++) {
            weight_diff[n * K_ + k] += top_diff[m * N_ + n] * sign(weight[n * K_ + k] - bottom_data[m * K_ + k]);
          }
        }
      }
    }
    else {
      NOT_IMPLEMENTED;
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set<Dtype>(M_ * K_, 0, bottom_diff);
    if (distance_type_ == "L2") {
      for (int m = 0; m < M_; m++) {
        for (int k = 0; k < K_; k++) {
          for (int n = 0; n < N_; n++) {
            bottom_diff[m * K_ + k] += top_diff[m * N_ + n] * (bottom_data[m * K_ + k] - weight[n * K_ + k]) * Dtype(2);
          }
        }
      }
    }
    else if (distance_type_ == "L1") {
      for (int m = 0; m < M_; m++) {
        for (int k = 0; k < K_; k++) {
          for (int n = 0; n < N_; n++) {
            bottom_diff[m * K_ + k] += top_diff[m * N_ + n] * sign(bottom_data[m * K_ + k] - weight[n * K_ + k]);
          }
        }
      }
    }
    else {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerDistanceLayer);
#endif

INSTANTIATE_CLASS(InnerDistanceLayer);
REGISTER_LAYER_CLASS(InnerDistance);

}  // namespace caffe
