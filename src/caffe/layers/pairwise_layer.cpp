#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PairwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().pairwise_param().coeff_size() == 0
      || this->layer_param().pairwise_param().coeff_size() == bottom.size()) <<
      "Pairwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().pairwise_param().operation()
      == PairwiseParameter_PairwiseOp_PROD
      && this->layer_param().pairwise_param().coeff_size())) <<
      "Pairwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.pairwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().pairwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().pairwise_param().coeff(i);
    }
  }
}

template <typename Dtype>
void PairwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  M_ = bottom[0]->num();
  N_ = bottom[1]->num();
  K_ = bottom[0]->channels();
  top[0]->Reshape({ M_, N_, K_ });
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.pairwise_param().operation() ==
      PairwiseParameter_PairwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(top[0]->shape());
  }
}

template <typename Dtype>
void PairwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const Dtype* bottom_data_a = bottom[0]->cpu_data();
  const Dtype* bottom_data_b = bottom[1]->cpu_data();
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case PairwiseParameter_PairwiseOp_PROD:
    for (int m = 0; m < M_; m++) {
      for (int n = 0; n < N_; n++) {
        caffe_mul(K_, bottom_data_a + m*K_, bottom_data_b + n*K_, top_data + (m*N_ + n)*K_);
      }
    }
    break;
  case PairwiseParameter_PairwiseOp_SUM:
    caffe_set(count, Dtype(0), top_data);
    for (int m = 0; m < M_; m++) {
      for (int n = 0; n < N_; n++) {
        caffe_axpy(K_, coeffs_[0], bottom_data_a + m*K_, top_data + (m*N_ + n)*K_);
        caffe_axpy(K_, coeffs_[1], bottom_data_b + n*K_, top_data + (m*N_ + n)*K_);
        //for (int k = 0; k < K_; k++) {
        //  top_data[(m*N_ + n)*K_ + k] = coeffs_[0] * bottom_data_a[m*K_ + k] + coeffs_[1] * bottom_data_b[n*K_ + k];
        //}
      }
    }
    break;
  case PairwiseParameter_PairwiseOp_MAX:
    // Initialize
    mask = max_idx_.mutable_cpu_data();
    for (int m = 0; m < M_; m++) {
      for (int n = 0; n < N_; n++) {
        for (int k = 0; k < K_; k++) {
          if (bottom_data_a[m*K_ + k] > bottom_data_b[n*K_ + k]) {
            top_data[(m*N_ + n)*K_ + k] = bottom_data_a[m*K_ + k];
            mask[(m*N_ + n)*K_ + k] = 0;
          }
          else {
            top_data[(m*N_ + n)*K_ + k] = bottom_data_b[n*K_ + k];
            mask[(m*N_ + n)*K_ + k] = 1;
          }
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown pairwise operation.";
  }
}

template <typename Dtype>
void PairwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data_a = bottom[0]->cpu_data();
  const Dtype* bottom_data_b = bottom[1]->cpu_data();
  Dtype* bottom_diff_a = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
  if (propagate_down[0]) {
    switch (op_) {
    case PairwiseParameter_PairwiseOp_PROD:
      caffe_set(bottom[0]->count(), Dtype(0), bottom_diff_a);
      for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) {
          for (int k = 0; k < K_; k++) {
            bottom_diff_a[m*K_ + k] += top_diff[(m*N_ + n)*K_ + k] * bottom_data_b[n*K_ + k];
          }
        }
      }
      break;
    case PairwiseParameter_PairwiseOp_SUM:
      caffe_set(bottom[0]->count(), Dtype(0), bottom_diff_a);
      for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) {
          for (int k = 0; k < K_; k++) {
            bottom_diff_a[m*K_ + k] += top_diff[(m*N_ + n)*K_ + k] * coeffs_[0];
          }
        }
      }
      break;
    case PairwiseParameter_PairwiseOp_MAX:
      mask = max_idx_.cpu_data();
      for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) {
          for (int k = 0; k < K_; k++) {
            if (mask[(m*N_ + n)*K_ + k] == 0) {
              bottom_diff_a[m*K_ + k] += top_diff[(m*N_ + n)*K_ + k];
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown pairwise operation.";
    }
  }
  if (propagate_down[1]) {
    switch (op_) {
    case PairwiseParameter_PairwiseOp_PROD:
      caffe_set(bottom[1]->count(), Dtype(0), bottom_diff_b);
      for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) {
          for (int k = 0; k < K_; k++) {
            bottom_diff_b[n*K_ + k] += top_diff[(m*N_ + n)*K_ + k] * bottom_data_a[m*K_ + k];
          }
        }
      }
      break;
    case PairwiseParameter_PairwiseOp_SUM:
      caffe_set(bottom[1]->count(), Dtype(0), bottom_diff_b);
      for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) {
          for (int k = 0; k < K_; k++) {
            bottom_diff_b[n*K_ + k] += top_diff[(m*N_ + n)*K_ + k] * coeffs_[1];
          }
        }
      }
      break;
    case PairwiseParameter_PairwiseOp_MAX:
      caffe_set(bottom[1]->count(), Dtype(0), bottom_diff_b);
      mask = max_idx_.cpu_data();
      for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) {
          for (int k = 0; k < K_; k++) {
            if (mask[(m*N_ + n)*K_ + k] == 1) {
              bottom_diff_b[n*K_ + k] += top_diff[(m*N_ + n)*K_ + k];
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown pairwise operation.";
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PairwiseLayer);
#endif

INSTANTIATE_CLASS(PairwiseLayer);
REGISTER_LAYER_CLASS(Pairwise);

}  // namespace caffe
