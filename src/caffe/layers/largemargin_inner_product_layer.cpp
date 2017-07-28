#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/largemargin_inner_product_layer.hpp"

namespace caffe {

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Number of labels must match number of output; "
      << "DO NOT support multi-label this version."
      << "e.g., if prediction shape is (M X N), "
      << "label count (number of labels) must be M, "
      << "with integer values in {0, 1, ..., N-1}.";

  type_ = this->layer_param_.largemargin_inner_product_param().type();
  iter_ = this->layer_param_.largemargin_inner_product_param().iteration();
  lambda_ = (Dtype)0.;

  if (bottom.size() == 2) {
    const int num_output = this->layer_param_.largemargin_inner_product_param().num_output();
    N_ = num_output;
  }
  else {
    N_ = bottom[2]->num();
  }
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.largemargin_inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0 || bottom.size() == 3) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.largemargin_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  if (bottom.size() == 2) this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.largemargin_inner_product_param().axis());
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

  // if needed, reshape top[1] to output lambda
  vector<int> lambda_shape(1, 1);
  top[1]->Reshape(lambda_shape);
  
  // common variables
  vector<int> shape_1_X_M(1, M_);
  x_norm_.Reshape(shape_1_X_M);
  vector<int> shape_1_X_N(1, N_);
  w_norm_.Reshape(shape_1_X_N);

  sign_0_.Reshape(top_shape);
  cos_theta_.Reshape(top_shape);

  // optional temp variables
  switch (type_) {
  case LargeMarginInnerProductParameter_LargeMarginType_SINGLE:
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE:
    cos_theta_quadratic_.Reshape(top_shape);
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE:
    cos_theta_quadratic_.Reshape(top_shape);
    cos_theta_cubic_.Reshape(top_shape);
    sign_1_.Reshape(top_shape);
    sign_2_.Reshape(top_shape);
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE:
    cos_theta_quadratic_.Reshape(top_shape);
    cos_theta_cubic_.Reshape(top_shape);
    cos_theta_quartic_.Reshape(top_shape);
    sign_3_.Reshape(top_shape);
    sign_4_.Reshape(top_shape);
    break;
  default:
    LOG(FATAL) << "Unknown L-Softmax type.";
  }
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  iter_ += (Dtype)1.;
  Dtype base_ = this->layer_param_.largemargin_inner_product_param().base();
  Dtype gamma_ = this->layer_param_.largemargin_inner_product_param().gamma();
  Dtype power_ = this->layer_param_.largemargin_inner_product_param().power();
  Dtype lambda_min_ = this->layer_param_.largemargin_inner_product_param().lambda_min();
  lambda_ = base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
  lambda_ = std::max(lambda_, lambda_min_);
  top[1]->mutable_cpu_data()[0] = lambda_;

  /************************* common variables *************************/
  // x_norm = |x|
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* mutable_x_norm_data = x_norm_.mutable_cpu_data();
  for (int i = 0; i < M_; i++) {
    mutable_x_norm_data[i] = sqrt(caffe_cpu_dot(K_, bottom_data + i * K_, bottom_data + i * K_));
  }
  const Dtype* weight = bottom.size() == 3 ? bottom[2]->cpu_data() : this->blobs_[0]->cpu_data();
  Dtype* mutable_w_norm_data = w_norm_.mutable_cpu_data();
  for (int i = 0; i < N_; i++) {
    mutable_w_norm_data[i] = sqrt(caffe_cpu_dot(K_, weight + i * K_, weight + i * K_));
  }
  // cos_theta = x'w/(|x|*|w|)
  Blob<Dtype> xw_norm_product_;
  xw_norm_product_.Reshape(cos_theta_.shape());
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
      x_norm_.cpu_data(), w_norm_.cpu_data(), (Dtype)0., xw_norm_product_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., cos_theta_.mutable_cpu_data());
  caffe_add_scalar(M_ * N_, (Dtype)0.000000001, xw_norm_product_.mutable_cpu_data());
  caffe_div(M_ * N_, cos_theta_.cpu_data(), xw_norm_product_.cpu_data(), cos_theta_.mutable_cpu_data());
  // sign_0 = sign(cos_theta)
  caffe_cpu_sign(M_ * N_, cos_theta_.cpu_data(), sign_0_.mutable_cpu_data());

  /************************* optional variables *************************/
  switch (type_) {
  case LargeMarginInnerProductParameter_LargeMarginType_SINGLE:
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE:
    // cos_theta_quadratic
    caffe_powx(M_ * N_, cos_theta_.cpu_data(), (Dtype)2., cos_theta_quadratic_.mutable_cpu_data());
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE:
    // cos_theta_quadratic && cos_theta_cubic
    caffe_powx(M_ * N_, cos_theta_.cpu_data(), (Dtype)2., cos_theta_quadratic_.mutable_cpu_data());
    caffe_powx(M_ * N_, cos_theta_.cpu_data(), (Dtype)3., cos_theta_cubic_.mutable_cpu_data());
    // sign_1 = sign(abs(cos_theta) - 0.5)
    caffe_abs(M_ * N_, cos_theta_.cpu_data(), sign_1_.mutable_cpu_data());
    caffe_add_scalar(M_ * N_, -(Dtype)0.5, sign_1_.mutable_cpu_data());
    caffe_cpu_sign(M_ * N_, sign_1_.cpu_data(), sign_1_.mutable_cpu_data());
    // sign_2 = sign_0 * (1 + sign_1) - 2
    caffe_copy(M_ * N_, sign_1_.cpu_data(), sign_2_.mutable_cpu_data());
    caffe_add_scalar(M_ * N_, (Dtype)1., sign_2_.mutable_cpu_data());
    caffe_mul(M_ * N_, sign_0_.cpu_data(), sign_2_.cpu_data(), sign_2_.mutable_cpu_data());
    caffe_add_scalar(M_ * N_, - (Dtype)2., sign_2_.mutable_cpu_data());
    break;
  case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE:
    // cos_theta_quadratic && cos_theta_cubic && cos_theta_quartic
    caffe_powx(M_ * N_, cos_theta_.cpu_data(), (Dtype)2., cos_theta_quadratic_.mutable_cpu_data());
    caffe_powx(M_ * N_, cos_theta_.cpu_data(), (Dtype)3., cos_theta_cubic_.mutable_cpu_data());
    caffe_powx(M_ * N_, cos_theta_.cpu_data(), (Dtype)4., cos_theta_quartic_.mutable_cpu_data());
    // sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
    caffe_copy(M_ * N_, cos_theta_quadratic_.cpu_data(), sign_3_.mutable_cpu_data());
    caffe_scal(M_ * N_, (Dtype)2., sign_3_.mutable_cpu_data());
    caffe_add_scalar(M_ * N_, (Dtype)-1., sign_3_.mutable_cpu_data());
    caffe_cpu_sign(M_ * N_, sign_3_.cpu_data(), sign_3_.mutable_cpu_data());
    caffe_mul(M_ * N_, sign_0_.cpu_data(), sign_3_.cpu_data(), sign_3_.mutable_cpu_data());
    // sign_4 = 2 * sign_0 + sign_3 - 3
    caffe_copy(M_ * N_, sign_0_.cpu_data(), sign_4_.mutable_cpu_data());
    caffe_scal(M_ * N_, (Dtype)2., sign_4_.mutable_cpu_data());
    caffe_add(M_ * N_, sign_4_.cpu_data(), sign_3_.cpu_data(), sign_4_.mutable_cpu_data());
    caffe_add_scalar(M_ * N_, - (Dtype)3., sign_4_.mutable_cpu_data());
    break;
  default:
    LOG(FATAL) << "Unknown L-Softmax type.";
  }

  /************************* Forward *************************/
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* xw_norm_product_data = xw_norm_product_.cpu_data();
    switch (type_) {
  case LargeMarginInnerProductParameter_LargeMarginType_SINGLE: {
    break;
  }
  case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE: {
  	const Dtype* sign_0_data = sign_0_.cpu_data();
  	const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
    // the label[i]_th top_data
    for (int i = 0; i < M_; i++) {
      const int label_value = static_cast<int>(label[i]);
      // |x| * (2 * sign_0 * cos_theta_quadratic - 1)
      top_data[i * N_ + label_value] = xw_norm_product_data[i * N_ + label_value] * 
                                       ((Dtype)2. * sign_0_data[i * N_ + label_value] * 
                                       cos_theta_quadratic_data[i * N_ + label_value] - (Dtype)1.);
    }
    // + lambda * x'w
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, lambda_,
      bottom_data, weight, (Dtype)1., top_data);
    // * 1 / (1 + lambda)
    caffe_scal(M_ * N_, (Dtype)1./((Dtype)1. + lambda_), top_data);
    break;
  }
  case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE: {
  	const Dtype* sign_1_data = sign_1_.cpu_data();
    const Dtype* sign_2_data = sign_2_.cpu_data();
    const Dtype* cos_theta_data = cos_theta_.cpu_data();
    const Dtype* cos_theta_cubic_data = cos_theta_cubic_.cpu_data();
    // the label[i]_th output
    for (int i = 0; i < M_; i++) {
      const int label_value = static_cast<int>(label[i]);
      // |x| * (sign_1 * (4 * cos_theta_cubic - 3 * cos_theta) + sign_2)
      top_data[i * N_ + label_value] = xw_norm_product_data[i * N_ + label_value] * 
                                      (sign_1_data[i * N_ + label_value] * ((Dtype)4. * 
                                      	cos_theta_cubic_data[i * N_ + label_value] - 
                                       (Dtype)3. * cos_theta_data[i * N_ + label_value]) + 
                                       sign_2_data[i * N_ + label_value]);
    }
    // + lambda * x'w
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, lambda_,
      bottom_data, weight, (Dtype)1., top_data);
    // / (1 + lambda)
    caffe_scal(M_ * N_, (Dtype)1./((Dtype)1. + lambda_), top_data);
    break;
  }
  case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE: {
  	const Dtype* sign_3_data = sign_3_.cpu_data();
    const Dtype* sign_4_data = sign_4_.cpu_data();
    const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
    const Dtype* cos_theta_quartic_data = cos_theta_quartic_.cpu_data();
    // the label[i]_th output
    for (int i = 0; i < M_; i++) {
      const int label_value = static_cast<int>(label[i]);
      // // |x| * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
      top_data[i * N_ + label_value] = xw_norm_product_data[i * N_ + label_value] * 
                                       (sign_3_data[i * N_ + label_value] * ((Dtype)8. * 
                                       	cos_theta_quartic_data[i * N_ + label_value] - 
                                        (Dtype)8. * cos_theta_quadratic_data[i * N_ + label_value] + 
                                        (Dtype)1.) + sign_4_data[i * N_ + label_value]);
    }
    // + lambda * x'w
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, lambda_,
      bottom_data, weight, (Dtype)1., top_data);
    // / (1 + lambda)
    caffe_scal(M_ * N_, (Dtype)1./((Dtype)1. + lambda_), top_data);
    break;
  }
  default: {
    LOG(FATAL) << "Unknown L-Softmax type.";
  }
  }
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //Dtype base_ = this->layer_param_.largemargin_inner_product_param().base();
  //Dtype gamma_ = this->layer_param_.largemargin_inner_product_param().gamma();
  //Dtype power_ = this->layer_param_.largemargin_inner_product_param().power();
  //Dtype lambda_ = base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
  //Dtype lambda_min_ = this->layer_param_.largemargin_inner_product_param().lambda_min();
  //lambda_ = std::max(lambda_, lambda_min_);
  // |x|/|w|
  Blob<Dtype> inv_w_norm_;
  inv_w_norm_.Reshape(w_norm_.shape());
  Blob<Dtype> xw_norm_ratio_;
  xw_norm_ratio_.Reshape(cos_theta_.shape());
  caffe_add_scalar(N_, (Dtype)0.000000001, w_norm_.mutable_cpu_data());
  caffe_set(N_, (Dtype)1., inv_w_norm_.mutable_cpu_data());
  caffe_div(N_, inv_w_norm_.cpu_data(), w_norm_.cpu_data(), inv_w_norm_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
      x_norm_.cpu_data(), inv_w_norm_.cpu_data(), (Dtype)0., xw_norm_ratio_.mutable_cpu_data());

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weight = bottom.size() == 3 ? bottom[2]->cpu_data() : this->blobs_[0]->cpu_data();
 
  // Gradient with respect to weight
  if ((bottom.size()==3 && propagate_down[2]) || this->param_propagate_down_[0]) {
    Dtype* weight_diff = bottom.size() == 3 ? bottom[2]->mutable_cpu_diff() : this->blobs_[0]->mutable_cpu_diff();
    const Dtype* xw_norm_ratio_data = xw_norm_ratio_.cpu_data();
    switch (type_) {
    case LargeMarginInnerProductParameter_LargeMarginType_SINGLE: {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
      break;
    }
    case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE: {
      const Dtype* sign_0_data = sign_0_.cpu_data();
      const Dtype* cos_theta_data = cos_theta_.cpu_data();
      const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
      for (int i = 0; i < N_; i++) {
        for (int j = 0; j < M_; j++) {
          const int label_value = static_cast<int>(label[j]);
          if (label_value != i) {
            // 1 / (1 + lambda) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i], 
                            bottom_data + j * K_, (Dtype)1., weight_diff + i * K_);
          } else {
            // 1 / (1 + lambda) * 4 * sign_0 * cos_theta * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i] * 
            	                (Dtype)4. * sign_0_data[j * N_ + i] * cos_theta_data[j * N_ + i],
                            bottom_data + j * K_, (Dtype)1., weight_diff + i * K_);
            // 1 / (1 + lambda) * (-|x|/|w|) * (2 * sign_0 * cos_theta_quadratic + 1) * w
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i] * 
            	                (-xw_norm_ratio_data[j * N_ + i]) * ((Dtype)2. * sign_0_data[j * N_ + i] * 
            	                cos_theta_quadratic_data[j * N_ + i] + (Dtype)1.), 
                            weight + i * K_, (Dtype)1., weight_diff + i * K_);
          }
        }
      }
      // + lambda/(1 + lambda) * x
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, lambda_/((Dtype)1. + lambda_),
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
      break;
    }
    case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE: {
      const Dtype* sign_1_data = sign_1_.cpu_data();
      const Dtype* sign_2_data = sign_2_.cpu_data();
      const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
      const Dtype* cos_theta_cubic_data = cos_theta_cubic_.cpu_data();
      for (int i = 0; i < N_; i++) {
        for (int j = 0; j < M_; j++) {
          const int label_value = static_cast<int>(label[j]);
          if (label_value != i) {
            // 1 / (1 + lambda) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i], 
                            bottom_data + j * K_, (Dtype)1., weight_diff + i * K_);
          } else {
            // 1 / (1 + lambda) * sign_1 * (12 * cos_theta_quadratic - 3) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i] * 
            	                sign_1_data[j * N_ + i] * ((Dtype)12. * cos_theta_quadratic_data[j * N_ + i] - 
            	                (Dtype)3.),
                            bottom_data + j * K_, (Dtype)1., weight_diff + i * K_);
            // 1 / (1 + lambda) * (-|x|/|w|) * (8 * sign_1 * cos_theta_cubic - sign_2) * w
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i] * 
            	                (-xw_norm_ratio_data[j * N_ + i]) * ((Dtype)8. * sign_1_data[j * N_ + i] * 
            	                cos_theta_cubic_data[j * N_ + i] - sign_2_data[j * N_ + i]), 
                            weight + i * K_, (Dtype)1., weight_diff + i * K_);
          }
        }
      }
      // + lambda/(1 + lambda) * x
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, lambda_/((Dtype)1. + lambda_),
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
      break;
    }
    case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE: {
      const Dtype* sign_3_data = sign_3_.cpu_data();
      const Dtype* sign_4_data = sign_4_.cpu_data();
      const Dtype* cos_theta_data = cos_theta_.cpu_data();
      const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
      const Dtype* cos_theta_cubic_data = cos_theta_cubic_.cpu_data();
      const Dtype* cos_theta_quartic_data = cos_theta_quartic_.cpu_data();
      for (int i = 0; i < N_; i++) {
        for (int j = 0; j < M_; j++) {
          const int label_value = static_cast<int>(label[j]);
          if (label_value != i) {
            // 1 / (1 + lambda) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i], 
                            bottom_data + j * K_, (Dtype)1., weight_diff + i * K_);
          } else {
            // 1 / (1 + lambda) * sign_3 * (32 * cos_theta_cubic - 16 * cos_theta) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i] * 
                                sign_3_data[j * N_ + i] * ((Dtype)32. * cos_theta_cubic_data[j * N_ + i] - 
                                (Dtype)16. * cos_theta_data[j * N_ + i]),
                            bottom_data + j * K_, (Dtype)1., weight_diff + i * K_);
            // 1 / (1 + lambda) * (-|x|/|w|) * (sign_3 * (24 * cos_theta_quartic - 8 * cos_theta_quadratic - 1) + 
            //                        sign_4) * w
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[j * N_ + i] *
                                (-xw_norm_ratio_data[j * N_ + i]) * (sign_3_data[j * N_ + i] * 
                                ((Dtype)24. * cos_theta_quartic_data[j * N_ + i] - 
                                 (Dtype)8. * cos_theta_quadratic_data[j * N_ + i] - (Dtype)1.) - 
                                sign_4_data[j * N_ + i]), 
                            weight + i * K_, (Dtype)1., weight_diff + i * K_);
          }
        }
      }
      // + lambda/(1 + lambda) * x
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, lambda_/((Dtype)1. + lambda_),
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
      break;
    }
    default: {
      LOG(FATAL) << "Unknown L-Softmax type.";
    }
    }
  }
  
  // Gradient with respect to bottom data
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* xw_norm_ratio_data = xw_norm_ratio_.cpu_data();
    caffe_set(M_ * K_, (Dtype)0., bottom_diff);
    switch (type_) {
    case LargeMarginInnerProductParameter_LargeMarginType_SINGLE: {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
      break;
    }
    case LargeMarginInnerProductParameter_LargeMarginType_DOUBLE: {
      const Dtype* sign_0_data = sign_0_.cpu_data();
      const Dtype* cos_theta_data = cos_theta_.cpu_data();
      const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
      for (int i = 0; i < M_; i++) {
        const int label_value = static_cast<int>(label[i]);
        for (int j = 0; j < N_; j++) {
          if (label_value != j) {
            // 1 / (1 + lambda) * w
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j], 
                            weight + j * K_, (Dtype)1., bottom_diff + i * K_);
          } else {
            // 1 / (1 + lambda) * 4 * sign_0 * cos_theta * w
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * 
            	                (Dtype)4. * sign_0_data[i * N_ + j] * cos_theta_data[i * N_ + j], 
                            weight + j * K_, (Dtype)1., bottom_diff + i * K_);

            // 1 / (1 + lambda) / (-|x|/|w|) * (2 * sign_0 * cos_theta_quadratic + 1) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] / 
            	                (-xw_norm_ratio_data[i * N_ + j]) * ((Dtype)2. * sign_0_data[i * N_ + j] * 
            	                cos_theta_quadratic_data[i * N_ + j] + (Dtype)1.), 
                            bottom_data + i * K_, (Dtype)1., bottom_diff + i * K_);
          }
        }
      }
      // + lambda/(1 + lambda) * w
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, lambda_/((Dtype)1. + lambda_),
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)1.,
        bottom[0]->mutable_cpu_diff());
      break;
    }
    case LargeMarginInnerProductParameter_LargeMarginType_TRIPLE: {
      const Dtype* sign_1_data = sign_1_.cpu_data();
      const Dtype* sign_2_data = sign_2_.cpu_data();
      const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
      const Dtype* cos_theta_cubic_data = cos_theta_cubic_.cpu_data();
      for (int i = 0; i < M_; i++) {
        const int label_value = static_cast<int>(label[i]);
        for (int j = 0; j < N_; j++) {
          if (label_value != j) {
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j], 
                            weight + j * K_, (Dtype)1., bottom_diff + i * K_);
          } else {
            // 1 / (1 + lambda) * sign_1 * (12 * cos_theta_quadratic - 3) * w
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * 
            	                sign_1_data[i * N_ + j] * ((Dtype)12. * cos_theta_quadratic_data[i * N_ + j] - 
            	                (Dtype)3.), 
                            weight + j * K_, (Dtype)1., bottom_diff + i * K_);

            // 1 / (1 + lambda) / (-|x|/|w|) * (8 * sign_1 * cos_theta_cubic - sign_2) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] / 
            	                (-xw_norm_ratio_data[i * N_ + j]) * ((Dtype)8. * sign_1_data[i * N_ + j] * 
            	                cos_theta_cubic_data[i * N_ + j] - sign_2_data[i * N_ +j]), 
                            bottom_data + i * K_, (Dtype)1., bottom_diff + i * K_);
          }
        }
      }
      // + lambda/(1 + lambda) * w
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, lambda_/((Dtype)1. + lambda_),
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)1.,
        bottom[0]->mutable_cpu_diff());
      break;
    }
    case LargeMarginInnerProductParameter_LargeMarginType_QUADRUPLE: {
      const Dtype* sign_3_data = sign_3_.cpu_data();
      const Dtype* sign_4_data = sign_4_.cpu_data();
      const Dtype* cos_theta_data = cos_theta_.cpu_data();
      const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_.cpu_data();
      const Dtype* cos_theta_cubic_data = cos_theta_cubic_.cpu_data();
      const Dtype* cos_theta_quartic_data = cos_theta_quartic_.cpu_data();
      for (int i = 0; i < M_; i++) {
        const int label_value = static_cast<int>(label[i]);
        for (int j = 0; j < N_; j++) {
          if (label_value != j) {
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j], 
                            weight + j * K_, (Dtype)1., bottom_diff + i * K_);
          } else {
            // 1 / (1 + lambda) * sign_3 * (32 * cos_theta_cubic - 16 * cos_theta) * w
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * 
                                sign_3_data[i * N_ + j] * ((Dtype)32. * cos_theta_cubic_data[i * N_ + j] -
                                (Dtype)16. * cos_theta_data[i * N_ + j]), 
                            weight + j * K_, (Dtype)1., bottom_diff + i * K_);
            // 1 / (1 + lambda) / (-|x|/|w|) * (sign_3 * (24 * cos_theta_quartic - 8 * cos_theta_quadratic - 1) + 
            //                        sign_4) * x
            caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] /
                                (-xw_norm_ratio_data[i * N_ + j]) * (sign_3_data[i * N_ + j] * 
                                ((Dtype)24. * cos_theta_quartic_data[i * N_ + j] - 
                                 (Dtype)8. * cos_theta_quadratic_data[i * N_ + j] - (Dtype)1.) - 
                                sign_4_data[i * N_ + j]), 
                            bottom_data + i * K_, (Dtype)1., bottom_diff + i * K_);
          }
        }
      }
      // + lambda/(1 + lambda) * w
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, lambda_/((Dtype)1. + lambda_),
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)1.,
        bottom[0]->mutable_cpu_diff());
      break;
    }
    default: {
      LOG(FATAL) << "Unknown L-Softmax type.";
    }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LargeMarginInnerProductLayer);
#endif

INSTANTIATE_CLASS(LargeMarginInnerProductLayer);
REGISTER_LAYER_CLASS(LargeMarginInnerProduct);

}  // namespace caffe
