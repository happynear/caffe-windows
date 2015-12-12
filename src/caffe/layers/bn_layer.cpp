#include <algorithm>
#include <vector>

#include "caffe/custom_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	// Figure out the dimensions
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	// extract param
	var_eps_ = this->layer_param_.bn_param().var_eps();
	decay_ = this->layer_param_.bn_param().decay();
	moving_average_ = this->layer_param_.bn_param().moving_average();

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		this->blobs_.resize(4);

		// fill scale with scale_filler
		this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1));
		shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
			this->layer_param_.bn_param().scale_filler()));
		scale_filler->Fill(this->blobs_[0].get());

		// fill shift with shift_filler
		this->blobs_[1].reset(new Blob<Dtype>(1, channels_, 1, 1));
		shared_ptr<Filler<Dtype> > shift_filler(GetFiller<Dtype>(
			this->layer_param_.bn_param().shift_filler()));
		shift_filler->Fill(this->blobs_[1].get());

		// history mean
		this->blobs_[2].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(0), this->blobs_[2]->mutable_cpu_data());

		// history variance
		this->blobs_[3].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(1), this->blobs_[3]->mutable_cpu_data());

	}  // parameter initialization
	this->param_propagate_down_.resize(this->blobs_.size(), true);
	this->param_propagate_down_[2] = false;
	this->param_propagate_down_[3] = false;
}

template <typename Dtype>
void BNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // reshape blob
	top[0]->Reshape(num_, channels_, height_, width_);
	x_norm_.Reshape(num_, channels_, height_, width_);
	x_std_.Reshape(1, channels_, 1, 1);

	// statistic
	spatial_statistic_.Reshape(num_, channels_, 1, 1);
	batch_statistic_.Reshape(1, channels_, 1, 1);

	// buffer blob
	buffer_blob_.Reshape(num_, channels_, height_, width_);

	// fill spatial multiplier
	spatial_sum_multiplier_.Reshape(1, 1, height_, width_);
	Dtype* spatial_multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
	caffe_set(spatial_sum_multiplier_.count(), Dtype(1), spatial_multiplier_data);
	// fill batch multiplier
	batch_sum_multiplier_.Reshape(num_, 1, 1, 1);
	Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
	caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);
}

template <typename Dtype>
void BNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* const_bottom_data = bottom[0]->cpu_data();
	const Dtype* const_top_data = top[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	const Dtype* scale_data = this->blobs_[0]->cpu_data();
	const Dtype* shift_data = this->blobs_[1]->cpu_data();

	// ---------- mean subtraction ---------- //
	// statistic across spatial
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1. / (height_ * width_)), const_bottom_data,
		spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
	// statistic across batch
	caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), spatial_statistic_.cpu_data(),
		batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());
	// save history mean
	if (this->phase_ == TRAIN) {
		caffe_cpu_axpby(batch_statistic_.count(), decay_, batch_statistic_.cpu_data(), Dtype(1) - decay_,
			this->blobs_[2]->mutable_cpu_data());
	}
	if (this->phase_ == TEST && moving_average_) {
		// use moving average mean
		caffe_copy(batch_statistic_.count(), this->blobs_[2]->cpu_data(), batch_statistic_.mutable_cpu_data());
	}
	// put mean blob into buffer_blob_
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
		batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
		spatial_statistic_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(-1),
		spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	// substract mean
	caffe_add(buffer_blob_.count(), const_bottom_data, buffer_blob_.cpu_data(), top_data);

	// ---------- variance normalization ---------- //
	// put the squares of X - mean into buffer_blob_
	caffe_powx(buffer_blob_.count(), const_top_data, Dtype(2), buffer_blob_.mutable_cpu_data());
	// statistic across spatial
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1. / (height_ * width_)), buffer_blob_.cpu_data(),
		spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
	// statistic across batch
	caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), spatial_statistic_.cpu_data(),
		batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());
	// save history variance
	if (this->phase_ == TRAIN) {
		caffe_cpu_axpby(batch_statistic_.count(), decay_, batch_statistic_.cpu_data(), Dtype(1) - decay_,
			this->blobs_[3]->mutable_cpu_data());
    // add eps
    caffe_add_scalar(batch_statistic_.count(), var_eps_, batch_statistic_.mutable_cpu_data());
	}
	if (this->phase_ == TEST && moving_average_) {
		// use moving average variance
		caffe_copy(batch_statistic_.count(), this->blobs_[3]->cpu_data(), batch_statistic_.mutable_cpu_data());
	}
	// std
	caffe_powx(batch_statistic_.count(), batch_statistic_.cpu_data(), Dtype(0.5),
		batch_statistic_.mutable_cpu_data());
	// put std blob into buffer_blob_
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
		batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
		spatial_statistic_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
		spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	// variance normalization
	caffe_div(buffer_blob_.count(), const_top_data, buffer_blob_.cpu_data(), top_data);

	// ---------- save x_norm and x_std ---------- //
	caffe_copy(buffer_blob_.count(), const_top_data, x_norm_.mutable_cpu_data());
	caffe_copy(batch_statistic_.count(), batch_statistic_.cpu_data(), x_std_.mutable_cpu_data());

	// ---------- scale ---------- //
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
		batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
		spatial_statistic_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
		spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	caffe_mul(buffer_blob_.count(), const_top_data, buffer_blob_.cpu_data(), top_data);

	// ---------- shift ---------- //
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
		batch_sum_multiplier_.cpu_data(), shift_data, Dtype(0),
		spatial_statistic_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
		spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	caffe_add(buffer_blob_.count(), const_top_data, buffer_blob_.cpu_data(), top_data);

}

template <typename Dtype>
void BNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	const Dtype* const_bottom_diff = bottom[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* const_top_diff = top[0]->cpu_diff();	

	Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();
	const Dtype* scale_data = this->blobs_[0]->cpu_data();

	// ---------- gradient w.r.t. scale ---------- //
	caffe_mul(buffer_blob_.count(), x_norm_.cpu_data(), const_top_diff, buffer_blob_.mutable_cpu_data());
	// statistic across spatial
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), buffer_blob_.cpu_data(),
		spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
	// statistic across batch
	caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
		batch_sum_multiplier_.cpu_data(), Dtype(0), scale_diff);

	// ---------- gradient w.r.t. shift ---------- //
	// statistic across spatial
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), const_top_diff,
		spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
	// statistic across batch
	caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
		batch_sum_multiplier_.cpu_data(), Dtype(0), shift_diff);

	// ---------- gradient w.r.t. to bottom blob ---------- //
	// put scale * top_diff to buffer_blob_
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
		batch_sum_multiplier_.cpu_data(), scale_data, Dtype(0),
		spatial_statistic_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
		spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	caffe_mul(buffer_blob_.count(), const_top_diff, buffer_blob_.cpu_data(), buffer_blob_.mutable_cpu_data());

  if (this->phase_ == TRAIN) {
    // use new top diff for computation
    caffe_mul(buffer_blob_.count(), x_norm_.cpu_data(), buffer_blob_.cpu_data(), bottom_diff);
    // statistic across spatial
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), const_bottom_diff,
                          spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
    // statistic across batch
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
                          batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
                          batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
                          spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
                          spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
                          bottom_diff);

    caffe_mul(buffer_blob_.count(), x_norm_.cpu_data(), const_bottom_diff, bottom_diff);

    // statistic across spatial
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), buffer_blob_.cpu_data(),
                          spatial_sum_multiplier_.cpu_data(), Dtype(0), spatial_statistic_.mutable_cpu_data());
    // statistic across batch
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.cpu_data(),
                          batch_sum_multiplier_.cpu_data(), Dtype(0), batch_statistic_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
                          batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(), Dtype(0),
                          spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
                          spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(1),
                          bottom_diff);

    caffe_cpu_axpby(buffer_blob_.count(), Dtype(1), buffer_blob_.cpu_data(), Dtype(-1. / (num_ * height_ * width_)),
                    bottom_diff);
  }
  if (this->phase_ == TEST && moving_average_) {
    // use moving average variance
    caffe_copy(buffer_blob_.count(), buffer_blob_.cpu_data(), bottom_diff);
  }
        
	// variance normalization
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
		batch_sum_multiplier_.cpu_data(), x_std_.cpu_data(), Dtype(0),
		spatial_statistic_.mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
		spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());

	caffe_div(buffer_blob_.count(), const_bottom_diff, buffer_blob_.cpu_data(), bottom_diff);

}

#ifdef CPU_ONLY
	STUB_GPU(BNLayer);
#endif

	INSTANTIATE_CLASS(BNLayer);
	REGISTER_LAYER_CLASS(BN);

}  // namespace caffe