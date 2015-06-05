#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#pragma once
namespace caffe {

	template <typename Dtype>
	void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* const_bottom_data = bottom[0]->gpu_data();
		const Dtype* const_top_data = top[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();		

		const Dtype* scale_data = this->blobs_[0]->gpu_data();
		const Dtype* shift_data = this->blobs_[1]->gpu_data();

		// put the squares of bottom into buffer_blob_
		caffe_gpu_powx(bottom[0]->count(), const_bottom_data, Dtype(2),
			buffer_blob_.mutable_gpu_data());

		// computes variance using var(X) = E(X^2) - (EX)^2
		// EX across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1. / (H_ * W_)), const_bottom_data,
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_mean_.mutable_gpu_data());
		// EX across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1. / N_), spatial_mean_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), batch_mean_.mutable_gpu_data());

		// E(X^2) across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1. / (H_ * W_)), buffer_blob_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_variance_.mutable_gpu_data());
		// E(X^2) across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1. / N_), spatial_variance_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), batch_variance_.mutable_gpu_data());

		caffe_gpu_powx(batch_mean_.count(), batch_mean_.gpu_data(), Dtype(2),
			buffer_blob_.mutable_gpu_data());  // (EX)^2
		caffe_gpu_sub(batch_mean_.count(), batch_variance_.gpu_data(), buffer_blob_.gpu_data(),
			batch_variance_.mutable_gpu_data());  // variance

		// do mean and variance normalization
		// subtract mean
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), batch_mean_.gpu_data(), Dtype(0),
			spatial_mean_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(-1),
			spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());

		caffe_gpu_add(buffer_blob_.count(), const_bottom_data, buffer_blob_.gpu_data(), top_data);

		// normalize variance
		caffe_gpu_add_scalar(batch_variance_.count(), var_eps_, batch_variance_.mutable_gpu_data());
		caffe_gpu_powx(batch_variance_.count(), batch_variance_.gpu_data(), Dtype(0.5),
			batch_variance_.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), batch_variance_.gpu_data(), Dtype(0),
			spatial_variance_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(1),
			spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());

		caffe_gpu_div(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data);

		// save x_norm
		caffe_copy(buffer_blob_.count(), const_top_data, x_norm_.mutable_gpu_data());

		// scale
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), scale_data, Dtype(0),
			spatial_variance_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(1),
			spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_mul(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data);

		// shift
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), shift_data, Dtype(0),
			spatial_mean_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(1),
			spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_add(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data);

	}

	template <typename Dtype>
	void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* const_top_diff = top[0]->gpu_diff();	

		Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
		Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
		const Dtype* scale_data = this->blobs_[0]->gpu_data();

		// gradient w.r.t. scale
		caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(), const_top_diff, buffer_blob_.mutable_gpu_data());
		// EX across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1), buffer_blob_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_variance_.mutable_gpu_data());
		// EX across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1), spatial_variance_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), scale_diff);

		// gradient w.r.t. shift
		// EX across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1), const_top_diff,
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_mean_.mutable_gpu_data());
		// EX across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1), spatial_mean_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), shift_diff);

		// put scale * top_diff to buffer_blob_
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), scale_data, Dtype(0),
			spatial_variance_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(1),
			spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_mul(buffer_blob_.count(), const_top_diff, buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());
		
		// use new top diff for computation
		caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(), buffer_blob_.gpu_data(), bottom_diff);
		// EX across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1), const_bottom_diff,
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_mean_.mutable_gpu_data());
		// EX across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1), spatial_mean_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), batch_mean_.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), batch_mean_.gpu_data(), Dtype(0),
			spatial_mean_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(1),
			spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			bottom_diff);

		caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(), const_bottom_diff, bottom_diff);

		// EX across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1), buffer_blob_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_mean_.mutable_gpu_data());
		// EX across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1), spatial_mean_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), batch_mean_.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), batch_mean_.gpu_data(), Dtype(0),
			spatial_mean_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(1),
			spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(1),
			bottom_diff);

		caffe_gpu_axpby(buffer_blob_.count(), Dtype(1), buffer_blob_.gpu_data(), Dtype(-1. / (N_ * H_ * W_)),
			bottom_diff);
        
		// variance normalization
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), batch_variance_.gpu_data(), Dtype(0),
			spatial_variance_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1, Dtype(1),
			spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());

		caffe_gpu_div(buffer_blob_.count(), const_bottom_diff, buffer_blob_.gpu_data(), bottom_diff);

	}

	INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);
	 
}  // namespace caffe