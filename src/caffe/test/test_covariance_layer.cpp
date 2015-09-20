#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class CovarianceLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		CovarianceLayerTest()
			: blob_data_(new Blob<Dtype>(2, 5, 7, 7)),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			filler_param.set_value(1);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_data_);

			blob_bottom_vec_.push_back(blob_data_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~CovarianceLayerTest() { delete blob_data_; delete blob_top_; }
		Blob<Dtype>* const blob_data_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(CovarianceLayerTest, TestDtypesAndDevices);

	TYPED_TEST(CovarianceLayerTest, TestForward) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		CovarianceLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		// Test sum
		int spatial_dim = this->blob_data_->height() * this->blob_data_->width();
		int height = this->blob_data_->height();
		int	width = this->blob_data_->width();
		int channel = this->blob_data_->channels();

		for (int n = 0; n < this->blob_top_->num(); ++n) {
			for (int i = 0; i < this->blob_top_->shape(1); ++i) {
				for (int j = 0; j < this->blob_top_->shape(2); ++j) {
					Dtype value = 0;
					for (int ii = 0; ii < height; ii++) {
						for (int jj = 0; jj < width; jj++)
						{
							value += this->blob_data_->data_at(n, i, ii, jj) * this->blob_data_->data_at(n, j, ii, jj);
						}
					}
					value /= spatial_dim * channel;
					LOG(INFO) << "(" << n << "," << i << "," << j << ")" << "(" << value << "," << this->blob_top_->cpu_data()[n*channel*channel + i*channel + j] << ")";
					CHECK(abs(value - this->blob_top_->cpu_data()[n*channel*channel + i*channel + j]) < 1e-4) << "(" << n << "," << i << "," << j << ")" << "(" << value << "," << this->blob_top_->cpu_data()[n*channel*channel + i*channel + j] << ")";

				}
			}
		}
	}

	TYPED_TEST(CovarianceLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		CovarianceLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_,-1);
	}

}  // namespace caffe
