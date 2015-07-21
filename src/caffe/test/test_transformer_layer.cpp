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
	class TransformerLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		TransformerLayerTest()
			: blob_data_(new Blob<Dtype>(2, 3, 5, 5)),
			blob_theta_(new Blob<Dtype>(vector<int>{ 2, 6 })),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			filler_param.set_value(1);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_data_);
			filler_param.set_value(0);
			ConstantFiller<Dtype> constant_filler(filler_param);
			constant_filler.Fill(this->blob_theta_);
			/*this->blob_theta_->mutable_cpu_data()[0] = 1;
			this->blob_theta_->mutable_cpu_data()[4] = 1;*/
			this->blob_theta_->mutable_cpu_data()[0] = sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[1] = sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[2] = -0.1;
			this->blob_theta_->mutable_cpu_data()[3] = -sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[4] = sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[5] = 0.1;
			this->blob_theta_->mutable_cpu_data()[0+6] = sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[1+6] = sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[2+6] = 0.1;
			this->blob_theta_->mutable_cpu_data()[3+6] = -sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[4+6] = sqrt(2) / 2;
			this->blob_theta_->mutable_cpu_data()[5+6] = 0.1;
			/*this->blob_theta_->mutable_cpu_data()[0+6] = 1;
			this->blob_theta_->mutable_cpu_data()[4+6] = 1;*/
			blob_bottom_vec_.push_back(blob_data_);
			blob_bottom_vec_.push_back(blob_theta_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~TransformerLayerTest() { delete blob_data_; delete blob_theta_; delete blob_top_; }
		Blob<Dtype>* const blob_data_;
		Blob<Dtype>* const blob_theta_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(TransformerLayerTest, TestDtypesAndDevices);

	TYPED_TEST(TransformerLayerTest, TestForward) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		TransformerLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		// Test sum

		
		for (int n = 0; n < this->blob_data_->num(); ++n) {
			for (int c = 0; c < this->blob_data_->channels(); ++c) {
				for (int i = 0; i < this->blob_data_->height(); ++i) {
					for (int j = 0; j < this->blob_data_->width(); ++j) {
						Dtype x = (i / (Dtype)this->blob_data_->height() * 2 - 1) * this->blob_theta_->cpu_data()[0 + n * 6] + (j / (Dtype)this->blob_data_->width() * 2 - 1) * this->blob_theta_->cpu_data()[1 + n * 6] + this->blob_theta_->cpu_data()[2 + n * 6];
						x = x  * this->blob_data_->height() / 2 + (Dtype)this->blob_data_->height() / 2;
						Dtype y = (i / (Dtype)this->blob_data_->height() * 2 - 1) * this->blob_theta_->cpu_data()[3 + n * 6] + (j / (Dtype)this->blob_data_->width() * 2 - 1) * this->blob_theta_->cpu_data()[4 + n * 6] + this->blob_theta_->cpu_data()[5 + n * 6];
						y = y  * this->blob_data_->width() / 2 + (Dtype)this->blob_data_->width() / 2;
						if (x >= 0 && x <= this->blob_data_->height() - 1 && y >= 0 && y <= blob_data_->width() - 1)
						{
							Dtype value = 0, y1, y2;
							y1 = this->blob_data_->data_at(n, c, floor(x), floor(y)) * (1 - (x - floor(x)))+ this->blob_data_->data_at(n, c, ceil(x), floor(y)) * (1 - (ceil(x) - x));
							if (floor(x) == ceil(x)) y1 /= 2;

							if (floor(y) == ceil(y)) value = y1;
							else
							{
								y2 = this->blob_data_->data_at(n, c, floor(x), ceil(y)) * (1 - (x - floor(x))) + this->blob_data_->data_at(n, c, ceil(x), ceil(y)) * (1 - (ceil(x) - x));
								if (floor(x) == ceil(x)) y2 /= 2;
								value = y1 * (1 - (y - floor(y))) + y2 * (1 - (ceil(y) - y));
							}
							//LOG(INFO) << "(" << n << " " << c << " " << i << " " << j << ")("<<x<<"," <<y<< ")(" << value << "," << this->blob_top_->data_at(n, c, i, j) << ")";
							CHECK(abs(value - this->blob_top_->data_at(n, c, i, j)) < 1e-4) << "(" << n << " " << c << " " << i << " " << j << ")" << "(" << value << "," << this->blob_top_->data_at(n, c, i, j)<<")";
						}
						
					}
				}
			}
		}
	}

	TYPED_TEST(TransformerLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		TransformerLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_,-1);
	}

}  // namespace caffe
