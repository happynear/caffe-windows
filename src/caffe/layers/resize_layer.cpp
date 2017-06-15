#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/util_img.hpp"
namespace caffe {

  template <typename Dtype>
  void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    // Configure the kernel size, padding, stride, and inputs.
    ResizeParameter resize_param = this->layer_param_.resize_param();


    bool is_pyramid_test = resize_param.is_pyramid_test();
    if (is_pyramid_test == false) {
      CHECK(resize_param.has_height()) << "output height is required ";
      CHECK(resize_param.has_width()) << "output width is required ";
      this->out_height_ = resize_param.height();
      this->out_width_ = resize_param.width();
    }
    else {
      CHECK(resize_param.has_out_height_scale()) << "output height scale is required ";
      CHECK(resize_param.has_out_width_scale()) << "output width scale is required ";
      int in_height = bottom[0]->height();
      int in_width = bottom[0]->width();
      this->out_height_ = int(resize_param.out_height_scale() * in_height);
      this->out_width_ = int(resize_param.out_width_scale() * in_width);
    }




    for (int i = 0; i<4; i++) {
      this->locs_.push_back(new Blob<Dtype>);
    }
  }

  template <typename Dtype>
  void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {

    ResizeParameter resize_param = this->layer_param_.resize_param();

    bool is_pyramid_test = resize_param.is_pyramid_test();
    if (is_pyramid_test == false) {

      this->out_height_ = resize_param.height();
      this->out_width_ = resize_param.width();
    }
    else {
      int in_height = bottom[0]->height();
      int in_width = bottom[0]->width();
      this->out_height_ = int(resize_param.out_height_scale() * in_height);
      this->out_width_ = int(resize_param.out_width_scale() * in_width);
    }


    this->out_num_ = bottom[0]->num();
    this->out_channels_ = bottom[0]->channels();
    top[0]->Reshape(out_num_, out_channels_, out_height_, out_width_);

    for (int i = 0; i<4; ++i) {
      this->locs_[i]->Reshape(1, 1, out_height_, out_width_);
    }
  }

  template <typename Dtype>
  void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    //	ResizeBlob_cpu(bottom[0],top[0],this->locs_[0],this->locs_[1],this->locs_[2],this->locs_[3]);
    ResizeBlob_cpu(bottom[0], top[0]);

  }

  template <typename Dtype>
  void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* top_diff = top[0]->mutable_cpu_diff();

    const Dtype* loc1 = this->locs_[0]->cpu_data();
    const Dtype* weight1 = this->locs_[0]->cpu_diff();
    const Dtype* loc2 = this->locs_[1]->cpu_data();
    const Dtype* weight2 = this->locs_[1]->cpu_diff();
    const Dtype* loc3 = this->locs_[2]->cpu_data();
    const Dtype* weight3 = this->locs_[2]->cpu_diff();
    const Dtype* loc4 = this->locs_[3]->cpu_data();
    const Dtype* weight4 = this->locs_[3]->cpu_diff();


    caffe::caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    caffe::GetBiLinearResizeMatRules_cpu(bottom[0]->height(), bottom[0]->width(),
                                         top[0]->height(), top[0]->width(),
                                         this->locs_[0]->mutable_cpu_data(), this->locs_[0]->mutable_cpu_diff(),
                                         this->locs_[1]->mutable_cpu_data(), this->locs_[1]->mutable_cpu_diff(),
                                         this->locs_[2]->mutable_cpu_data(), this->locs_[2]->mutable_cpu_diff(),
                                         this->locs_[3]->mutable_cpu_data(), this->locs_[3]->mutable_cpu_diff());

    for (int n = 0; n< this->out_num_; ++n) {
      for (int c = 0; c < this->out_channels_; ++c) {
        int bottom_diff_offset = bottom[0]->offset(n, c);
        int top_diff_offset = top[0]->offset(n, c);

        for (int idx = 0; idx < this->out_height_* this->out_width_; ++idx) {
          bottom_diff[bottom_diff_offset + static_cast<int>(loc1[idx])] += top_diff[top_diff_offset + idx] * weight1[idx];
          bottom_diff[bottom_diff_offset + static_cast<int>(loc2[idx])] += top_diff[top_diff_offset + idx] * weight2[idx];
          bottom_diff[bottom_diff_offset + static_cast<int>(loc3[idx])] += top_diff[top_diff_offset + idx] * weight3[idx];
          bottom_diff[bottom_diff_offset + static_cast<int>(loc4[idx])] += top_diff[top_diff_offset + idx] * weight4[idx];
        }
      }
    }
  }

#ifdef CPU_ONLY
  STUB_GPU(ResizeLayer);
#endif

  INSTANTIATE_CLASS(ResizeLayer);
  REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe