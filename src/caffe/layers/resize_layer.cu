#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/util_img.hpp"
namespace caffe {


  template <typename Dtype>
  void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    ResizeBlob_gpu(bottom[0], top[0]);
  }



  template <typename Dtype>
  __global__ void kernel_ResizeBackward(const int nthreads, const Dtype* top_diff, const int top_step,
                                        Dtype* bottom_diff, const int bottom_step,
                                        const Dtype* loc1, const  Dtype* weight1, const Dtype* loc2, const Dtype* weight2,
                                        const Dtype* loc3, const Dtype* weight3, const Dtype* loc4, const Dtype* weight4) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int bottom_diff_offset = bottom_step*index;
      int top_diff_offset = top_step*index;
      for (int idx = 0; idx < top_step; ++idx) {
        bottom_diff[bottom_diff_offset + int(loc1[idx])] += top_diff[top_diff_offset + idx] * weight1[idx];
        bottom_diff[bottom_diff_offset + int(loc2[idx])] += top_diff[top_diff_offset + idx] * weight2[idx];
        bottom_diff[bottom_diff_offset + int(loc3[idx])] += top_diff[top_diff_offset + idx] * weight3[idx];
        bottom_diff[bottom_diff_offset + int(loc4[idx])] += top_diff[top_diff_offset + idx] * weight4[idx];
      }
    }
  }
  template <typename Dtype>
  void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* top_diff = top[0]->mutable_gpu_diff();

    const Dtype* loc1 = this->locs_[0]->gpu_data();
    const Dtype* weight1 = this->locs_[0]->gpu_diff();
    const Dtype* loc2 = this->locs_[1]->gpu_data();
    const Dtype* weight2 = this->locs_[1]->gpu_diff();
    const Dtype* loc3 = this->locs_[2]->gpu_data();
    const Dtype* weight3 = this->locs_[2]->gpu_diff();
    const Dtype* loc4 = this->locs_[3]->gpu_data();
    const Dtype* weight4 = this->locs_[3]->gpu_diff();

    caffe::caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

    caffe::GetBiLinearResizeMatRules_gpu(bottom[0]->height(), bottom[0]->width(),
                                         top[0]->height(), top[0]->width(),
                                         this->locs_[0]->mutable_gpu_data(), this->locs_[0]->mutable_gpu_diff(),
                                         this->locs_[1]->mutable_gpu_data(), this->locs_[1]->mutable_gpu_diff(),
                                         this->locs_[2]->mutable_gpu_data(), this->locs_[2]->mutable_gpu_diff(),
                                         this->locs_[3]->mutable_gpu_data(), this->locs_[3]->mutable_gpu_diff());

    const int top_step = top[0]->offset(0, 1);
    const int bottom_step = bottom[0]->offset(0, 1);

    int loop_n = this->out_num_ * this->out_channels_;

    kernel_ResizeBackward<Dtype> << <CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >> >(
      loop_n, top_diff, top_step,
      bottom_diff, bottom_step,
      loc1, weight1, loc2, weight2,
      loc3, weight3, loc4, weight4);
    CUDA_POST_KERNEL_CHECK;
  }



  INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);

}  // namespace caffe