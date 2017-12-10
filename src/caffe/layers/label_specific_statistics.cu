#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_statistics_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void CreateMask(const int num, const int dim, const Dtype* label, Dtype* positive_mask, Dtype* negative_mask) {
    CUDA_KERNEL_LOOP(index, num) {
      int gt = static_cast<int>(label[index]);
      positive_mask[index*dim + gt] = Dtype(1);
      negative_mask[index*dim + gt] = Dtype(0);
    }
  }

  template <typename Dtype>
  void LabelSpecificStatisticsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Dtype *positive_mask_data = positive_mask.mutable_gpu_data();
    Dtype *negative_mask_data = negative_mask.mutable_gpu_data();
    caffe_gpu_set(count, Dtype(0), positive_mask_data);
    caffe_gpu_set(count, Dtype(1), negative_mask_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    CreateMask<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, positive_mask.mutable_gpu_data(), negative_mask.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    Dtype positive_mean;
    Dtype positive_std;
    Dtype negative_mean;
    Dtype negative_std;

    caffe_gpu_powx(count, bottom_data, Dtype(2), bottom_square.mutable_gpu_data());
    caffe_gpu_dot(count, bottom_data, positive_mask.gpu_data(), &positive_mean);
    caffe_gpu_dot(count, bottom_square.gpu_data(), positive_mask.gpu_data(), &positive_std);
    caffe_gpu_dot(count, bottom_data, negative_mask.gpu_data(), &negative_mean);
    caffe_gpu_dot(count, bottom_square.gpu_data(), negative_mask.gpu_data(), &negative_std);

    positive_mean /= M_PI / Dtype(180.0);
    negative_mean /= M_PI / Dtype(180.0);
    positive_std /= M_PI / Dtype(180.0) * M_PI / Dtype(180.0);
    negative_std /= M_PI / Dtype(180.0) * M_PI / Dtype(180.0);
    positive_mean /= num;
    positive_std = sqrt(positive_std / num - positive_mean * positive_mean);
    negative_mean /= num * (dim - 1);
    negative_std = sqrt(negative_std / num / (dim - 1) - negative_mean * negative_mean);

    top[0]->mutable_cpu_data()[0] = positive_mean;
    top[0]->mutable_cpu_data()[1] = positive_std;
    top[0]->mutable_cpu_data()[2] = negative_mean;
    top[0]->mutable_cpu_data()[3] = negative_std;
    if (top.size() == 2) {
      top[1]->mutable_cpu_data()[0] = (negative_mean - positive_mean)/2;
    }
  }

  template <typename Dtype>
  void LabelSpecificStatisticsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0] && top.size() == 2) {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

      int num = bottom[0]->num();
      int count = bottom[0]->count();
      int dim = count / num;
      Dtype top_diff = top[1]->cpu_diff()[0];

      caffe_gpu_scal(count, -top_diff / Dtype(num) / Dtype(M_PI) * Dtype(180.0), positive_mask.mutable_gpu_data());
      caffe_gpu_scal(count, top_diff / Dtype(num) / Dtype(dim - 1) / Dtype(M_PI) * Dtype(180.0), negative_mask.mutable_gpu_data());
      caffe_gpu_add(count, positive_mask.gpu_data(), negative_mask.gpu_data(), bottom_diff);
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificStatisticsLayer);


}  // namespace caffe
