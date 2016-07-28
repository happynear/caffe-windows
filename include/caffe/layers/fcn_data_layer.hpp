#if USE_OPENCV
#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

  /**
  * @brief Provides data to the Net from image files.
  *
  * TODO(dox): thorough documentation for Forward and proto params.
  */
  template <typename Dtype>
  class FCNDataLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
    explicit FCNDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
    virtual ~FCNDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "FCNData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

  protected:
    shared_ptr<Caffe::RNG> prefetch_rng_;
    virtual void ShuffleImages();
    virtual void load_batch(Batch<Dtype>* batch);

    int image_id;

    vector<string> image_folder_list;
    vector<pair<string, cv::Rect2d>> image_rect_list;
    cv::Size template_size = cv::Size(42, 48);
    double expand_left = 0.25, expand_right = 0.25, expand_top = 0.25, expand_bottom = 0.25;//can only deal with "top=left+right" up to now
    cv::Size2d roi_multiply = cv::Size(5, 5);
    double scale_step = 1.1;
    int scale_step_num = 5;
    cv::Size gaussian_size = cv::Size(7, 7);
    double gaussian_std_h = 1.5, gaussian_std_w = 1.5;
    cv::Mat target_temp;
    cv::Size2d target_wheel_size;
    cv::Size target_roi_size;
    cv::Size target_featuremap_size;
    cv::Size target_heatmap_size;
    bool use_hog;
    int hog_cell_size;
  };

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
#endif  // USE_OPENCV