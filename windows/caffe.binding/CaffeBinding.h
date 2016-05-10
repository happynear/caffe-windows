#pragma once

#ifdef CAFFEBINDING_EXPORTS
#define CAFFE_DLL __declspec(dllexport)
#else
#define CAFFE_DLL __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>

namespace caffe {
  struct DataBlob {
    const float* data;
    std::vector<int> size;
  };
  class CAFFE_DLL CaffeBinding {
  public:
    CaffeBinding();
    int AddNet(std::string model_definition, std::string weights, int gpu_id = 0);
    std::vector<DataBlob> Forward(std::vector<cv::Mat> input_image, int net_id);
    ~CaffeBinding();
  };
}