#pragma once

#ifdef CAFFEBINDING_EXPORTS
#define CAFFE_DLL __declspec(dllexport)
#else
#define CAFFE_DLL __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace caffe {
  struct DataBlob {
    const float* data;
    std::vector<int> size;
    std::string name;
  };
  class CAFFE_DLL CaffeBinding {
  public:
    CaffeBinding();
    int AddNet(std::string prototxt_path, std::string weights_path, int gpu_id = 0);
    std::unordered_map<std::string, DataBlob> Forward(int net_id);
    std::unordered_map<std::string, DataBlob> Forward(std::vector<cv::Mat>&& input_image, int net_id);
    std::unordered_map<std::string, DataBlob> Forward(std::vector<cv::Mat>& input_image, int net_id) {
      return Forward(std::move(input_image), net_id);
    }
    void SetMemoryDataLayer(std::string layer_name, std::vector<cv::Mat>&& input_image, int net_id);
    void SetMemoryDataLayer(std::string layer_name, std::vector<cv::Mat>& input_image, int net_id) {
      SetMemoryDataLayer(layer_name, std::move(input_image), net_id);
    }
    void SetBlobData(std::string blob_name, std::vector<int> blob_shape, float* data, int net_id);
    DataBlob GetBlobData(std::string blob_name, int net_id);
    void SetDevice(int gpu_id);
    ~CaffeBinding();
  };
}