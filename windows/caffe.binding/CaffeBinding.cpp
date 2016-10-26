#include "CaffeBinding.h"
#include <caffe\caffe.hpp>
#include <caffe\layers\memory_data_layer.hpp>

using namespace caffe;
using namespace std;

std::vector<Net<float>*> net_;//no shared_ptr here.

CaffeBinding::CaffeBinding() {
  FLAGS_minloglevel = google::FATAL;
}

int CaffeBinding::AddNet(string model_definition, string weights, int gpu_id) {
  if (gpu_id < 0) {
    Caffe::set_mode(Caffe::CPU);
  }
  else {
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  }
  auto new_net = new Net<float>(model_definition, Phase::TEST);//boost::make_shared<Net<float> >(model_definition, Phase::TEST);
  new_net->CopyTrainedLayersFrom(weights);
  net_.push_back(new_net);
  return net_.size() - 1;
}

std::unordered_map<std::string, DataBlob> caffe::CaffeBinding::Forward(int net_id) {
  const std::vector<Blob<float>*>& net_output = net_[net_id]->ForwardPrefilled();
  std::unordered_map<std::string, DataBlob> result;
  for (int n = 0; n < net_output.size(); n++) {
    DataBlob blob = { net_output[n]->cpu_data(), net_output[n]->shape(), net_[net_id]->blob_names()[net_[net_id]->output_blob_indices()[n]] };
    result[blob.name] = blob;
  }
  return result;
}

std::unordered_map<std::string, DataBlob> CaffeBinding::Forward(std::vector<cv::Mat> input_image, int net_id) {
  //std::vector<cv::Mat> datum_vector;
  //datum_vector.push_back(input_image);
  std::vector<int> labels;
  labels.push_back(1);
  auto data_layer_ptr = static_pointer_cast<MemoryDataLayer<float>, Layer<float>>(net_[net_id]->layers()[0]);
  //MemoryDataLayer<float>* data_layer_ptr = (MemoryDataLayer<float>*)&(*net_[net_id]->layers()[0]);
  data_layer_ptr->AddMatVector(input_image, labels);
  return Forward(net_id);
}

void caffe::CaffeBinding::SetMemoryDataLayer(std::string layer_name, std::vector<cv::Mat> input_image, int net_id) {
  std::vector<int> labels;
  labels.push_back(1);
  auto data_layer_ptr = static_pointer_cast<MemoryDataLayer<float>, Layer<float>>(net_[net_id]->layer_by_name(layer_name));
  data_layer_ptr->AddMatVector(input_image, labels);
}

CaffeBinding::~CaffeBinding() {
  for (auto& net : net_) {
    try {
      delete net;
    }
    catch (...) {

    }
  }
}
