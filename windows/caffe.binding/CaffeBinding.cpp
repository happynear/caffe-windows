#include "CaffeBinding.h"
#include <caffe\caffe.hpp>
#include <caffe\layers\memory_data_layer.hpp>
#include <boost/thread.hpp>

using namespace caffe;
using namespace std;

std::vector<Net<float>*> nets_;//no shared_ptr here.
std::vector<std::shared_ptr<boost::thread_specific_ptr<caffe::Net<float>>>> predictors_;
std::vector<string> prototxts;

CaffeBinding::CaffeBinding() {
  FLAGS_minloglevel = google::FATAL;
}

int CaffeBinding::AddNet(string model_definition, string weights, int gpu_id) {
  SetDevice(gpu_id);
  auto new_net = new Net<float>(model_definition, Phase::TEST);//boost::make_shared<Net<float> >(model_definition, Phase::TEST);
  new_net->CopyTrainedLayersFrom(weights);
  nets_.push_back(new_net);
  predictors_.push_back(make_shared<boost::thread_specific_ptr<caffe::Net<float>>>());
  prototxts.push_back(model_definition);
  return nets_.size() - 1;
}

std::unordered_map<std::string, DataBlob> CaffeBinding::Forward(int net_id) {
  if (!(*predictors_[net_id]).get()) {
    auto predictor =
      std::make_unique<caffe::Net<float>>(prototxts[net_id], Phase::TEST);
    predictor->ShareTrainedLayersWith(nets_[net_id]);
    (*predictors_[net_id]).reset(predictor.release());
  }
  auto* predictor = (*predictors_[net_id]).get();
  const std::vector<Blob<float>*>& nets_output = predictor->ForwardPrefilled();
  std::unordered_map<std::string, DataBlob> result;
  for (int n = 0; n < nets_output.size(); n++) {
    DataBlob blob = { nets_output[n]->cpu_data(), nets_output[n]->shape(), predictor->blob_names()[predictor->output_blob_indices()[n]] };
    result[blob.name] = blob;
  }
  return result;
}

std::unordered_map<std::string, DataBlob> CaffeBinding::Forward(std::vector<cv::Mat>&& input_image, int net_id) {
  SetMemoryDataLayer("data", move(input_image), net_id);
  return Forward(net_id);
}

void CaffeBinding::SetMemoryDataLayer(std::string layer_name, std::vector<cv::Mat>&& input_image, int net_id) {
  if (!(*predictors_[net_id]).get()) {
    auto predictor =
      std::make_unique<caffe::Net<float>>(prototxts[net_id], Phase::TEST);
    predictor->ShareTrainedLayersWith(nets_[net_id]);
    (*predictors_[net_id]).reset(predictor.release());
  }
  auto* predictor = (*predictors_[net_id]).get();
  std::vector<int> labels;
  labels.push_back(1);
  auto data_layer_ptr = static_pointer_cast<MemoryDataLayer<float>, Layer<float>>(predictor->layer_by_name(layer_name));
  data_layer_ptr->AddMatVector(input_image, labels);
}

void CaffeBinding::SetBlobData(std::string blob_name, std::vector<int> blob_shape, float* data, int net_id) {
  if (!(*predictors_[net_id]).get()) {
    auto predictor =
      std::make_unique<caffe::Net<float>>(prototxts[net_id], Phase::TEST);
    predictor->ShareTrainedLayersWith(nets_[net_id]);
    (*predictors_[net_id]).reset(predictor.release());
  }
  auto* predictor = (*predictors_[net_id]).get();
  predictor->blob_by_name(blob_name)->Reshape(blob_shape);
  predictor->blob_by_name(blob_name)->set_cpu_data(data);
}

DataBlob CaffeBinding::GetBlobData(std::string blob_name, int net_id) {
  if (!(*predictors_[net_id]).get()) {
    auto predictor =
      std::make_unique<caffe::Net<float>>(prototxts[net_id], Phase::TEST);
    predictor->ShareTrainedLayersWith(nets_[net_id]);
    (*predictors_[net_id]).reset(predictor.release());
  }
  auto* predictor = (*predictors_[net_id]).get();
  auto blob = predictor->blob_by_name(blob_name);
  if (blob == nullptr) return { NULL, {0}, blob_name };
  else return { predictor->blob_by_name(blob_name)->cpu_data(), blob->shape(), blob_name };
}

void CaffeBinding::SetDevice(int gpu_id) {
  if (gpu_id < 0) {
    Caffe::set_mode(Caffe::CPU);
  }
  else {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_id);
  }
}

CaffeBinding::~CaffeBinding() {
  for (auto net : nets_) {
    try {
      delete net;
    }
    catch (...) {

    }
  }
}
