#ifndef CAFFE_FORCE_LINK_H_
#define CAFFE_FORCE_LINK_H_
#include "caffe/caffe.hpp"
#include <string>
#include "caffe/common.hpp"

using namespace caffe;

#define FORCE_LINK(type) void force_link_function_##type(void) { extern LayerRegisterer<float> g_creator_f_##type; g_creator_f_##type.NeedDoSth(1); }

#endif //CAFFE_FORCE_LINK_H_
