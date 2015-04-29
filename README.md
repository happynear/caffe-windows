Forked from https://www.github.com/BVLC/caffe dev branch in 2015/3/31

Added [Batch Normalization](http://arxiv.org/abs/1502.03167), [Parametric ReLU](http://arxiv.org/abs/1502.01852), Locally Connected Layer, Normalize Layer.

Setup step:
======
Download third-party libraries from http://pan.baidu.com/s/1qAVPs , and put the 3rdparty folder under the root of caffe-windows.

Double click build/MSVC/MainBuilder.sln to open the solution in Visual Studio 2012 (only VS2012 supported).

Change the compile mode to Release and X64.

Change the CUDA include and library path to your own ones.

Compile.

tips: If you have MKL library, please add the preprocess macro "USE_MKL" defined in the setting of the project.


