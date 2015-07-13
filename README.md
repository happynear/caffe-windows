Forked from https://www.github.com/BVLC/caffe master branch in 2015/6/5

Added [Batch Normalization](http://arxiv.org/abs/1502.03167), [Parametric ReLU](http://arxiv.org/abs/1502.01852), Locally Connected Layer, Normalize Layer, [Randomized ReLU](http://arxiv.org/abs/1505.00853).

Update
======
2015/07/07  Visual Studio 2013 with CUDA 7.0 is now supported. A beta version 3rdparty library can be downloaded from http://pan.baidu.com/s/1sj3IvzZ. All the libraries have been updated to the latest version. Please help me try and report bugs.

WARNING: Due to the low compile speed of VS2012 with CUDA 6.5, VS2012 3rdparty library will not continue to be updated after September, 2015. If you are configuring a new platform, we strongly recommend you to use Visual Studio 2013 and CUDA 7.0.

Setup step:
======
1. Download third-party libraries from http://pan.baidu.com/s/1sjE5ER7 (for VS2012), and put the 3rdparty folder under the root of caffe-windows. **Please don't forgert to add the `./3rdparty/bin` folder to your environment variable `PATH`.**

2. Run `./src/caffe/proto/extract_proto.bat` to create `caffe.pb.h`, `caffe.pb.cc` and `caffe_pb2.py`.

3. Double click ./build/MSVC/MainBuilder.sln to open the solution in Visual Studio 2012. If you are using VS2013, please download 3rdparty libraries and solution files from http://pan.baidu.com/s/1sj3IvzZ.

4. Change the compile mode to Release and X64.

5. Change the CUDA include and library path to your own ones.

6. Compile.

TIPS: If you have MKL library, please add the preprocess macro "USE_MKL" defined in the setting of the project.

中文安装说明：http://blog.csdn.net/happynear/article/details/45372231

Matlab Wrapper
======
Just change the Matlab include and library path defined in the settings and compile.
**Don't forget to add `./matlab` to your Matlab path.**

Python Wrapper
======
Similar with Matlab, just change the python include and library path defined in the settings and compile.

MNIST example
======
Please download the mnist leveldb database from http://pan.baidu.com/s/1mgl9ndu and extract it to `./examples/mnist`. Then double click `./run_mnist.bat` to run the MNIST demo.

Acknowlegement
======
We greatly thank [Yangqing Jia](https://github.com/Yangqing) and [BVLC group](https://www.github.com/BVLC/caffe) for developing Caffe,

[@niuzhiheng](https://github.com/niuzhiheng) for his contribution on the first generation of caffe-windows,

[@ChenglongChen](https://github.com/ChenglongChen/batch_normalization) for his implementation of Batch Normalization,

[@jackculpepper](https://github.com/jackculpepper/caffe) for his implementation of locally-connected layer,

and all people who have contributed to the caffe user group.
