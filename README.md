Forked from https://www.github.com/BVLC/caffe master branch in 2015/9/1

Added [Batch Normalization](http://arxiv.org/abs/1502.03167), [Parametric ReLU](http://arxiv.org/abs/1502.01852), Locally Connected Layer, Normalize Layer, [Randomized ReLU](http://arxiv.org/abs/1505.00853), Triplet Loss, SmoothL1 Layer, ROI Layer. 

Setup step:
======

1. Download third-party libraries from [BaiduYun Disk](http://pan.baidu.com/s/1pJmW7tL) or [OneDrive](http://1drv.ms/1JIc9gd
) and extract the files to `caffe-windows_root/3rdparty/`. **Please don't forget to add the `./3rdparty/bin` folder to your environment variable `PATH`.**

2. Run `./src/caffe/proto/extract_proto.bat` to create `caffe.pb.h`, `caffe.pb.cc` and `caffe_pb2.py`.

3. Double click ./build/MainBuilder.sln to open the solution. 

4. Change the compile mode to Release and X64. For Debug mode, you may need these 3rparty libraries http://pan.baidu.com/s/1qW88MTY .

5. Modify the cuda device compute capability defined in the settings (`caffelib properties` -> `CUDA C/C++` -> `Device` -> `Code Generation`) to your GPU's compute capability (such as compute_30,sm_30; etc). You can look up for your GPU's compute capability in https://en.wikipedia.org/wiki/CUDA . Some general GPUs' compute capabilities are listed below.

 - If your GPU's compute capability is below or equal to 2.1, please remove the `USE_CUDNN` macro in the proprocessor definition of all projects.

 - If you do not have a Nvidia GPU, please also add `CPU_ONLY` macro besides removing `USE_CUDNN`.

6. Compile.

| GPU                                         | Compute Capability    |
| ------------------------------------------- |:---------------------:|
| GTX660, 680, 760, 770                       | compute_30,sm_30      |
| GTX780, Titan Z, Titan Black, K20, K40      | compute_35,sm_35      |
| GTX960, 980, Titan X                        | compute_52,sm_52      |



TIPS: If you have MKL library, please add the preprocess macro "USE_MKL" defined in the setting of the project.

If you want build other tools, just copy and rename `./build/MSVC` folder to another one, and add the new project to the VS solution. Remove `caffe.cpp` and add your target cpp file. Compile it, then you will get a corresponding exe file in `./bin`.

中文安装说明：http://blog.csdn.net/happynear/article/details/45372231

Matlab Wrapper
======
Just replace the Matlab include and library path defined in the settings and compile.
**Don't forget to add `./matlab` to your Matlab path.**

Python Wrapper
======
Similar with Matlab, replace the python include and library path and compile.

Most of the libraries listed in `./python/requirements.txt` can be installed by `pip install`. However, some of them cannot be installed so easily.

For protobuf, you may download the codes from https://github.com/google/protobuf. Copy `caffe-windows-root/src/caffe/proto/protoc.exe` to `protobuf-root/src`. Then run `python setup.py install` in `protobuf-root/python`.

For leveldb, I have created a repository https://github.com/happynear/py-leveldb-windows . Please follow the instructions in `README.md` to install it.

MNIST example
======
Please download the mnist leveldb database from http://pan.baidu.com/s/1mgl9ndu and extract it to `./examples/mnist`. Then double click `./run_mnist.bat` to run the MNIST demo.

Update log
======
2015/09/14 Multi-GPU is supported now. 

WARNING: When you are using multiple gpus to train a model, please do not directly close the command window. Instead, please use `Ctrl+C` to avoid the gpu driver from crash.

You can also press `Ctrl+Break` to save a model snapshot whenever you want during training.

2015/08/18 The lmdb problem has been fixed. Download the new lmdb lib file from http://pan.baidu.com/s/1dDHbbgP (only a small patch), overwrite the original one in `3rdparty/lib`, and re-link the convert_imageset, convert_mnist etc projects, you will be able to create lmdb on Windows.

2015/08/08 The cuDNN v3 is not very stable at present. The master branch has been rolled back to cuDNN v2. The cuDNN v3 will come back as soon as it has been tested enough. Nonetheless, you can still find cuDNN v3 version in branch `cuDNNV3`. 

Fortunately, cuDNN is backward-compatible, so the 3rdparty libraries (http://pan.baidu.com/s/1i390tZB) need not to be changed.

2015/08/06 cuDNN v3 is released! The new 3rdparty library with cuDNN v3 can be downloaded from http://pan.baidu.com/s/1i390tZB. In this update, I use an ungainly method to build the caffe core functions in one project as a static lib. I am still looking for better solutions. Issues and Pull Requests are welcomed.

**Please help me test the speed of cuDNN v3 on non-MaxWell architecture GPUs. On my GTX780, some kinds of net, such as VGG, are quite slower than cuDNN v2.**

**WARNING: Visual Studio 2012 and CUDA6.5 are no longer supported. Please update your CUDA to version 7.0. If you are still using VS2012, please try this solution file and 3rdparty library http://pan.baidu.com/s/1i3hGef7. I haven't check it. So if you find bugs, please report to me.**

Acknowlegement
======
We greatly thank [Yangqing Jia](https://github.com/Yangqing) and [BVLC group](https://www.github.com/BVLC/caffe) for developing Caffe,

[@niuzhiheng](https://github.com/niuzhiheng) for his contribution on the first generation of caffe-windows,

[@ChenglongChen](https://github.com/ChenglongChen/batch_normalization) for his implementation of Batch Normalization,

[@jackculpepper](https://github.com/jackculpepper/caffe) for his implementation of locally-connected layer,

and all people who have contributed to the caffe user group.
