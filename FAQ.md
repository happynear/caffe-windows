FAQ for caffe-windows
================

 - I don't have a GPU, how can I build caffe without CUDA support?
 
   If you do not have a Nvidia GPU, please add `CPU_ONLY` macro to the preprocessor definition and delete `USE_CUDNN` macro.
   Then remove all .cu files in the caffelib project to avoid compiling the GPU-related codes.
   
 - How to use the static library `caffelib.lib` in my own project?
 
   Directly link the `caffelib.lib` may lead the compiler ignore the layer and solver register macros. When loading a layer or solver,
   similar error will be reported:
   ```
   check failed: registry.count(type)==1(0 vs 1)unknown layer type:convolution
   ```
   There are two ways to solve this problem. One is to add the caffelib project to your own solution, and open the property window of
   your project set
   `Common Property` - `Reference` - `caffelib` - `Project Reference Property` - `Library Dependency Link`: True .
   
   Another method is to add `layer_factory.cpp` and `force_link.cpp` to your own project, to let the compiler know the existence of
   the layers and solvers.
   
   If you came across similar error when using `caffe.exe`, I guess you may have modified the `.vcxproj` files by yourself. VS lost some configurations during your modification. Never mind, you can still follow the above instructions to fix it.
