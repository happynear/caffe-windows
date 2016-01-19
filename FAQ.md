FAQ for caffe-windows
================

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
   the layers and solvers. However, when using this method, the layers will be registered twice and you will get an error in `include/caffe/layer_factory.hpp` line `68`. To fix this error, you can just remove line `68-69` or use `if(registry.count(type) > 0) continue;` to replace the `CHECK` statement.
   
   If you came across similar error when using `caffe.exe`, I guess you may have modified the `.vcxproj` files manually by yourself. VS lost some configurations during your modification. Never mind, you can still follow the above instructions to fix it.
   
 - How to compile the codes in Debug mode?
   
  There is a [3rdparty library archive file](http://pan.baidu.com/s/1qW88MTY) provided by a friend of me. However, I haven't tested it.

  You can compile your own third party libraries from https://github.com/willyd/caffe-windows-dependencies . This way is the most recommended, because you can better understand Visual Studio during configuring so much applications.
  
  In addition, you can still debug the codes in Release mode, by following the instructions here https://msdn.microsoft.com/en-us/library/fsk896zz.aspx .

 - How can I create other tools, such as `extract_features.cpp` and `cpp_classification.cpp`?
  
  I have only created projects which is used most frequently in my eyes. If you want to compile other tools, there is no need to create a new project. You can just copy and rename `./build/MSVC` folder to another one, and add the new project to the VS solution. Remove `caffe.cpp` and add your target cpp file. Compile it, then you will get a corresponding exe file in `./bin`.

 - Why can't my VS open the projects?
  
  This is mainly because your CUDA version is different from mine, CUDA 7.0. You can modify the CUDA configurations in the `.vcxproj` file manually by open each `.vcxproj` with notepad or other text processor. 

  However, manually modifying the project file may destroy some of the configurations. Someone reported that the reference relationship between the projects was lost after they modified the project files. You may refer to the second question to solve this problem.
