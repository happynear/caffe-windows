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
   the layers and solvers.
   
   If you came across similar error when using `caffe.exe`, I guess you may have modified the `.vcxproj` files manually by yourself. VS lost some configurations during your modification. Never mind, you can still follow the above instructions to fix it.
   
 - How to compile the codes in Debug mode?
   
  There is a [3rdparty library archive file](http://pan.baidu.com/s/1qW88MTY) provided by a friend of me. However, I haven't tested it.

  You can compile your own third party libraries from https://github.com/willyd/caffe-windows-dependencies . This way is the most recommended, because you can better understand Visual Studio during configuring so much applications.
  
  In addition, you can still debug the codes in Release mode, by following the instructions here https://msdn.microsoft.com/en-us/library/fsk896zz.aspx .
