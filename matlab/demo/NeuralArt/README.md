A Caffe Implementation of Neural Art
============

Based on paper
> A Neural Algorthm of Artistic Style' by Leon Gatys, Alexander Ecker, and Matthias Bethge (http://arxiv.org/abs/1508.06576).

The model used here is VGG-16. I have thinned the net model with only conv layer paramters retaining. The thinned model can 
be downloaded from http://pan.baidu.com/s/1kT8d3Iv .

Usage
===========
I exploited this on my laptop and it's too slow to tune the parameters to be the best. The performance is not very good right now. However, you can still try it.

For caffe linux users: I have written a new layer called [*covariance layer*](https://github.com/happynear/caffe-windows/blob/master/src/caffe/layers/covariance_layer.cpp) to calculate the covariance matrix* of a feature map. If you want to run this code with your own caffe, please add the layer to your caffe project.


*: Actually, I haven't done the mean substraction, so it should be named as correlation matrix?
