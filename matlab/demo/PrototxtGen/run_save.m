activation = 'Sigmoid';
layers = {
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
    struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 3, 'activation', activation) %convolution layer
%     struct('type', 'convolution', 'outputmaps', 80, 'kernelsize', 3, 'activation', activation) %convolution layer
%     struct('type', 'convolution', 'outputmaps', 80, 'kernelsize', 3, 'activation', activation) %convolution layer
%     struct('type', 'convolution', 'outputmaps', 10, 'kernelsize', 3, 'activation', activation) %convolution layer
%     struct('type', 'pooling', 'scale', 2, 'method', 'AVE')  
%     struct('type', 'inception', 'node1x1', 100, 'reduce3x3', 50, 'node3x3', 100, 'reduce5x5', 50, 'node5x5', 100, 'poolconv', 100) 
%     struct('type', 'inception', 'node1x1', 50, 'reduce3x3', 25, 'node3x3', 50, 'reduce5x5', 25, 'node5x5', 50, 'poolconv', 50) 
%     struct('type', 'inception', 'node1x1', 20, 'reduce3x3', 10, 'node3x3', 20, 'reduce5x5', 10, 'node5x5', 20, 'poolconv', 20) 
%     struct('type', 'inception', 'node1x1', 20, 'reduce3x3', 10, 'node3x3', 20, 'reduce5x5', 10, 'node5x5', 20, 'poolconv', 20) 
%     struct('type', 'inception', 'node1x1', 20, 'reduce3x3', 10, 'node3x3', 20, 'reduce5x5', 10, 'node5x5', 20, 'poolconv', 20) 
%     struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 1, 'activation', activation) %convolution layer
%     struct('type', 'convolution', 'outputmaps', 20, 'kernelsize', 1, 'activation', activation) %convolution layer
%     struct('type', 'pooling', 'scale', 2, 'method', 'AVE') 
};
CNNComplexitySave(layers,'8convSigmoid',[640 480],10);