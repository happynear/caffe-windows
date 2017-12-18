GoogLeNet --model=train_val_googlenet.prototxt --weights=thinned_net.caffemodel --imagefile="%~f1" --labelfile=synset_words.txt --gpu=0
pause