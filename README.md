# keras
Machine Learning using Keras as a front-end and Tensorflow (GPU) as backend

## kerasCNN.py and kerasCNNpredict.py

A simple Convolutional Neural Network with 2 sets of Convolution/Maxpool layers, a flatten layer and two fully-connected layers ending in a softmax. Several lines are commented out; they will be restored if they appreciably impact the model accuracy. My test system is:

* Dell Inspiron 15 7000
* Intel Core i7 6700HQ CPU @ 2.60GHz
* 8 GB of DDR3 SDRAM
* Nvidia GTX 960M GPU with 4GB GDDR5 VRAM

Using the CPU verison of Tensorflow (backend) epochs routinely took over 530 seconds to complete; however GPU acceleration now yields epochs that take ~ 55 seconds to complete - a ~10x drop. My CUDA information is:

* CUDA Toolkit 8.0
* cuDNN v5.1
* tensorflow-gpu 1.1.0 <pip>

### Training and Testing

Here are the last few epochs of training on a dataset from Wikimedia Commons used in a Google Developers Codelab for Machine Learning. With the last epoch, the model attained an accuracy of 87.25%.

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)

Now, with the weights of the model saved as a .h5 file, I use the prediction module to predict m.jpg, an image from the test set:



