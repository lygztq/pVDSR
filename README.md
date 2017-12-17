# Image Super-Resolution Using Deep Concolutional Neural Networks with PReLU

- - -

## Experimental Setup

Firstly, this project was implemented using Python 2.7, so at the begining you should check your Python version. 

The Python package needed:

* [TensorFlow(GPU version)](https://www.tensorflow.org/)
* [Numpy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/install.html)
* [matplotlab](http://matplotlib.org/)
* [PIL](http://www.pythonware.com/products/pil/)
* pickle
* argparse
* glob

Secondly, we use MATLAB(Linux version) to preprocess the training data(because when we generate the training data from image, using MATLAB is much faster than using openCV on Python). So make sure you have already installed MATLAB on your server.

Finally, you can get the training data from: [data](https://jbox.sjtu.edu.cn/l/hJjgMC)

- - -

## How To Use

#### preprocessing data

* Download the training and test data from [here](https://jbox.sjtu.edu.cn/l/hJjgMC)
* Unzip the data.zip, unzip 291.zip and Set14.zip inside.
* Copy the 291 and Set14 directory to "/data" directory.
* Run generate_train.m and generate_test.m to generate training and test data.

#### Training

Run "train.py" to train. If you want to start from a checkpoint, using "py -2 train.py --model_path [your checkpoint path]"

#### Testing

Run "python test.py"

#### Plot Result

Run "python plot.py"
