# MNIST-Classifier

## Prerequisites

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies
- ``torch >= 1.0.0``

## Prepare Dataset

+ `Make Folder`

    ```sh
    mkdir ./dataset/
    ```

+ `Download`
    
    ```sh
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O ./dataset/train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O ./dataset/train-labels-idx1-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O ./dataset/t10k-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O ./dataset/t10k-labels-idx1-ubyte.gz
    ```
    
+ `Unpack`
    ```sh
    gzip -d ./dataset/train-images-idx3-ubyte.gz
    gzip -d ./dataset/train-labels-idx1-ubyte.gz
    gzip -d ./dataset/t10k-images-idx3-ubyte.gz
    gzip -d ./dataset/t10k-labels-idx1-ubyte.gz
    ```

## Usage

+ `1.Train` Use:

    ```sh
    python main.py train
    ```
    
  to train model, and your model will be saved to ./model.pth

+ `2.Test` Use: 

    ```sh
    python main.py test
    ```

  to test your model via the test dataset. 

+ `3.Recognize your own images` Use:

    ```sh
    python main.py recognize
    ```

  to recognize your `.bmp` images in the path (`./unlabeled/img_name.bmp`) (the images must be `28*28` in size, and with `256 channels in black&white`), the results will be been both in console and the path (`./labeled/result_img_name.bmp`) (the 'result' refers to the recognized number).
  
