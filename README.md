# MNIST-Classifier

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

  to recognize your .bmp images in the path (./unlabeled/), the results will be been both in console and the path (./labeled) (as the prefix of the file names).
  
