# binary-image-classifier
This is a binary classifier to detect human faces from images. The model used here is a convolutional nerual network. The training data set are images crawled from ImageNet.
## Usage
* Run `python data_gathering.py` to acquire data from ImageNet. This will create two folders, face to store human face images and non-face to store images that are not human faces.
* Run `python train_test.py` to train the convolutional nerual network build in the file model.py. This will output the training process and provide a accuracy. Hyperparameters can be altered in the model file.
