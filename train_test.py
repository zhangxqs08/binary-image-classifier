__author__ = 'Xiang'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


from model import load_data
from model import preprocess_data
from model import keras_model
from model import simple_model
from keras.utils.np_utils import to_categorical
from keras import optimizers


size = 64
class_size = 2000
batch_size = 32

X, y = load_data(size, class_size)
n_classes = len(np.unique(y))
print "Number of classes =", n_classes
print "Img set shape", X.shape

X = preprocess_data(X)
y = to_categorical(y)
print "y shape", y.shape

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
print "train shape X", X_train.shape
print "train shape y", y_train.shape

np.save("X_train.data", X_train)
np.save("y_train.data", y_train)
np.save("X_test.data", X_test)
np.save("y_test.data", y_test)

X_train = np.load("X_train.data.npy")
y_train = np.load("y_train.data.npy")
X_test = np.load("X_test.data.npy")
y_test = np.load("y_test.data.npy")

label_binarizer = LabelBinarizer()
input_shape = (size, size, 1)
model = simple_model(input_shape)

sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_split=0.2)

y_one_hot_test = label_binarizer.fit_transform(y_test)
metrics = model.evaluate(X_test, y_test)
for i in xrange(len(model.metrics_names)):
    metric_name = model.metrics_names[i]
    metric_value = metrics[i]
    print "{}: {}".format(metric_name, metric_value)

# model.save('model.h5')
