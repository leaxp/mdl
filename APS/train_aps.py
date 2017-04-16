# append the package folder to sys.path
import sys
sys.path.append("core")

# import the necessary packages
from core import LeNet, load_images, deshear, normalize_images
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

# Loading image data sets and normalizing color scale
training_set, training_labels = load_images("output/images/train/")
test_set, test_labels = load_images("output/images/test/")
rw_set, rw_file_labels = load_images("output/images/real_world/")
training_set = normalize_images(training_set)
training_set = training_set[..., np.newaxis]
test_set = normalize_images(test_set)
test_set = test_set[..., np.newaxis]
rw_set = normalize_images(rw_set)
rw_set = rw_set[..., np.newaxis]
rw_set = np.array([x for (y, x) in sorted(zip(rw_file_labels, rw_set))])

# Getting labels for real world set from file
f = open('output/images/real_world/labels.txt', "r")
lines = f.readlines()
rw_labels = []
for x in lines:
    rw_labels.append(int((x.split('	')[1]).replace('\n', '')))
f.close()

# Parameters for LeNet convolutional network
classes = 3  # number of classes to identify
hidden = 500  # number of nuerons in hidden layer
conv_1 = (20, (15, 15))  # (num of filters in first layer, filter size)
conv_2 = (50, (15, 15))  # (num of filters in second layer, filter size)
pool_1 = ((6, 6), (6, 6))  # (size of pool matrix, stride)
pool_2 = ((6, 6), (6, 6))  # (size of pool matrix, stride)

# Converting integer labels to categorical labels
training_labels = np_utils.to_categorical(training_labels, classes)
test_labels = np_utils.to_categorical(test_labels, classes)
rw_labels = np_utils.to_categorical(rw_labels, classes)

# Initialize the optimizer and model for training
print("[INFO] compiling model...")
aps = LeNet(training_set[1].shape, conv_1, pool_1, conv_2, pool_2, hidden,
            classes)
aps.model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=["accuracy"])

# Training the CNN
print("[INFO] training...")
aps.model.fit(training_set, training_labels, batch_size=10, epochs=50,
              verbose=1)

# Testing aps model of both sets
print("[INFO] Test model in both sets...")
test_probs = aps.model.predict(test_set)
test_prediction = test_probs.argmax(axis=1)
rw_probs = aps.model.predict(rw_set)
rw_prediction = rw_probs.argmax(axis=1)

# show the accuracy on the testing set
print("[INFO] evaluating test set...")
(loss, accuracy) = aps.model.evaluate(test_set, test_labels, verbose=0)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100) +
      " - loss: {:.2f}".format(loss))

# show the accuracy on the real world
print("[INFO] evaluating real world set...")
(loss, accuracy) = aps.model.evaluate(rw_set, rw_labels, verbose=0)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100) +
      " - loss: {:.2f}".format(loss))

# Save weights trained
print("[INFO] dumping weights to file...")
aps.model.save_weights('output/lenet_weights.hdf5', overwrite=True)

print("[INFO] Done!")
