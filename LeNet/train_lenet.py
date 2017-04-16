# append the package folder to sys.path
import sys
sys.path.append("core")

# import the necessary packages
from core import LeNet
from sklearn.cross_validation import train_test_split
from sklearn.datasets.mldata import fetch_mldata
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

# grab the MNIST dataset (first time download later use the local file)
print("[INFO] downloading MNIST...")
dataset = fetch_mldata("MNIST original", data_home='output')

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[ :, :, :, np.newaxis]
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.target.astype("int"), test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model for training
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10,
                    weightsPath=None)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Training the CNN
print("[INFO] training...")
model.fit(trainData, trainLabels, batch_size=128, epochs=20,
		verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# Save weights trained
print("[INFO] dumping weights to file...")
model.save_weights('output/lenet_weights.hdf5', overwrite=True)

print("[INFO] Done!")