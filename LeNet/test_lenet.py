# append the package folder to sys.path
import sys
sys.path.append("core")

from core import LeNet
from numpy import uint8
from sklearn.model_selection import train_test_split
from sklearn.datasets.mldata import fetch_mldata
from keras.utils import np_utils
import numpy as np
import cv2

# load the MNIST dataset
dataset = fetch_mldata("MNIST original", data_home='output')

# reshape the MNIST dataset
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[ :, :, :, np.newaxis]
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.target.astype("int"), test_size=0.33)

# transform the training and testing labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# Load the training model weights
model = LeNet.build(width=28, height=28, depth=1, classes=10,
                    weightsPath='output/lenet_weights.hdf5')

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    # resize the image from a 28 x 28 image to a 96 x 96 image so we
    # can better see it
    image = (testData[i] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
          np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.imwrite("output/{}.jpg".format(i), image)
    cv2.waitKey(0)