# append the package folder to sys.path
import sys
sys.path.append("core")

# import the necessary packages
from core import LeNet, load_images, deshear, normalize_images
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import cv2
# Loading image data sets and normalizing color scale
rw_set, rw_file_labels = load_images("output/images/real_world/")
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
rw_labels = np_utils.to_categorical(rw_labels, classes)

# Initialize the optimizer and model for training
print("[INFO] compiling model...")
aps = LeNet(rw_set[1].shape, conv_1, pool_1, conv_2, pool_2, hidden,
            classes)
aps.model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=["accuracy"])
aps.model.load_weights('output/lenet_weights.hdf5')

# Testing aps model of both sets
print("[INFO] Test model in real world set...")
rw_probs = aps.model.predict(rw_set)
rw_prediction = rw_probs.argmax(axis=1)

# show the accuracy on the real world
print("[INFO] evaluating real world set...")
(loss, accuracy) = aps.model.evaluate(rw_set, rw_labels, verbose=0)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100) +
      " - loss: {:.2f}".format(loss))

imlabel = ['any', 'einstein', 'curie']
for i in np.random.choice(np.arange(0, len(rw_labels)), size=(12,)):
    print("[INFO] Predicted: {}, Actual: {}".format(rw_prediction[i],
		np.argmax(rw_labels[i])))
    image = (rw_set[i] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    cv2.putText(image, str(imlabel[rw_prediction[i]]), (5, 20),
	 	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow("Digit", image)
    #cv2.imwrite("output/{}.jpg".format(i), image)
    cv2.waitKey(0)
    

