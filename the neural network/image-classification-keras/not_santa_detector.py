#！/usr/bin/env python3
# -*- coding:UTF-8 -*-
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import os

# define the paths to the Not Santa Keras deep learning model and
MODEL_PATH = r"E:\学习\Python\image-classification-keras\santa_not_santa.model"
# initialize the total number of frames that *consecutively* contain
# santa along with threshold required to trigger the santa alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20
# initialize is the santa alarm has been triggered
SANTA = False
# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)
while True:
    start = time.clock()
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    image = cv2.resize(frame, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image and initialize the label and
    # probability of the prediction
    (notSanta, santa) = model.predict(image)[0]
    label = "Not Santa"
    proba = notSanta
    # check to see if santa was detected using our convolutional
    # neural network
    if santa > notSanta:
        # update the label and prediction probability
        label = "Santa"
        proba = santa

        # increment the total number of consecutive frames that
        # contain santa
        TOTAL_CONSEC += 1
    if not SANTA and TOTAL_CONSEC >= TOTAL_THRESH:
        SANTA = True
    # otherwise, reset the total number of consecutive frames and the
    # santa alarm
    else:
        TOTAL_CONSEC = 0
        SANTA = False

    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(time.clock()-start)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()