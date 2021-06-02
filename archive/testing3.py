# import warnings
# warnings.filterwarnings("ignore")
import os
from tensorflow.keras import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras import models
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# import cv2_imshow
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

model=models.load_model("./models/mymodel_3class.h5",custom_objects={'focal_loss':focal_loss})
print("Model3 Loaded !!\n")
# # download_file_from_google_drive(id,"\imageTest.jpg")

dir = "testingImages"

for filename in os.listdir(dir):
    if filename.endswith(".jpg") or filename.endswith(".png"): 

        p = os.path.join(dir, filename)
        
        img = cv2.imread(p,cv2.IMREAD_UNCHANGED)

        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
        image = img_to_array(img)
        image = np.expand_dims(image, axis=0)
        preds = model.predict(image)
        class_=np.argmax(preds[0])

        print("\n")
        print("=============== ",filename," ======================")
        # print(p)
        if(class_==0):
            label="not_height"    
        elif class_==1:    
            label="height"
        else:
            label="hard"
        
        print(label + ", Probability " + str(preds[0][class_]))

        cv2.imshow('image',img)
        cv2.waitKey(0)
        print("=========================================")