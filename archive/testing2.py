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


model=models.load_model("./models/mymodel_2class.h5")
print("Model Loaded !!\n")
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
        print("\n")
        print("=============== ",filename," ======================")
        # print(p)
        if(preds[0]>0.5):
            label="Height"
            print(label + ", Probability " + str(preds[0]))
        else:
            label=" Not height"
            print(label + ", Probability " + str(preds[0]))

        cv2.imshow('image',img)
        cv2.waitKey(0)
        print("=========================================")