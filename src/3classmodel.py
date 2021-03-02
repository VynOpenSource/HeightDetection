
import numpy as np
import tensorflow as tf
import math
import h5py
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.regularizers import l2
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
# %matplotlib inline
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import tensorflow_addons as tfa
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
K.clear_session()

def categorical_focal_loss(num_classes=3, gamma=2., alpha=.25, smooth_alpha=0.05):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        if smooth_alpha > 0:
            y_true = y_true * (1 - smooth_alpha) + smooth_alpha / num_classes

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

# import zipfile
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/neg2.zip", 'r')
# zip_ref.extractall("/content/train")
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/pos2.zip", 'r')
# zip_ref.extractall("/content/train")
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/train_hard.zip", 'r')
# zip_ref.extractall("/content/train")

# import zipfile
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/cross_valid_neg (2).zip", 'r')
# zip_ref.extractall("/content/cross_val")
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/cross_valid_positive (2).zip", 'r')
# zip_ref.extractall("/content/cross_val")
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/hard_cross_val.zip", 'r')
# zip_ref.extractall("/content/cross_val")
# zip_ref.close()

#from keras.applications.vgg16 import VGG16
base_model=ResNet50(include_top = False, pooling = "avg")
from keras.applications.resnet import preprocess_input
model = Sequential()
model.add(base_model)

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='relu',))
model.add(layers.Dense(3, activation='softmax'))
#model.layers[0].trainable = False

for layer in base_model.layers:
   layer.trainable = False

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

model.compile(optimizer='RMSprop', loss=[focal_loss], metrics=['accuracy'])

img_width, img_height = 400,400

train_data_dir = './data/train'
validation_data_dir = './data/cross_val'
nb_train_samples = 1282
nb_validation_samples = 503
epochs = 12
batch_size = 16
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen=ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=20,
		zoom_range=0.15,
    channel_shift_range=150.0,
    brightness_range=(0.3, 0.9),
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
#

train_generator = train_datagen.flow_from_directory(
    "./data/train",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    interpolation="bicubic",
    class_mode='categorical')

#
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_generator = test_datagen.flow_from_directory(
    './data/cross_val',
    target_size=(img_width, img_height),
    interpolation="bicubic",
    batch_size=batch_size,
    class_mode='categorical')

#reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                         #     patience=2, min_lr=0.00001)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=7
)

model.save("./modelNew/mymodel_3class.h5")


for layer in base_model.layers[:165]:
  layer.trainable = False
for layer in base_model.layers[165:]:
  layer.trainable= True

model.compile(optimizer='adam', loss=[focal_loss], metrics=['accuracy'])

history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=14
)

model.save_weights("./modelNew/mymodelweights_3class.h5")

model.save("./modelNew/mymodel_3class.h5")


print(history.history.keys())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

