
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
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
K.clear_session()

# import zipfile
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/neg2.zip", 'r')
# zip_ref.extractall("/content/train")
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/pos2.zip", 'r')
# zip_ref.extractall("/content/train")

# import zipfile
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/cross_valid_neg (2).zip", 'r')
# zip_ref.extractall("/content/cross_val")
# zip_ref = zipfile.ZipFile("C:/Users/amensodhi/Downloads/cross_valid_positive (2).zip", 'r')
# zip_ref.extractall("/content/cross_val")
# zip_ref.close()

#from keras.applications.vgg16 import VGG16
base_model=ResNet50(include_top = False, pooling = "avg")
from keras.applications.resnet import preprocess_input
model = Sequential()
model.add(base_model)

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu',))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
#model.layers[0].trainable = False


for layer in base_model.layers:
   layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



img_width, img_height = 400,400

train_data_dir = './data/data2/train'
validation_data_dir = './data/data2/cross_val'
nb_train_samples = 1282
nb_validation_samples = 503
epochs = 12
batch_size = 16
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen=ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
    channel_shift_range=150.0,
    brightness_range=(0.3, 0.9),
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		preprocessing_function=preprocess_input,
		fill_mode="nearest")

train_generator = train_datagen.flow_from_directory(
    "./data/data2/train",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    interpolation="bicubic",
    class_mode='binary')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_generator = test_datagen.flow_from_directory(
    './data/data2/cross_val',
    target_size=(img_width, img_height),
    interpolation="bicubic",
    batch_size=batch_size,
    class_mode='binary')

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20
)

model.save_weights("./modelNew/mymodelweights_2class.h5")
#!mv "/content/my_weights_resnet50.h5" "/content/drive/My Drive/open_src_data/cross_valid/my_weights_resnet50.h5"

model.save("./modelNew/mymodel_2class.h5")
#!mv "/content/mymodel3_r.h5" "/content/drive/My Drive/open_src_data/cross_valid/mymodel3_r.h5"

for layer in base_model.layers[:165]:
  layer.trainable = False
for layer in base_model.layers[165:]:
  layer.trainable= True

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=13
)

model.save_weights("./modelNew/mymodelweights_2class.h5")
#!mv "/content/my_weights_resnet50_f.h5" "/content/drive/My Drive/open_src_data/cross_valid/my_weights_resnet50_f.h5"

model.save("./modelNew/mymodel_2class.h5")
#!mv "/content/mymodel3_r_f.h5" "/content/drive/My Drive/open_src_data/cross_valid/mymodel3_r_f.h5"

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

