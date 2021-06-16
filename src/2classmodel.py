import tensorflow as tf
import h5py
import numpy
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model, load_model
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
K.clear_session()

#Initializig base model resnet 50 with imagenet weights
base_model=ResNet50(include_top = False, pooling = "avg")
from keras.applications.resnet import preprocess_input
model = Sequential()
model.add(base_model)

#adding connected layers at resnet head for fine tuning
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu',))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

#freezing the resnet and only training the added layers
for layer in base_model.layers:
   layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#setting up data loading and augmentation pipeines
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

train_datagen=ImageDataGenerator(rotation_range=20,zoom_range=0.15,channel_shift_range=150.0,brightness_range=(0.3, 0.9),width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,preprocessing_function=preprocess_input,fill_mode="nearest")

train_generator = train_datagen.flow_from_directory("./data/data2/train",target_size=(img_width, img_height),batch_size=batch_size,interpolation="bicubic",class_mode='binary')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator = test_datagen.flow_from_directory('./data/data2/cross_val',target_size=(img_width, img_height),interpolation="bicubic",batch_size=batch_size,class_mode='binary')

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history=model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,validation_steps=STEP_SIZE_VALID,epochs=20)

#unfreezing last few alyers of resent  
for layer in base_model.layers[:165]:
  layer.trainable = False
for layer in base_model.layers[165:]:
  layer.trainable= True

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
filepath = './modelNew/mymodel_2class.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_loss",verbose=1,save_best_only=True,save_weights_only=False,mode="auto",save_freq="epoch")
callbacks = [checkpoint]
history=model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,validation_steps=STEP_SIZE_VALID,epochs=20,callbacks=callbacks)



#plot stats
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
