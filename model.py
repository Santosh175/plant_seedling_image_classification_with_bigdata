# for the convolutional network
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import  ReduceLROnPlateau, ModelCheckpoint
from keras.wrappers.scikit_learn import kerasClassifier

import config

def cnn_model(kernel_size = (3,3)
pool_size = (2,2),
first_filters = 32,
second_filters = 64,
third_filters = 128,

dropout_conv = 0.3,
dropout_dense = 0.3,
image_size = 50):

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu',
                input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(Conv2D(first_filters,kernel_size,activation = 'relu'))

#model.add(Conv2D(first_filters, kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(Conv2D(second_filters,kernel_size,activation='relu'))
# model.add(Conv2D(second_filters,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters,kernel_size,activation='relu'))
model.add(Conv2D(third_filters,kernel_size,activation='relu'))
#model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256,activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(12,activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile( loss='binary_crossentropy',optimizer=opt,
              metrics=['accuracy'])

return model

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1,
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=1,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

cnn_clf = KerasClassifier(build_fn = cnn_model,
                          batch_size = config.Batch_SIZE,
                          validation_split =10,
                          epochs = config.EPOCHS,
                          verbose = 2,
                          callbacks = callbacks_list,
                          image_size = config.IMAGE_SIZE)