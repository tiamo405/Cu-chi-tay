import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tensorflow.keras as keras

epochs       = 1
batch_size   = 32
width        = 224
height       = 224


def load_data():
  train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
    )
  train_generator = train_datagen.flow_from_directory(
        directory=r"data\\train",
        target_size=(width, height),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
  
  valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
    )
  valid_generator = valid_datagen.flow_from_directory(
        directory=r"data\\valid",
        target_size=(width, height),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42
      )
  test_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
      )
  test_generator = test_datagen.flow_from_directory(
      directory=r"data\\test",
      target_size=(224, 224),
      color_mode="rgb",
      batch_size=1,
      class_mode=None,
      shuffle=False,
      seed=42
  )
  return train_generator, valid_generator,test_generator
train_gen, valid_gen, test_gen=load_data()

def define_model(width, height):

    model_input = tf.keras.layers.Input(shape=(width, height, 3), name='image_input')

    model_main = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet')(model_input)
    
    model_dense1 = tf.keras.layers.Flatten()(model_main)
    model_dense2 = tf.keras.layers.Dense(128, activation='relu')(model_dense1)
    model_out = tf.keras.layers.Dense(5, activation="softmax")(model_dense2)

    model = tf.keras.models.Model(model_input,  model_out)
    optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", 
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    return model

model = define_model(width, height)

model.fit(train_gen, epochs = epochs, validation_data = valid_gen)
result = model.predict(test_gen, verbose = 1)
res=[]
for i in range(len(result)):
  res.append(np.argmax(result[i]))
print(res)
