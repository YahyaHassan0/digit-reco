import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnsit
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_train = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.squential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorcial_crossentropy', metrics=['accuracy'])

model.fit(x_test,y_train,epochs=3)

accuracy,loss= model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('digits.model')