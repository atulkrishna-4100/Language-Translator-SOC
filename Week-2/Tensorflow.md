## Tensorflow
``` python
# Basic TensorFlow Operations
import tensorflow as tf
# Create some tensors
a = tf.constant(2)
b = tf.constant(3)
# Add two tensors
c = tf.add(a, b)
print(c)  

# Building a Simple Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
x_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=(1000, 1))

model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)
loss, accuracy = model.evaluate(x_train, y_train)
print(f'Loss: {loss}, Accuracy: {accuracy}')
