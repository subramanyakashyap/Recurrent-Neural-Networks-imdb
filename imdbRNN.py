# Importing project dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Dataset preprocessing

# Setting up dataset parameters
number_of_words = 20000
max_len = 100

# Loading the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

# Padding all sequences to be the same length
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# Setting up Embedding Layer parameters
vocab_size = number_of_words

# Building a Recurrent Neural Network

# Defining the model
model = tf.keras.Sequential()

# Adding the Embeding Layer
model.add(tf.keras.layers.Embedding(vocab_size, embed_size, input_shape=(X_train.shape[1],)))

# Adding the LSTM Layer
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

# Adding the Dense output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Training the model
model.fit(X_train, y_train, epochs=3, batch_size=128)

# Evaluating the model
test_loss, test_acurracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_acurracy))
