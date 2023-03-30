"""
The main code for the recurrent and convolutional networks assignment.
See README.md for details.
"""
from keras.models import Sequential
from keras.layers import Conv1D,Conv2D, MaxPooling2D,Embedding, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Flatten,Dense,SimpleRNN,Dropout

import os
from typing import Tuple, List, Dict
import tensorflow as tf
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def create_toy_rnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tf.keras.models.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = tf.keras.Sequential()
    # LSTM (Long Short-Term Memory)
    #   Designed to handle the vanishing gradient problem that can occur in RNN
    model.add(tf.keras.layers.LSTM(16, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(8, return_sequences=True))
    # Regression problem
    model.add(tf.keras.layers.Dense(n_outputs, activation="linear"))
    # Converge quickly and accurately (Weights minimazing loss function)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # MSE: Commonly used for regression 
    model.compile(optimizer=optimizer, loss='mse')

    kwargs = {'batch_size': 1}

    return (model, kwargs)

def create_mnist_cnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tf.keras.models.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = Sequential()
    # Used to extract features from sequential data. 2d for the image
    model.add(tf.keras.layers.Conv2D(256,kernel_size=(2,2),strides=(1,1),\
                                     activation='tanh',input_shape=input_shape))    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(128,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(64,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(32,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(16,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    # reduces the spatial size of the feature maps even further, 
    #   resulting in a smaller set of features
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    # Helps prevent overfitting
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # Multiclass classification problems
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    # SGD: Simple and effective optimization algorithm
    # Loss: Measures the difference between the true label and 
    #   the predicted probability distribution over all classes
    model.compile(loss='categorical_crossentropy',optimizer="sgd",metrics=['accuracy'])
    kwargs = {'verbose':1,'epochs':100,'validation_data':None}

    return (model,kwargs)

    model = tf.keras.models.Sequential()
    # Used to extract features from sequential data. 2d for the image
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),\
                            activation='relu', input_shape=input_shape))
    # Reduces the spatial size of the feature maps produced by the convolutional layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Helps prevent overfitting
    model.add(tf.keras.layers.Dropout(0.2))
    # reduces the spatial size of the feature maps even further, 
    #   resulting in a smaller set of features
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # Multiclass classification problems
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    # Adam: Converge quickly and accurately (Weights minimazing loss function)
    # Loss: Measures the difference between the true label and 
    #   the predicted probability distribution over all classes
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Stop training when the validation loss stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    kwargs = {"batch_size": 128, 'epochs': 20, "verbose": 2,"callbacks":[early_stop]}

    return (model, kwargs)

def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tf.keras.models.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(len(vocabulary), 128))
    # Processes the input sequence in both forward and backward directions
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    # Helps prevent overfitting
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))

    # Converge quickly and accurately (Weights minimazing loss function)
    # Loss function for binary classification problems
    model.compile(loss='binary_crossentropy', optimizer="Adam",metrics=['accuracy'])

    # Stops training if the loss func doesn't improve for 10 epochs
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                  restore_best_weights=True)

    kwargs = {"batch_size": 128, 'epochs': 50, "verbose": 2, 
              "callbacks": [early_stop]}

    return (model, kwargs)

def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tf.keras.models.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = tf.keras.models.Sequential()
    # Create a dense vector representation of words
    model.add(tf.keras.layers.Embedding(len(vocabulary), 128, input_length=200))
    # Used to extract features from sequential data
    model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
    # Reduces the spatial size of the feature maps produced by the convolutional layer
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
    # reduces the spatial size of the feature maps even further, 
    #   resulting in a smaller set of features
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    # Helps prevent overfitting
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    # Binary classification for each input
    model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
    
    # Converge quickly and accurately (Weights minimazing loss function)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Loss function for binary classification problems
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    kwargs = {"batch_size": 128, 'epochs': 200, "verbose": 0}

    return (model, kwargs)

