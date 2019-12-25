import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from multiprocessing import Process


def startTensorboard(logdir):
    # Start tensorboard with system call
    os.system("tensorboard --logdir {}".format(logdir))


def fitModel():
    # Create your model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Some mock training data
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    # Run the fit function
    model.fit(data, labels, epochs=100, batch_size=32)


if __name__ == '__main__':
    # Run both processes simultaneously
    Process(target=startTensorboard, args=("logs",)).start()
    Process(target=fitModel).start()