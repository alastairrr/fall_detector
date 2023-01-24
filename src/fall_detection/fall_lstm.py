"""
 *
 *  LSTMFallDetector =========================================================
 * 
 *  > Description:
 *
 *  > Author: Alastair Kho
 *  > Year: 2023
 *
 * ===========================================================================
 *
 * """


#  ======  Dependencies  ======  #

import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam


#  ======  Classes  ======  #

class LSTMFallDetector:

    def __init__(self, learning_rate=0.001, input_shape=(None,52)) -> None:
        log_dir = os.path.join('Logs')
        self.tb_callback = TensorBoard(log_dir=log_dir)
        self.model = Sequential()
        self.model.add(LSTM(48, return_sequences=True, activation='tanh', input_shape=(None,52)))
        self.model.add(LSTM(32, return_sequences=True, activation='tanh'))
        self.model.add(Dropout(0.4))
        self.model.add(LSTM(32, return_sequences=False, activation='tanh'))

        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))

        self.model.add(Dense(1, activation='sigmoid')) # softmax for multi class, sigmoid for binary

        adam = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy']) 

    def getModelSummary(self) -> None:
        print(self.model.summary())
    
    def modelFit(self, verbose=1, epoch=500) -> None:
        x_train, x_test, y_train, y_test = self.init_test_train()
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=verbose, batch_size=1, callbacks=[self.tb_callback])
        self.model.save('model_weights/action2.h5')

    def _init_test_train(self) -> None:
        pass
