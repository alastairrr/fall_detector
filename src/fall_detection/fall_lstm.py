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


DATA_PATH = os.path.join('../../data/extracted_data')


#  ======  Classes  ======  #

class LSTMFallDetector:

    def __init__(self, learning_rate=0.001, input_shape=(55,52)) -> None:
        log_dir = os.path.join('../../Logs')
        self.tb_callback = TensorBoard(log_dir=log_dir)
        self.model = self._model_compile(learning_rate, input_shape)
        
    def getModelSummary(self) -> None:
        print(self.model.summary())
    
    def modelFit(self, verbose=1, epoch=500) -> None:
        x_train, x_test, y_train, y_test = self._init_test_train()
        class_weights = {0:0.3/0.7,1:0.4/0.7}
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=verbose, class_weight=class_weights, callbacks=[self.tb_callback])
        self.model.save('model_weights/action2.h5')

    def modelLoadWeights(self, weights):
        self.model.load_weights(weights)
    
    def modelPredict(self, sequence):
        return self.model.predict(np.expand_dims(sequence,axis=0))[0]

    def _model_compile(self, learning_rate, input_shape) -> Sequential:
        model = Sequential()
        model.add(LSTM(48, return_sequences=True, activation='tanh', input_shape=input_shape))
        model.add(LSTM(32, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.4))
        model.add(LSTM(32, return_sequences=False, activation='tanh'))

        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation='sigmoid')) # softmax for multi class, sigmoid for binary

        adam = Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy']) 
        return model


    def _get_count_in_dir(self, dir_path) -> int:
        count = 0
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)) or os.path.isdir(os.path.join(dir_path, path)):
                count += 1
        return count

    def _init_test_train(self, test_size=0.2) -> None:
        actions = np.array(['fall', 'not_fall'])
        sequences, labels = [], []
        for idx, action in enumerate(actions):
            no_sequences = self._get_count_in_dir(os.path.join(DATA_PATH, action))

            for sequence in range(1, no_sequences+1):
                window = []
                sequence_length = self._get_count_in_dir(os.path.join(DATA_PATH, action, str(sequence)))

                for frame_num in range(0, sequence_length):
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(np.array(window))
                labels.append(idx)
        x = np.array(sequences)
        y = np.array(labels)
        print(x.shape)
        return train_test_split(x , y, test_size=test_size)
        # x[0:49], x[49:69], y[0:49], y[49:69]

