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

    def __init__(self) -> None:
        pass
