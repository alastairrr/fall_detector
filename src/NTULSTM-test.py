

import movenet_dai_pose as mnet
import numpy as np
from fall_detection import LSTMFallDetector
import os

# Testing Video from https://doi.org/10.1016/s0140-6736(12)61263-x

DATA_PATH="../data/ntu_npy_padded_original_label"
sample_id = "S001C001P003R002A043"
model = LSTMFallDetector()
model.getModelSummary()
model.modelLoadWeights("./fall_detection/model_weights/ntu_weights.h5")

if __name__ == "__main__":

    sample = np.load(os.path.join(DATA_PATH, f"{sample_id}.npy")).reshape(300,-1)
    print(sample)
    res = model.modelPredict(np.array(sample).reshape(1, 300, 26))

    if res > 0.5: 
        print(f"{res} - fall")
    else:
        print(f"{res} - normal")

