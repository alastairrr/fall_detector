from sklearn.model_selection import train_test_split
import os
import numpy as np
from fall_detection import LSTMFallDetector
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.model_selection import cross_val_score, KFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

DATA_PATH = os.path.join('../data/ntu_npy_padded')
no_sequences = 50 # 50 video datas each
sequence_length = 300 # 300 frames to classify

def main():

    actions = np.array(['not_fall', 'fall'])
    datasize = np.array([54072, 929])
    label_map = []

    sequences, labels = [], []

    for idx, action in enumerate(actions):
        no_samples = datasize[idx]
        sample = None

        for sample_id in range(0, no_samples):
            sample = np.load(os.path.join(DATA_PATH, action, f"{sample_id}.npy")).reshape(300,-1)
            #sample = print(np.load(os.path.join("../../data/ntu_npy_padded", "fall", "0.npy")).flatten().shape)

            sequences.append(np.array(sample))
            labels.append(idx)

    x = np.array(sequences)
    y = np.array(labels)



    model = LSTMFallDetector()
    model.getModelSummary()

    initial_weights = model.getModelWeights()



    # ------------------------
    num_splits = 5 # 100 samples, so 80 train, 20 test each fold
    #cross_val = KFold(n_splits=num_splits, shuffle=True)
    stratKfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    #shufflesplit = StratifiedShuffleSplit(n_splits=num_splits, random_state=42, test_size=0.2)

    scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []

    count = 1
    for train_index, test_index in stratKfold.split(x,y): # split uniquely and randomly between folds.
        print("\nfold = ", count)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.setModelWeights(initial_weights) # reset weights
        model.model.fit(x_train, y_train, epochs=20, verbose=1, batch_size=32) 

        y_pred = model.modelPredict(x_test)
        y_discretized = np.where(y_pred > 0.5, 1, 0) # observations = minimum confidence value here is proportional to epochs. Lower epochs seems to cap the confidence of each fall
        # lower confidence means model may have more false positive, higher confidence is stricter - model has to be confident that a fall occurs.
        # setting this too high may mean that true positives + false positives == 0, thus 0 for recall metric.

        # Evaluate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_discretized, average='binary')
        print(y_test, y_discretized)
        auc = roc_auc_score(y_test, y_discretized)

        scores.append(model.model.evaluate(x_test, y_test)[1])  # Append accuracy score
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)
        count += 1

    # Calculate average scores
    average_score = np.mean(scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_f1 = np.mean(f1_scores)
    average_auc = np.mean(auc_scores)

    print("Cross-validation Binary Accuracies:", scores)
    print("Average accuracy score:", average_score)
    print("\n")
    print('Average Precision:', average_precision)
    print('Average Recall:', average_recall)
    print('Average F1-score:', average_f1)
    print("Average AUC score:", average_auc)

if __name__ == "__main__":
    main()