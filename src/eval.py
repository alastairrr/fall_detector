import numpy as np
import cv2
from movenet_dai_pose import MovenetDAI
from fall_detection import LSTMFallDetector

def main():
    model = LSTMFallDetector()
    model.getModelSummary()
    model.modelLoadWeights("./fall_detection/model_weights/ntu_weights.h5")

    x_test, y_test = model._init_val_set("../data/ntu_npy_padded_val")

    y_pred = model.modelPredict(x_test)
    
    y_discretized = []
    for i in y_pred:
        if i[0] > 0.5:
            y_discretized.append([1])
        else:
            y_discretized.append([0])
    y_discretized = np.array(y_discretized)

    """
    for i in range(len(y_pred)):
        print(y_pred[i][0], y_discretized[i][0])
    print(y_discretized)
    print(y_pred)"""

    #-------------------------------------------------

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import roc_auc_score

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_discretized, average='binary')
    auc_score = roc_auc_score(y_test, y_discretized)

    print("\n")

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1_score)

    print("AUC score:", auc_score)

if __name__ == "__main__":
    main()