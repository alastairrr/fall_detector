
import numpy as np
import cv2
from movenet_dai_pose import MovenetDAI
from fall_detection import LSTMFallDetector

def flatten_keypoints(kps, norm_kps):
    pose = np.concatenate((kps, norm_kps), axis=1)
    print("pose with norms:\n", pose)
    flattened = pose.flatten()
    print("flattened:\n", flattened)

    print("\n")
    return flattened

def main():
    model = LSTMFallDetector()
    model.getModelSummary()
    model.modelLoadWeights("./fall_detection/model_weights/action.h5")

    movenet = MovenetDAI()
    sequence = [] # get 55 frames then pass to detection
    predictions = []

    while True:
        frame, data, _ = movenet.getFrameInference()
        if frame is None: break
        if frame is not None:
            movenet.renderKeyPoints(frame, data)
            normalized = movenet.normalize(data)
            normDataFrame = movenet.renderNormalized(frame, movenet.normalize(data))
            normDataFrame = cv2.resize(normDataFrame, (1360, 480)) 

            flattened = flatten_keypoints(data, normalized)
            sequence.append(flattened)
            sequence = sequence[-55:]
            if len(sequence) == 55:
                res = model.modelPredict(sequence)

                if res < 0.5: 
                    currentPrediction = f"{res} - fall"
                    cv2.putText(normDataFrame, currentPrediction, (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 4, cv2.LINE_AA)

                else:
                    currentPrediction = f"{res} - normal"
                    cv2.putText(normDataFrame, currentPrediction, (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200, 0), 4, cv2.LINE_AA)
                    
                predictions.append(np.argmax(res))
            cv2.imshow("preview", normDataFrame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()

