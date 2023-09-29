
import numpy as np
import cv2
from movenet_dai_pose import MovenetDAI
from fall_detection import LSTMFallDetector
import time

def reshape_keypoints(kps):
    #print("pose with norms:\n", pose)
    flattened = kps.reshape(-1)
    #print("flattened:\n", flattened)

    #print("\n")
    return flattened

def main():
    model = LSTMFallDetector()
    model.getModelSummary()
    model.modelLoadWeights("./fall_detection/model_weights/ntu_weights.h5")

    movenet = MovenetDAI()
    sequence = [] # get 55 frames then pass to detection
    predictions = []
    firstRun = True
    #smoothing ideas: simple moving average?
    while True:
        startTime = time.time()
        frame, data, _ = movenet.getFrameInference()
        timeTaken = time.time() - startTime

        if frame is None: break
        if frame is not None:
            movenet.renderKeyPoints(frame, data)
            normalized = movenet.normalize(data)
            normDataFrame = movenet.renderNormalized(frame, movenet.normalize(data))
            normDataFrame = cv2.resize(normDataFrame, (1564, 552)) 
            normDataFrame = cv2.flip(normDataFrame,1)

            reshaped = reshape_keypoints(normalized)
            sequence.append(reshaped)
            if firstRun == True:
                for i in range(300):
                    sequence.append(reshaped)
                firstRun = False
            sequence = sequence[-300:]
            if len(sequence) == 300:
                res = model.modelPredict(np.array(sequence).reshape(1, 300, 26))
                currentPrediction = None
                if res > 0.5: 
                    currentPrediction = f"{res} - fall"
                    cv2.putText(normDataFrame, currentPrediction, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 200), 4, cv2.LINE_AA)
                    # yellow = (0,0,200)

                else:
                    currentPrediction = f"{res} - normal"
                    cv2.putText(normDataFrame, currentPrediction, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200, 0), 4, cv2.LINE_AA)

                predictions.append(np.argmax(res))
            fps = f"Edge-AI FPS: {int(1/timeTaken)}"
            cv2.putText(normDataFrame, fps, (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200, 200), 4, cv2.LINE_AA)

            cv2.imshow("preview", normDataFrame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()

