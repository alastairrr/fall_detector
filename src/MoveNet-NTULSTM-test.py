
import movenet_dai_pose as mnet
import cv2
import numpy as np
import time
from fall_detection import LSTMFallDetector

# Testing with OAK-D Movenet


#import mediapipe as mp
#import imutils

input_path="../data/2763_08252019.mp4"
#input_path="../data/S001C001P003R002A043_rgb.mp4"
movenet = mnet.MovenetDAI(input_path=input_path)

model = LSTMFallDetector()
model.getModelSummary()
model.modelLoadWeights("./fall_detection/model_weights/ntu_weights.h5")

def reshape_keypoints(kps):
    #print("pose with norms:\n", pose)
    reshape = kps.reshape(1)
    #print("flattened:\n", flattened)

    #print("\n")
    return reshape

sequence = []
predictions = []
firstRun = True

while True:
    startTime = time.time()

    curr_frame, data, crop_region = movenet.getFrameInference()
    timeTaken = time.time() - startTime

    if curr_frame is None: break
    if curr_frame is not None:
        movenet.renderKeyPoints(curr_frame, data)
        normalized = movenet.normalize(data)
        normDataFrame = movenet.renderNormalized(curr_frame, movenet.normalize(data))
        normDataFrame = cv2.resize(normDataFrame, (1564, 552)) 

        reshaped = normalized

        sequence.append(reshaped)
        if firstRun == True:
            for i in range(300):
                sequence.append(reshaped)
            firstRun = False
        sequence = sequence[-300:]
        if len(sequence) == 300:
            res = model.modelPredict(np.array(sequence).reshape(-1, 300, 26))
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7, 
    model_complexity=1, enable_segmentation=True) as pose:"""

"""from skimage.metrics import structural_similarity
import cv2
import numpy as np
init_frame = None

while True:
    new_init_frame, _, _ = movenet.getFrameInference()
    cv2.imshow("preview", new_init_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        init_frame = new_init_frame
        break



while True:
    curr_frame, data, crop_region = movenet.getFrameInference()
    if curr_frame is None: break
    if curr_frame is not None:


        diff = cv2.absdiff(curr_frame, init_frame)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        th = 4
        imask =  mask<th

        curr_frame = np.zeros_like(init_frame, np.uint8)
        curr_frame[imask] = init_frame[imask]
        movenet.renderKeyPoints(curr_frame, data)

        cv2.imshow("preview", curr_frame)
    if cv2.waitKey(1) == ord('q'):
        break"""



"""mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7, 
    model_complexity=1, enable_segmentation=True) as pose:
    while True:
        frame, data, _ = movenet.getFrameInference()
        if frame is None: break
        if frame is not None:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Draw pose segmentation.
            BG_COLOR = (25, 25, 25) # gray
        
            # apply mask on image with gray

            if results.segmentation_mask is not None:
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.array(frame)
                frame[:] = BG_COLOR
                frame = np.where(condition, frame, bg_image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                movenet.renderKeyPoints(frame, data)

            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            normDataFrame = movenet.renderNormalized(frame, movenet.normalize(data))
            normDataFrame = cv2.resize(normDataFrame, (1360, 480)) 
            cv2.imshow("preview", normDataFrame)
        if cv2.waitKey(1) == ord('q'):
            break"""





"""

import movenet_dai_pose as mnet
import cv2
import numpy as np
import mediapipe as mp

movenet = mnet.MovenetDAI()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7, 
    model_complexity=1, enable_segmentation=True) as pose:

    while True:
        frame, data = movenet.getFrameInference()
        if frame is None: break
        if frame is not None:


            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Draw pose segmentation.
            BG_COLOR = (25, 25, 25) # gray
        
            # apply mask on image with gray

            if results.segmentation_mask is not None:
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.array(frame)
                frame[:] = BG_COLOR
                frame = np.where(condition, frame, bg_image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                movenet.renderKeyPoints(frame, data)

            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("preview", frame)


        if cv2.waitKey(1) == ord('q'):
            break


########################################3# Canny test
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
# define range of red color in HSV   
lower_red = np.array([30, 150, 50])  
upper_red = np.array([255, 255, 180])  

# create a red HSV colour boundary and    
# threshold HSV image   
mask = cv2.inRange(hsv, lower_red, upper_red)  

# Bitwise-AND mask and original image   
res = cv2.bitwise_and(frame, frame, mask=mask)  

# Display an original image   
#cv2.imshow('Original', frame)  #uncomment to see the original window frame

# discovers edges in the input image image and   
# marks them in the output map edges   
edges = cv2.Canny(np.mean(frame, axis=2).astype(np.uint8), 50, 100)
print(edges)

out = np.bitwise_or(frame, edges[:,:,np.newaxis])
# Canny test
"""