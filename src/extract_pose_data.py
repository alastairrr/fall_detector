import movenet_dai_pose as mnet
import cv2
import os
import numpy as np

DATA_PATH = os.path.join('../data/extracted_data')


actions = np.array(['normal', 'fall'])


def extract_data(verbose=False):
    print("Datagen Detection Begin...")

    file_path = "../data/dataset/fall/fall-01-cam0-rgb/fall-01-cam0-rgb-001.png"
    movenet = mnet.MovenetDAI(input_path=file_path)

    for i in range(1, 31):
        numFrames = 1
        with open(f"../data/dataset/fall/fall-{i:02d}-cam0-rgb/fall-{i:02d}-data.csv") as file:
            for line in file:
                if line.strip() != "":
                    numFrames += 1
        try:
            os.makedirs(os.path.join(DATA_PATH, "fall", str(i)))
        except FileExistsError as e:
            print(e)

        file_path = f"../data/dataset/fall/fall-{i:02d}-cam0-rgb/fall-{i:02d}-cam0-rgb-001.png"

        for j in range(1, numFrames):
            print(file_path)

            movenet.updateInputPath(file_path)
            frame, data, _ = movenet.getFrameInference()
            normalized = movenet.normalize(data)

            if verbose == True:

                movenet.renderKeyPoints(frame, data)
                
                normDataFrame = movenet.renderNormalized(frame, normalized)
                normDataFrame = cv2.resize(normDataFrame, (1360, 480)) 
                cv2.imshow("preview", normDataFrame)
                cv2.waitKey(9)

            flattend = flatten_keypoints(data, normalized)

            npy_path = os.path.join(DATA_PATH, "fall", str(i), str(j))
            np.save(npy_path, flattend)
            file_path = f"../data/dataset/fall/fall-{i:02d}-cam0-rgb/fall-{i:02d}-cam0-rgb-{j:03d}.png"


    file_path = "../data/dataset/not_fall/adl-01-cam0-rgb/adl-01-cam0-rgb-001.png"

    for i in range(1, 41):
        
        numFrames = 1
        with open(f"../data/dataset/not_fall/adl-{i:02d}-cam0-rgb/adl-{i:02d}-data.csv") as file:
            for line in file:
                if line.strip() != "":
                    numFrames += 1

        try:
            os.makedirs(os.path.join(DATA_PATH, "not_fall", str(i)))
        except FileExistsError as e:
            print(e)
        file_path = f"../data/dataset/not_fall/adl-{i:02d}-cam0-rgb/adl-{i:02d}-cam0-rgb-001.png"

        for j in range(1, numFrames):
            print(file_path)

            movenet.updateInputPath(file_path)
            frame, data, _ = movenet.getFrameInference()
            normalized = movenet.normalize(data)

            if verbose == True:

                movenet.renderKeyPoints(frame, data)

                normDataFrame = movenet.renderNormalized(frame, normalized)
                normDataFrame = cv2.resize(normDataFrame, (1360, 480)) 
                cv2.imshow("preview", normDataFrame)
                cv2.waitKey(9)

            flattend = flatten_keypoints(data, normalized)

            npy_path = os.path.join(DATA_PATH, "not_fall", str(i), str(j))            
            np.save(npy_path, flattend)
            file_path = f"../data/dataset/not_fall/adl-{i:02d}-cam0-rgb/adl-{i:02d}-cam0-rgb-{j:03d}.png"


    cv2.destroyAllWindows()


def flatten_keypoints(kps, norm_kps):
    pose = np.concatenate((kps, norm_kps), axis=1)
    print("pose with norms:\n", pose)
    flattened = pose.flatten()
    print("flattened:\n", flattened)

    print("\n")
    return flattened


if __name__ == "__main__":
    extract_data(verbose=True)

