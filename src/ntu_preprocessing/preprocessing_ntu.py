import os
import numpy as np
import glob
import math

# Replace 'ntu_npy' with the path to the directory containing .npy files
directory_path = 'ntu_npy'

# Find all .npy files in the directory
npy_files = glob.glob(os.path.join(directory_path, '*.npy'))


def findPrimarySample(dataSample: dict) -> int:
    numPersons = max(dataSample["nbodys"])
    x_centre = 1920/2
    y_centre = 1080/2 #kinectv2 dim = 1920x1080
    
    skelID_with_least_dist = 0

    if numPersons > 1:
        skelID_with_least_dist = None
        least_dist = np.inf
        for i in range(numPersons):
            skelID = f"rgb_body{i}"
            average_of_x = np.mean(dataSample[skelID][:, :, 0])
            average_of_y = np.mean(dataSample[skelID][:, :, 1])
            
            x_diff = average_of_x - x_centre
            y_diff = average_of_y - y_centre
            dist = math.sqrt(x_diff*x_diff + y_diff*y_diff)
            print(f"in Sample {dataSample['file_name']} skel{i}, dist: {dist}")
            if dist <= least_dist:
                least_dist = dist
                skelID_with_least_dist = i

        print(f"In Sample {dataSample['file_name']}; NumPersons: {numPersons}; primaryID: rgb_body{skelID_with_least_dist}; dist from centre: {least_dist};")
    return skelID_with_least_dist


def containsNANandZeros(dataSample: dict) -> bool:
    # Check for multiple persons
    numPersons = max(dataSample["nbodys"])
    for i in range(numPersons):
        skelID = f"rgb_body{i}"
        if np.any(np.isnan(dataSample[skelID])):
            print(f"In Sample {dataSample['file_name']};rgb_body{i}, NAN WAS FOUND")
            return True
        if np.any((dataSample[skelID][:, :, 0] == 0) & (dataSample[skelID][:, :, 1] == 0)):
            print(f"In Sample {dataSample['file_name']};rgb_body{i}, ZEROS WAS FOUND")
            return True
    return False


def normalizeSkeletonData(skelData: np.ndarray) -> np.ndarray:
    KEYPOINT_DICT = {
    'nose': 0,
    'left_shoulder': 1,
    'right_shoulder': 2,
    'left_elbow': 3,
    'right_elbow': 4,
    'left_wrist': 5,
    'right_wrist': 6,
    'left_hip': 7,
    'right_hip': 8,
    'left_knee': 9,
    'right_knee': 10,
    'left_ankle': 11,
    'right_ankle': 12
    }
        
    NTU_TO_MOVENET = { # NTU followed different skeleton labeling, needed conversion to MoveNet (OpenPose labelling)
    3: 0,
    8: 1,
    4: 2,
    9: 3,
    5: 4,
    10: 5,
    6: 6,
    16: 7,
    12: 8,
    17: 9,
    13: 10,
    18: 11,
    14: 12
    }

    filteredSequence = np.zeros(shape=(skelData.shape[0], 13, 2))

    for seq_idx, sequences in enumerate(skelData):
        filteredKeyPoints = np.zeros(shape=(13,2))
        for kp_id, keypoint in enumerate(sequences):
            if kp_id in NTU_TO_MOVENET.keys(): # if not filtered out
                moveNetIdx = NTU_TO_MOVENET[kp_id]
                filteredKeyPoints[moveNetIdx] = keypoint

        filteredSequence[seq_idx] = filteredKeyPoints
        x_centre = 1080/2
        y_centre = 1080/2

        hip_midpoint_y = (filteredKeyPoints[KEYPOINT_DICT['right_hip']][1] + filteredKeyPoints[KEYPOINT_DICT['left_hip']][1]) * 0.5
        hip_midpoint_x = (filteredKeyPoints[KEYPOINT_DICT['right_hip']][0] + filteredKeyPoints[KEYPOINT_DICT['left_hip']][0]) * 0.5
        x_dis = hip_midpoint_x - x_centre
        y_dis = hip_midpoint_y - y_centre

        for idx, x_y in enumerate(filteredKeyPoints):
            filteredKeyPoints[idx][0] = x_y[0] - x_dis
            filteredKeyPoints[idx][1] = x_y[1] - y_dis

        filteredSequence[seq_idx] = filteredKeyPoints

    filteredSequence[:, :, 1] = (1080 - filteredSequence[:, :, 1]) / 1080
    filteredSequence[:, :, 0] /= 1080


    return filteredSequence

def find_max_length(directory):
    """
    Function to determine the length of the longest array among all .npy files in the directory.
    """
    max_length = 0
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            data = np.load(filepath)
            max_length = max(max_length, data.shape[0])  # Assuming the first dimension is the one we're interested in
    return max_length

def pad_and_save_arrays(directory, max_length):
    """
    Function to pad each array in each .npy file in the directory to the given max_length.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            data = np.load(filepath)
            pad_length = max_length - data.shape[0]
            
            if pad_length > 0:
                # Pad with the first element, and adjust padding for multi-dimensional arrays
                pad_shape = (pad_length,) + data.shape[1:]
                pad_values = np.full(pad_shape, data[0])
                padded_data = np.concatenate([pad_values, data], axis=0)
                np.save(filepath, padded_data)
                print(f"New padding {filename}, old: {data.shape[0]} new: {padded_data.shape[0]}")

            else:
                print(f"No padding needed for {filename}, {data.shape[0]}")

# data preprocessing
"""def main():
    os.makedirs("preprocessed_ntu_npy")
    for npy_file in npy_files:
        try:
            dataSample = np.load(npy_file, allow_pickle=True).item()
            if containsNANandZeros(dataSample):
                print(f"Skipping sample: {dataSample['file_name']}\n")

            else:
                primarySampleID = findPrimarySample(dataSample)
                skelID = f"rgb_body{primarySampleID}"
                output_file = os.path.join("preprocessed_ntu_npy", f"{dataSample['file_name']}.npy")
                outData = normalizeSkeletonData(dataSample[skelID])
                np.save(output_file, outData)
                print(f"Processed sample: {dataSample['file_name']}\n")

        except Exception as e:
            print(f"Error loading {npy_file}: {e}")


if __name__ == "__main__":
    main()"""

if __name__ == "__main__":
    directory = "preprocessed_ntu_npy_lstm"
    
    if not os.path.exists(directory):
        print("Directory does not exist. Please enter a valid path.")
    else:
        max_length = 300
        print(max_length)
        pad_and_save_arrays(directory, max_length)
        print("Padding complete!")


# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------
# Temp testing code

"""
# Iterate through each .npy file
def getprimary():
    for npy_file in npy_files:
        try:
            # Load the .npy file using numpy
            data = np.load(npy_file, allow_pickle=True).item()
            # Check for multiple persons
            numPersons = max(data["nbodys"])

            x_centre = 1920/2
            y_centre = 1080/2 #kinectv2 dim = 1920x1080

            if numPersons > 1:
                skelID_with_least_dist = None
                least_dist = np.inf
                for i in range(numPersons):
                    skelID = f"rgb_body{i}"
                    average_of_x = np.mean(data[skelID][:, :, 0])
                    average_of_y = np.mean(data[skelID][:, :, 1])
                    
                    x_diff = average_of_x - x_centre
                    y_diff = average_of_y - y_centre
                    dist = math.sqrt(x_diff*x_diff + y_diff*y_diff)
                    print(f"in Sample {data['file_name']} skel{i}, dist: {dist}")
                    if dist <= least_dist:
                        least_dist = dist
                        skelID_with_least_dist = i

                heuristic_skelID, variance_list = identify_main_subject_2(data)
                print(f"In Sample {data['file_name']}; NumPersons: {numPersons}; both_algorithms_match: {True if (heuristic_skelID == skelID_with_least_dist) else (False)}; primaryID: rgb_body{skelID_with_least_dist}; heuristic primary: {heuristic_skelID}; heuristic_variance_max: {variance_list} ")

        except Exception as e:
            print(f"Error loading {npy_file}: {e}")


def filterNANandZeros():
    for npy_file in npy_files:
        try:
            # Load the .npy file using numpy
            data = np.load(npy_file, allow_pickle=True).item()
            # Check for multiple persons
            numPersons = max(data["nbodys"])
            for i in range(numPersons):
                skelID = f"rgb_body{i}"
                if np.any(np.isnan(data[skelID])):
                    print(f"In Sample {data['file_name']};rgb_body{i}, NAN WAS FOUND")
                if np.any((data[skelID][:, :, 0] == 0) & (data[skelID][:, :, 1] == 0)):
                    print(f"In Sample {data['file_name']};rgb_body{i}, ZEROS WAS FOUND")
 

        except Exception as e:
            print(f"Error loading {npy_file}: {e}")


def calculate_variance(data):
    # Calculate variance for each joint's X, Y, and Z values
    variances = np.var(data, axis=0)
    return variances

def identify_main_subject_2(data): # get highest variance
    main_subject_index = None
    highest_variance_sum = -np.inf # neg inf
    variance_list = []
    all_list = []

    for npy_file in npy_files:
        try:
            # Check for multiple persons
            numPersons = max(data["nbodys"])
            for i in range(numPersons):
                skelID = f"rgb_body{i}"
                variances = calculate_variance(data[skelID][0])
                variance_sum = np.sum(variances)
                all_list.append(variance_sum)
                if variance_sum > highest_variance_sum:
                    highest_variance_sum = variance_sum
                    main_subject_index = i

        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            
    variance_list.append(max(all_list))
    variance_list.append(min(all_list))

    return main_subject_index, variance_list
"""
