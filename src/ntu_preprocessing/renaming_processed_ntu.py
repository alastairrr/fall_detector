import os
import shutil

# Define the paths
source_directory = "preprocessed_ntu_npy_padded_lstm"
out_dir = "split_ntu_npy_lstm"
fall_directory = os.path.join(out_dir, 'fall')
not_fall_directory = os.path.join(out_dir, 'not_fall')

# Create the sub-directories if they don't exist
if not os.path.exists(fall_directory):
    os.makedirs(fall_directory)

if not os.path.exists(not_fall_directory):
    os.makedirs(not_fall_directory)

# Loop through each file and move them based on the condition
for filename in os.listdir(source_directory):
    if filename.endswith('A043.npy'):
        shutil.move(os.path.join(source_directory, filename), fall_directory)
    else:
        shutil.move(os.path.join(source_directory, filename), not_fall_directory)


def rename_files_in_directory(path, start=0):
    """Rename all files in a directory to numbers, beginning from start."""
    files = sorted(os.listdir(path))
    for i, file in enumerate(files, start=start):
        old_file_path = os.path.join(path, file)
        # Extracting file extension to keep it in the renamed file
        file_extension = os.path.splitext(file)[1]
        new_file_path = os.path.join(path, str(i) + file_extension)
        os.rename(old_file_path, new_file_path)

rename_files_in_directory(fall_directory)
rename_files_in_directory(not_fall_directory)

