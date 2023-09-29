# Fall Detection From MoveNet Pose Estimation with OAK-D Hardware
LSTM based fall detection model with MoveNet pose estimation via OAK-D hardware and DepthAI.

Latest Update: Trained with NTU-RGB Action Recognition Dataset.

## Demo
![Demo](doc/img/lstmfall_sample_2763_08252019.gif)
Normali
Note: This demo was trained with NTU-RGB Action Recognition Dataset in attempt to increase robustness. The current trained weights for demo.py uses src\fall_detection\model_weights\ntu_weights.h5

Dataset: NTU-RGB Action Recognition (~ 55k samples, ~1000 of which are falls).

## Cross Validation Results
* Average accuracy score: 0.9953637242317199
* Average Precision: 0.887873831329372
* Average Recall: 0.8373728567276955
* Average F1-score: 0.8579635847009552
* Average AUC score: 0.9177247300260225

CrossVal Params:
* number of splits: 5
* split type: stratKfold
* min fall confidence - 0.5
* random_state = 42

Model Params:
* batch_size: 32
* learning_rate: 0.0001
* loss: binary cross entropy with class weights.

## Future Work
Further validation may be needed as dataset classes were heavily imbalanced. Future work may involve synthetic data by altering angles of joints by a few degrees or mirroring the skeleton along the vertical axis (left hand becomes right hand joint). Introducing samples of squats from NTU-RGB 120 in future will also ensure that model is able to distinguish between squats and falls better. Model also needs to know what an empty scene looks like.

## Normalization
Initial filtering involved removing skeletons with Nan or zeroes and selecting one participant where multiple existed(by selecting skeletons with higher variance).
Normalisation algorithm involves moving the HPE skeleton to the centre of the frame to reduce bias inflicted by horizontal and vertical movements when training the LSTM network.

## Install
Install the python packages DepthAI, Opencv with the following command:
```
python3 -m pip install -r requirements.txt
```

## Retrieve Dataset
NTU Data is From https://github.com/shahroudy/NTURGB-D
Testing Data is From https://doi.org/10.1016/s0140-6736(12)61263-x

# References
* [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) - Ultra fast and accurate pose detection model, from 'TensorFlow'.
* [depthai_movenet](https://github.com/geaxgx/depthai_movenet) - MoveNet Single Pose tracking on DepthAI, from 'geaxgx'.
* [A Framework for Fall Detection Based on OpenPose Skeleton and LSTM/GRU Models](https://doi.org/10.3390/app11010329) - Lin, C.B., et al
* [NTU RGB Action Recognition](https://github.com/shahroudy/NTURGB-D) -  Shahroudy, A., et al
* [Video capture of the circumstances of falls in elderly people residing in long-term care: an observational study](https://doi.org/10.1016/s0140-6736(12)61263-x) - Robinovitch, S.N., et al
