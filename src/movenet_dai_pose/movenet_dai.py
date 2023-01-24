"""
 *
 *  MoveNetDAI ===============================================================
 *
 *  > Description: Pose detection using TensorFlow's movenet model linked with 
 *                 DepthAI and OpenCV for Luxonis OAK-D compatitablity. 
 *                 Modified version from depthai_movenet (see References[1]).
 *
 *  > Author: Alastair Kho
 *  > Year: 2023
 *  > References: 
 *      [1] https://github.com/geaxgx/depthai_movenet
 *      [2] https://doi.org/10.3390/app11010329
 *
 * ===========================================================================
 *
 * """


#  ======  Dependencies  ======  #

import time
from collections import namedtuple
from math import gcd
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

#  ======  Class: MovenetDAI  ======  #
_CROP_REGION = namedtuple('_CROP_REGION',['xmin', 'ymin', 'xmax',  'ymax', 'size']) # All values are in pixel. The region is a square of size 'size' pixels

class MovenetDAI:


    # ---- Class Fields ---- #

    _MODEL_FILE_PATH = "models/movenet_singlepose_thunder_U8_transpose.blob"
    
    # Dictionary that maps from joint names to keypoint indices. 
    KEYPOINT_DICT = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }

    # ---- Constructor ---- #
    def __init__(self, 
                 input_path: str=None,
                 pose_score_threshold: float=0.2,
                 internal_fps: int=12,
                 internal_frame_height: int=640
                 ) -> None:
        
        self.pose_score_threshold = pose_score_threshold
        self.model = str(Path(__file__).resolve().parent / self._MODEL_FILE_PATH)
        self.pd_input_length = 256

        if input_path:
            self.img = cv2.imread(input_path)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]

            self.input_type= "image"
        else:

            self.internal_fps = internal_fps

            # find the scale params, i.e. 700 -> 720, 
            width, self.scale_nd = self._find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
            
            self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
            self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
            self.frame_size = self.img_w    
            print(f"Internal camera image size: {self.img_w} x {self.img_h}")
            
            self.input_type= "rgb"

        box_size = max(self.img_w, self.img_h)
        x_min = (self.img_w - box_size) // 2
        y_min = (self.img_h - box_size) // 2
        self.init_crop_region = _CROP_REGION(x_min, y_min, x_min+box_size, y_min+box_size, box_size)
        self.crop_region = self.init_crop_region

        self.next_crop_region = None # test

        self.device = dai.Device(self._construct_pipeline())

        print("Pipeline Begin")


        if self.input_type == "rgb":

            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_manip_cfg = self.device.getInputQueue(name="manip_cfg")
        else:
            self.movenet_in = self.device.getInputQueue(name="movenet_in")
        
        self.movenet_out = self.device.getOutputQueue(name="movenet_out", maxSize=4, blocking=False)


    # ---- Public Methods ---- #

    def getFrameInference(self) -> tuple[np.ndarray, dai.NNData, _CROP_REGION]:
        frame, inference = self._get_device_out_feed()
        inference = self._post_process_data(inference)
        crop_region = self.crop_region
        self.crop_region = self.next_crop_region

        return frame, inference, crop_region


    def normalize(self, kps: np.ndarray) -> np.ndarray:
        """
         *
         * Relative normalization algorithm. Normalizes pose to centre and removes of unnecessary head keypoints (keypoints 1 -> 4). 
         * Returns: normalized keypoints array (np.ndarray)
         * Adapted from References[2]
         *
         * """
        data = np.copy(kps)
        if np.any(data[self.KEYPOINT_DICT['right_hip'] - 4 ]) and np.any(data[self.KEYPOINT_DICT['left_hip'] - 4 ][1]):

            data[:,0] = data[:,0]
            data[:,1] = data[:,1]

            x_centre = 0.5
            y_centre = 0.5

            #hip_midpoint_y = (data[self.KEYPOINT_DICT['right_hip'] - 4 ][1] + data[self.KEYPOINT_DICT['left_hip'] - 4 ][1]) * 0.5
            hip_midpoint_x = (data[self.KEYPOINT_DICT['right_hip'] - 4 ][0] + data[self.KEYPOINT_DICT['left_hip'] - 4 ][0]) * 0.5
            x_dis = hip_midpoint_x - x_centre
            #y_dis = hip_midpoint_y - y_centre

            for idx, x_y in enumerate(data):
                if x_y[0] != 0 and x_y[1] != 0:
                    data[idx][0] = x_y[0] - x_dis
                    #data[idx][1] = x_y[1] - y_dis

            return data 
        else:
            return np.zeros(data.shape)

    
    def renderNormalized(self, frame: np.ndarray, data = np.ndarray) -> np.ndarray:
        LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
                [10,8],[8,2],[7,8],[1,7],[7,9],
                [10,12],[9,11],[3,5],
                [4,6],[2,1]]
        
        expectedSize = (13,2)
        if data.shape == expectedSize:
            normFrame = np.full((frame.shape[0], frame.shape[0], 3), 25)
            frame = np.concatenate((frame,normFrame),axis=1).astype(np.uint8)
            lines = []
            for idx in range(len(data)):
                x_pos = 0
                y_pos = 0
                if data[idx][0] != 0 and data[idx][1] != 0:
                    x_pos = int(data[idx][0] * self.img_w + frame.shape[1] * 0.5 )
                    y_pos = int(data[idx][1] * self.img_h)
                lines.append([x_pos, y_pos])

            newlines = [np.array([lines[point] for point in line]) for line in LINES_BODY if lines[line[0]][0] != 0 and lines[line[1]][1] != 0]
            cv2.polylines(frame, newlines, False, (180, 255, 90), 2, cv2.LINE_AA)

            for idx, line in enumerate(lines): 
                if idx % 2 == 1:
                    color = (0,255,0) 
                elif idx == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (line[0], line[1]), 4, color , -11)

            return frame
        else:
            raise ValueError(f"Incorrect argument (data) shape. Supplied: {data.shape}, Expected: {expectedSize}")


    def renderKeyPoints(self, frame: np.ndarray, kps: np.ndarray) -> None:
        LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
                [10,8],[8,2],[7,8],[1,7],[7,9],
                [10,12],[9,11],[3,5],
                [4,6],[2,1]]

        expectedSize = (13,2)
        data = np.copy(kps)
        data[:,0] = data[:,0]*self.img_w
        data[:,1] = data[:,1]*self.img_h
        data = data.astype(np.int32)

        cv2.rectangle(frame, (self.crop_region.xmin, self.crop_region.ymin), (self.crop_region.xmax, self.crop_region.ymax), (0,255,255), 2)
        
        if data.shape == expectedSize:
            lines = [np.array([data[point] for point in line]) for line in LINES_BODY if data[line[0]][0] != 0 and data[line[1]][1] != 0]
            cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
            for idx, x_y in enumerate(data):
                # [0 is y, 1 is x]
                if x_y[0] != 0 and x_y[1] != 0:
                    if idx % 2 == 1:
                        color = (0,255,0) 
                    elif idx == 0:
                        color = (0,255,255)
                    else:
                        color = (0,0,255)
                    cv2.circle(frame, (x_y[0], x_y[1]), 4, color , -11)
        else:
            raise ValueError(f"Incorrect argument (data) shape. Supplied: {data.shape}, Expected: {expectedSize}")


    def updateInputPath(self, path: str) -> None:
        if self.input_type == "image":
            self.img = cv2.imread(path)


    # ---- Private Methods ---- #

    def _crop_and_resize(self, frame, crop_region) -> np.ndarray:
        """Crops and resize the image to prepare for the model input."""

        cropped = frame[max(0,crop_region.ymin):min(self.img_h,crop_region.ymax), max(0,crop_region.xmin):min(self.img_w,crop_region.xmax)]
        if crop_region.xmin < 0 or crop_region.xmax >= self.img_w or crop_region.ymin < 0 or crop_region.ymax >= self.img_h:
            # Padding is necessary   
     
            cropped = cv2.copyMakeBorder(cropped, 
                                        max(0,-crop_region.ymin), 
                                        max(0, crop_region.ymax-self.img_h),
                                        max(0,-crop_region.xmin), 
                                        max(0, crop_region.xmax-self.img_w),
                                        cv2.BORDER_CONSTANT)

        cropped = cv2.resize(cropped, (self.pd_input_length, self.pd_input_length), interpolation=cv2.INTER_AREA)
        return cropped
    

    def _torso_visible(self, scores) -> bool:
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((scores[self.KEYPOINT_DICT['left_hip']] > self.pose_score_threshold or
                scores[self.KEYPOINT_DICT['right_hip']] > self.pose_score_threshold) and
                (scores[self.KEYPOINT_DICT['left_shoulder']] > self.pose_score_threshold or
                scores[self.KEYPOINT_DICT['right_shoulder']] > self.pose_score_threshold))


    def _determine_torso_and_body_range(self, keypoints, scores, center_x, center_y) -> list[int, int, int, int ]:
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - keypoints[self.KEYPOINT_DICT[joint]][1])
            dist_x = abs(center_x - keypoints[self.KEYPOINT_DICT[joint]][0])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for i in range(len(self.KEYPOINT_DICT)):
            if scores[i] < self.pose_score_threshold:
                continue
            dist_y = abs(center_y - keypoints[i][1])
            dist_x = abs(center_x - keypoints[i][0])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


    def _determine_crop_region(self, keypoints, scores):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """

        if self._torso_visible(scores):

            center_x = (keypoints[self.KEYPOINT_DICT['left_hip']][0] + keypoints[self.KEYPOINT_DICT['right_hip']][0]) // 2
            center_y = (keypoints[self.KEYPOINT_DICT['left_hip']][1] + keypoints[self.KEYPOINT_DICT['right_hip']][1]) // 2

            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = self._determine_torso_and_body_range(keypoints, scores, center_x, center_y)

            crop_length_half = np.amax([max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])
            tmp = np.array([center_x, self.img_w - center_x, center_y, self.img_h - center_y])

            crop_length_half = int(round(np.amin([crop_length_half, np.amax(tmp)])))

            crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

            if crop_length_half > max(self.img_w, self.img_h) / 2:
                return self.init_crop_region
            else:
                crop_length = crop_length_half * 2
                return _CROP_REGION(crop_corner[0], crop_corner[1], crop_corner[0]+crop_length, crop_corner[1]+crop_length,crop_length)
        else:
            return self.init_crop_region


    def _construct_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)
        
        if self.input_type == "rgb":

            cam_rgb = pipeline.createColorCamera()

            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setIspScale(self.scale_nd[0], self.scale_nd[1])

            cam_rgb.setFps(self.internal_fps)
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

            cam_rgb.setVideoSize(self.img_w, self.img_h)
            cam_rgb.setPreviewSize(self.img_w, self.img_h)

            # Note: cam_out can be blocked in future to prevent cross device sharing RGB frames.
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")

            cam_rgb.video.link(cam_out.input)

            manip = pipeline.createImageManip()
            manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
            manip.setWaitForConfigInput(True)
            manip.inputImage.setQueueSize(1)
            manip.inputImage.setBlocking(False)            

            
            manip_cfg = pipeline.createXLinkIn()
            manip_cfg.setStreamName("manip_cfg")

            cam_rgb.preview.link(manip.inputImage)
            manip_cfg.out.link(manip.inputConfig)
        
        print("Attaching Model Blob")

        movenet_nn = pipeline.createNeuralNetwork()
        movenet_nn.setBlobPath(str(Path(self.model).resolve().absolute()))
        
        if self.input_type == "rgb":

            manip.out.link(movenet_nn.input)
        else:
            pd_in = pipeline.createXLinkIn()
            pd_in.setStreamName("movenet_in")
            pd_in.out.link(movenet_nn.input)

        movenet_out = pipeline.createXLinkOut()
        movenet_out.setStreamName("movenet_out")
        movenet_nn.out.link(movenet_out.input)

        print("Pipeline Complete")

        return pipeline


    def _get_device_out_feed(self) -> tuple[np.ndarray, dai.NNData]:
        
        if self.input_type == "rgb":
            cfg = dai.ImageManipConfig()
            points = [
                [self.crop_region.xmin, self.crop_region.ymin],
                [self.crop_region.xmax-1, self.crop_region.ymin],
                [self.crop_region.xmax-1, self.crop_region.ymax-1],
                [self.crop_region.xmin, self.crop_region.ymax-1]]
            point2fList = []
            for p in points:
                pt = dai.Point2f()
                pt.x, pt.y = p[0], p[1]
                point2fList.append(pt)
            cfg.setWarpTransformFourPoints(point2fList, False)
            cfg.setResize(self.pd_input_length, self.pd_input_length)
            cfg.setFrameType(dai.ImgFrame.Type.RGB888p)
            self.q_manip_cfg.send(cfg)

            in_video = self.q_video.get()
            frame = in_video.getCvFrame()

        elif self.input_type == "image":
            frame = self.img.copy()
                        # Cropping of the video frame
            cropped = self._crop_and_resize(frame, self.crop_region)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).transpose(2,0,1)
            
            frame_nn = dai.ImgFrame()
            frame_nn.setTimestamp(time.monotonic())
            frame_nn.setWidth(self.pd_input_length)
            frame_nn.setHeight(self.pd_input_length)
            frame_nn.setData(cropped)
            self.movenet_in.send(frame_nn)


        inference = self.movenet_out.get()

        return frame, inference


    def _post_process_data(self, data: dai.NNData) -> np.ndarray:
        kps = np.array(data.getLayerFp16('Identity')).reshape(-1,3)

        confidence_scores = kps[:,2]
        keypoints_norm=kps[:,[1,0]]
        keypoints = (np.array([self.crop_region.xmin, self.crop_region.ymin]) + keypoints_norm * self.crop_region.size)

        self.next_crop_region = self._determine_crop_region(keypoints.astype(np.int32), confidence_scores)
        
        out_keypoints = np.concatenate((keypoints[0:1], keypoints[5:17])) #original: keypoints_norm = np.append(data[0,[1,0]] + data[5:17,[1,0]])
        out_confidence_scores = np.concatenate((confidence_scores[0:1], confidence_scores[5:17])) #original: keypoints_norm = np.append(data[0,[1,0]] + data[5:17,[1,0]])

        for idx, score in enumerate(out_confidence_scores):
            if score < self.pose_score_threshold:
                out_keypoints[idx][0] = out_keypoints[idx][1] = 0.0
        
        out_keypoints[:,0] /= self.img_w
        out_keypoints[:,1] /= self.img_h
        return out_keypoints


    def _find_isp_scale_params(self, size, is_height: bool=True) -> tuple[int, set[int, int]]:
        """
         *
         * Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
         * This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
         * is_height : boolean that indicates if the value is the height or the width of the image
         * Returns: valid size, (numerator, denominator)
         *
         * Algorithm from References[1]
         *
         * """

        # We want size >= 288
        if size < 288:
            size = 288
        
        # We are looking for the list on integers that are divisible by 16 and
        # that can be written like n/d where n <= 16 and d <= 63
        if is_height:
            reference = 1080 
            other = 1920
        else:
            reference = 1920 
            other = 1080
        size_candidates = {}
        for s in range(288,reference,16):
            f = gcd(reference, s)
            n = s//f
            d = reference//f
            if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
                size_candidates[s] = (n, d)
                
        # What is the candidate size closer to 'size' ?
        min_dist = -1
        for s in size_candidates:
            dist = abs(size - s)
            if min_dist == -1:
                min_dist = dist
                candidate = s
            else:
                if dist > min_dist: break
                candidate = s
                min_dist = dist
        return candidate, size_candidates[candidate]
