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
 *
 * ===========================================================================
 *
 * """


#  ======  Dependencies  ======  #

from collections import namedtuple
from math import gcd
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


#  ======  Class: MovenetDAI  ======  #

class MovenetDAI:


    # ---- Class Fields ---- #

    _MODEL_FILE_PATH = "models/movenet_singlepose_thunder_U8_transpose.blob"
    _CROP_REGION = namedtuple('CropRegion',['xmin', 'ymin', 'xmax',  'ymax', 'size']) # All values are in pixel. The region is a square of size 'size' pixels


    # ---- Constructor ---- #
    def __init__(self, 
                 pose_score_threshold=0.3,
                 internal_fps=15,
                 internal_frame_height=640
                 ) -> None:

        self.internal_fps = internal_fps
        self.pose_score_threshold = pose_score_threshold
        self.model = str(Path(__file__).resolve().parent / self._MODEL_FILE_PATH)

        # find the scale params, i.e. 700 -> 720, 
        width, self.scale_nd = self._find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
        
        self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
        self.frame_size = self.img_w    

        self.pd_input_length = 256

        box_size = max(self.img_w, self.img_h)
        x_min = (self.img_w - box_size) // 2
        y_min = (self.img_h - box_size) // 2

        self.init_crop_region = self._CROP_REGION(x_min, y_min, x_min+box_size, y_min+box_size, box_size)
        self.crop_region = self.init_crop_region

        self.device = dai.Device(self._construct_pipeline())

        print("Pipeline Begin")

        self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_manip_cfg = self.device.getInputQueue(name="manip_cfg")
        self.movenet_out = self.device.getOutputQueue(name="movenet_out", maxSize=4, blocking=False)


    # ---- Public Methods ---- #

    def getFrameInference(self) -> tuple[np.ndarray, dai.NNData]:
        frame, inference = self._get_device_out_feed()
        inference = self._post_process_data(inference)

        return frame, inference


    def normalize(self, inference):
        # TODO: must ignore if kp is nan

        # NOTE: centres data, gets rid of uncessary kps
        keypoints_norm=inference[:,[1,0]]

        pass


    def renderKeyPoints(self, frame, inference) -> None:
        keypoints_norm=inference[:,[1,0]]
        keypoints = np.array([self.crop_region.xmin, self.crop_region.ymin]) + keypoints_norm * self.crop_region.size
        """
        for idx,x_y in enumerate(keypoints):
            if x_y[0] and x_y[1] != np.nan:
                if idx % 2 == 1:
                    color = (0,255,0) 
                elif idx == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

        """       
        for idx,x_y in enumerate(keypoints):
            if not (np.isnan(x_y[0]) and np.isnan(x_y[1])):
                if idx % 2 == 1:
                    color = (0,255,0) 
                elif idx == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (int(x_y[0]), int(x_y[1])), 4, color, -11)


    # ---- Private Methods ---- #

    def _construct_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

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
        manip.setMaxOutputFrameSize(self.img_h*self.img_w*3)
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
        manip.out.link(movenet_nn.input)

        movenet_out = pipeline.createXLinkOut()
        movenet_out.setStreamName("movenet_out")
        movenet_nn.out.link(movenet_out.input)

        print("Pipeline Complete")

        return pipeline


    def _get_device_out_feed(self) -> tuple[np.ndarray, dai.NNData]:
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
        inference = self.movenet_out.get()

        return frame, inference


    def _post_process_data(self, data) -> np.ndarray:
        keypoints = np.array(data.getLayerFp16('Identity')).reshape(-1,3)
        confidence_scores = keypoints[:,2]

        for idx, score in enumerate(confidence_scores):
            if score < self.pose_score_threshold:
                keypoints[idx][0] = keypoints[idx][1] = np.nan
                
        return keypoints


    def _find_isp_scale_params(self, size, is_height=True) -> tuple[int, set[int, int]]:
        """
        Algorithm from References[1]

        Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
        This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
        is_height : boolean that indicates if the value is the height or the width of the image
        Returns: valid size, (numerator, denominator)
        """
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
