import rclpy
from rclpy.node import Node
# qos_profile_sensor_data is a default QoS profile for sensor data
# It is defined in rclpy as a combination of the following settings:
# - history: KEEP_LAST
# - depth: 7
# - reliability: BEST_EFFORT
# - durability: VOLATILE
from rclpy.qos import qos_profile_sensor_data

import inspect
import yaml
import sys
from getpass import getuser
import numpy as np
import cv2
import math
from copy import copy

from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import UInt8
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue

import platform


def try_get(fn, default):
    try:
        val = fn().value
        if val is None:
            return default
        else:
            return val
    except:
        return default



class LandingPublisher(Node):

    def __init__(self):
        super().__init__('landing_publisher', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        # self.get_logger().info(f"Line number: {inspect.getframeinfo(inspect.currentframe()).lineno}")
        self.coordinates_pub = self.create_publisher(Vector3, 'camera/landing_position', 10)
        self.detection_pub = self.create_publisher(Image, "camera/detected_markers", 10)
        self.bridge = CvBridge()
        self.cam_info = None
        self.fov_x_rad = None
        self.fov_y_rad = None
        self.create_subscription(UInt8, 'landing_object_detector/landing_marker_id', self.landing_marker_id_callback, 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticStatus, 'diagnostics', 10)

        #####################################################
        #                       Params                      #
        #####################################################

        self.landing_marker_id = try_get(lambda: self.get_parameter("landing_marker_id"), 688)
        self.target_landing_marker_id = try_get(lambda: self.get_parameter("target_landing_marker_id"), None)
        self.robot_is_simulated = try_get(lambda: self.get_parameter("robot_is_simulated"), False)
        self.camera_port = try_get(lambda: self.get_parameter("camera_port"), "/dev/video0")
        self.calibration_config_path = try_get(lambda: self.get_parameter("calibration_config_path"), None)
        self.res_width = try_get(lambda: self.get_parameter("res_width"), None)
        self.res_height = try_get(lambda: self.get_parameter("res_height"), None)
        self.camera_params_custom_autosetup_period_ms = try_get(lambda: self.get_parameter("camera_params_custom_autosetup_period_ms"), None)

                    # self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    # exposure_value =  100 # Adjust this value based on your camera's range
                    # self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

        #####################################################
        #                 Sim or Real setup                 #
        #####################################################
        self.camera_pub = None
        if not self.robot_is_simulated:
            #####################################################
            #                     Real setup                    #
            #####################################################
            self.cam = cv2.VideoCapture(self.camera_port)
        
            if self.calibration_config_path is not None:
                with open(self.calibration_config_path, 'r') as file:
                    calib_config = yaml.safe_load(file)

                    # Extract the calibration parameters
                    D = [item for sublist in calib_config['D'] for item in sublist]
                    K = [item for sublist in calib_config['K'] for item in sublist]

                    width = None
                    height = None

                    if self.res_width is not None and self.res_height is not None:
                        width = self.res_width
                        height = self.res_height
                    else:
                        height = calib_config.get('height', 480)  # Default to 480 if not provided
                        width = calib_config.get('width', 640)    # Default to 640 if not provided

                    # Create the CameraInfo message
                    camera_info = CameraInfo()
                    camera_info.d = D
                    camera_info.k = K
                    camera_info.height = height
                    camera_info.width = width
                    camera_info.distortion_model = 'plumb_bob'  # Example distortion model, set accordingly
                    # camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Identity matrix for R
                    # camera_info.P = K + [0.0, 0.0, 0.0, 0.0]  # Assuming P is the same as K for monocular camera
                    camera_info.binning_x = 0
                    camera_info.binning_y = 0
                    camera_info.roi.x_offset = 0
                    camera_info.roi.y_offset = 0
                    camera_info.roi.height = 0
                    camera_info.roi.width = 0
                    camera_info.roi.do_rectify = False

                    self.cam_info = camera_info
                    fx = self.cam_info.k[0]
                    fy = self.cam_info.k[4]
                    width = self.cam_info.width
                    height = self.cam_info.height
                    
                    self.fov_x_rad = 2 * math.atan2(width, (2 * fx))
                    self.fov_y_rad = 2 * math.atan2(height, (2 * fy))

                    self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_info.width)
                    self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_info.height)

                    # self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    # # Set a specific focus value
                    # desired_focus_value = 0  # Example value; adjust based on your camera
                    # self.cam.set(cv2.CAP_PROP_FOCUS, desired_focus_value)
                    # brightness_value = 0  # Adjust this value based on your camera's range
                    # self.cam.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
                    # self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    # exposure_value =  100 # Adjust this value based on your camera's range
                    # self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
            else:
                raise Exception("No calibration config path provided")

            self.str_average = ""
            self.inimg = None

            self.light_is_bright = False
            timer_period = 0.1  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
            if self.camera_params_custom_autosetup_period_ms is not None:
                self.autoparams_timer = self.create_timer(float(self.camera_params_custom_autosetup_period_ms / 1000.0), self.params_autosetup_cb)
                self.err_i = 0
                self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            # self.light_timer = self.create_timer(10, self.light_timer)
            self.camera_pub = self.create_publisher(Image, "camera/image", 10)
            self.camera_info_pub = self.create_publisher(CameraInfo, "camera/camera_info", 10)
        else:
            #####################################################
            #                     Sim setup                     #
            #####################################################
            self.cam_sub = self.create_subscription(Image, 'camera', self.aruco_callback, 10)
            self.cam_info_sub = self.create_subscription(CameraInfo, 'camera/camera_info', self.cam_info_cb, 10)
            # self.camera_pub = self.create_publisher(Image, "camera/image", 10)


    def params_autosetup_cb(self):
        # bridge = args['cv_bridge']
        # dyn_client = args['dyn_client']
        # cv_image = bridge.imgmsg_to_cv2(image, desired_encoding = "bgr8")
        cv_image = self.inimg
        
        (rows, cols, channels) = cv_image.shape
        if (channels == 3):
            brightness_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)[:,:,2]
        else:
            brightness_image = cv_image

        #crop_size = 10
        #brightness_image = brightness_image[rows-crop_size:rows+crop_size, cols-crop_size:cols+crop_size]

        #(rows, cols) = brightness_image.shape
        
        hist = cv2.calcHist([brightness_image],[0],None,[5],[0,256])
        
        mean_sample_value = 0
        for i in range(len(hist)):
            mean_sample_value += hist[i]*(i+1)
            
        mean_sample_value /= (rows*cols)
        
        #focus_region = brightness_image[rows/2-10:rows/2+10, cols/2-10:cols/2+10]
        #brightness_value = numpy.mean(focus_region)

        # Middle value MSV is 2.5, range is 0-5
        # Note: You may need to retune the PI gains if you change this
        desired_msv = 2.5
        # desired_msv = 2.5
        # Gains
        # k_p = 0.05
        k_p = 2
        k_i = 0.2
        # Maximum integral value
        max_i = 5
        err_p = desired_msv - mean_sample_value
        self.err_i += err_p
        if abs(self.err_i) > max_i:
            self.err_i = np.sign(self.err_i)*max_i
        
        # Don't change exposure if we're close enough. Changing too often slows
        # down the data rate of the camera.
        if abs(err_p) > 0.25:
            # set_exposure(dyn_client, get_exposure(dyn_client)+k_p*err_p+k_i*self.err_i)
            exposure_value = self.cam.get(cv2.CAP_PROP_EXPOSURE) # Adjust this value based on your camera's range
            new_value = exposure_value + k_p*err_p + k_i*self.err_i
            # self.get_logger().info(f"current exposure = {exposure_value}")
            # self.get_logger().info(f"mean_sample_value = {mean_sample_value}")
            # self.get_logger().info(f"err_p = {err_p}")
            # self.get_logger().info(f"new exposure = {new_value}")
            self.cam.set(cv2.CAP_PROP_EXPOSURE, int(new_value))
        else:
            self.get_logger().info("Exposure OK!")


    def cam_info_cb(self, msg: CameraInfo):

        if self.cam_info is None and not self.robot_is_simulated:
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, msg.width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, msg.height)

        self.cam_info = msg
        
        # FOV calcualtion according to https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
        fx = msg.k[0]
        fy = msg.k[4]
        width = msg.width
        height = msg.height
        
        self.fov_x_rad = 2 * math.atan2(width, (2 * fx))
        self.fov_y_rad = 2 * math.atan2(height, (2 * fy))

    def landing_marker_id_callback(self, msg: UInt8):
        self.landing_marker_id = msg


    def aruco_callback(self, img_in):
        diagnostic_status = DiagnosticStatus()
        diagnostic_status.name = "camera_info_status"
        diagnostic_status.hardware_id = "landing_cam"

        if self.cam_info is None:
            diagnostic_status.level = DiagnosticStatus.ERROR
            diagnostic_status.message = "Camera info is not set"
        else:
            diagnostic_status.level = DiagnosticStatus.OK
            diagnostic_status.message = "Camera info is available"

        self.diagnostics_pub.publish(diagnostic_status)

        if self.cam_info is not None:
            msg = Vector3()

            ARUCO_DICT = {
                "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
                "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
                "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
                "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
                "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
                "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
                "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
                "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
                "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
                "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
                "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
                "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
                "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
                "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
                "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
                "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
                "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
                "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
            }

            image = self.bridge.imgmsg_to_cv2(img_in)
            image_detected = copy(image)
            cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
            frame_cx = int(image.shape[1] / 2)
            frame_cy = int(image.shape[0] / 2)
            #angle_by_pixel = 1.047 / image.shape[1]
            angle_by_pixel_x = self.fov_x_rad / image.shape[1]
            angle_by_pixel_y = self.fov_y_rad / image.shape[0]
            #args = {'type': 'DICT_4X4_1000'}
            #arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
            args = {'type': 'DICT_4X4_1000'}
            arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

            corners, ids, rejected = None, None, None

            if platform.freedesktop_os_release().get("VERSION_CODENAME") == 'jammy':
                arucoParams = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
                corners, ids, rejected = detector.detectMarkers(image)
            else:
                arucoParams = cv2.aruco.DetectorParameters_create()
                (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
                    parameters=arucoParams)


            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    id_is_not_searched_for = None

                    if self.target_landing_marker_id is not None:
                        id_is_not_searched_for = bool(markerID != self.landing_marker_id and markerID != self.target_landing_marker_id)
                    else:
                        id_is_not_searched_for = bool(markerID != self.landing_marker_id)
                    if id_is_not_searched_for:
                        continue
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                    cv2.circle(image_detected, (cX, cY), 3, (0, 0, 255), 3)

                    msg = Vector3()
                    
                    cv2.circle(image_detected, (frame_cx, frame_cy), 2, (0, 255, 0), 2)
                    cv2.line(image_detected, (cX, frame_cy), (frame_cx, frame_cy), (255, 0, 0), 2)
                    cv2.line(image_detected, (frame_cx, cY), (frame_cx, frame_cy), (255, 0, 0), 2)

                    # x_from_c = float(cX - frame_cx)
                    # y_from_c = - float(cY - frame_cy)

                    x_from_c = - float(cY - frame_cy)
                    y_from_c = - float(cX - frame_cx)

                    cv2.putText(image_detected, f"x: {x_from_c}\ny: {y_from_c}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                    
                    tr_x_from_c = - float(topRight[1] - frame_cy)
                    tr_y_from_c = - float(topRight[0] - frame_cx)

                    br_x_from_c = - float(bottomRight[1] - frame_cy)
                    br_y_from_c = - float(bottomRight[0] - frame_cx)

                    alpha = math.atan2(tr_y_from_c - br_y_from_c, tr_x_from_c - br_x_from_c)

                    msg.x = (angle_by_pixel_x * x_from_c)
                    msg.y = (angle_by_pixel_y * y_from_c) 
                    msg.z = alpha
                    
                    self.coordinates_pub.publish(msg)

            # cv2.imwrite(f"/home/{user}/Pictures/land.jpg",image_detected)
            imgmsg = self.bridge.cv2_to_imgmsg(image_detected)
            self.detection_pub.publish(imgmsg)
            # raw_imgmsg = self.bridge.cv2_to_imgmsg(image)
            # self.camera_pub.publish(raw_imgmsg)

    def light_timer(self):
        if self.inimg is not None:
            img = cv2.cvtColor(self.inimg,cv2.COLOR_BGR2HSV)
            array = []
            sum = 0
            for i in range(480):
                for j in range(640):
                        array.append(img[i][j][2])

            array2 = sorted(array)
            #array3 = array2[:76800] + array2[230400:]
            array3 = array2[76800:230400]

            for i in range(len(array3)):
                sum += array3[i]

            average = sum/(480*640/2)   
            
            if average > 55:
                self.light_is_bright = 1
            else:
                self.light_is_bright = 0

            str_aver=str(average)
            width = str(self.inimg.shape[0])
            heidth = str(self.inimg.shape[1])
            self.str_average = str_aver + " " + width + " " + heidth

    def timer_callback(self):
        # print(f"current exposure = {self.cam.get(cv2.CAP_PROP_EXPOSURE)}")
        diagnostic_status = DiagnosticStatus()
        diagnostic_status.name = "camera_info_status"
        diagnostic_status.hardware_id = "landing_cam"

        if self.cam_info is None:
            diagnostic_status.level = DiagnosticStatus.ERROR
            diagnostic_status.message = "Camera info is not set"
        else:
            diagnostic_status.level = DiagnosticStatus.OK
            diagnostic_status.message = "Camera info is available"

        self.diagnostics_pub.publish(diagnostic_status)
        
        if self.cam_info is not None:
            msg = Vector3()

            ARUCO_DICT = {
                "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
                "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
                "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
                "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
                "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
                "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
                "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
                "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
                "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
                "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
                "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
                "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
                "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
                "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
                "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
                "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
                "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
                "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
                    }

            result, image = self.cam.read() 
            self.inimg = copy(image)
            if result==1 and image is not None:        
                image_detected = copy(image)
                cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
                frame_cx = int(image.shape[1] / 2)
                frame_cy = int(image.shape[0] / 2)
                angle_by_pixel_x = self.fov_x_rad / image.shape[1]
                angle_by_pixel_y = self.fov_y_rad / image.shape[0]
                #args = {'type': 'DICT_4X4_1000'}
                #arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
                args = {'type': 'DICT_4X4_1000'}
                arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

                corners, ids, rejected = None, None, None

                if platform.freedesktop_os_release().get("VERSION_CODENAME") == 'jammy':
                    arucoParams = cv2.aruco.DetectorParameters()
                    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
                    corners, ids, rejected = detector.detectMarkers(image)
                else:
                    arucoParams = cv2.aruco.DetectorParameters_create()
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
                        parameters=arucoParams)

                # verify *at least* one ArUco marker was detected
                if len(corners) > 0:
                    # flatten the ArUco IDs list
                    ids = ids.flatten()
                    # loop over the detected ArUCo corners
                    for (markerCorner, markerID) in zip(corners, ids):
                        id_is_not_searched_for = None

                        if self.target_landing_marker_id is not None:
                            id_is_not_searched_for = bool(markerID != self.landing_marker_id and markerID != self.target_landing_marker_id)
                        else:
                            id_is_not_searched_for = bool(markerID != self.landing_marker_id)
                        if id_is_not_searched_for:
                            continue
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners

                        # convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1]))
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                        topLeft = (int(topLeft[0]), int(topLeft[1]))
                        
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                        cv2.circle(image_detected, (cX, cY), 3, (0, 0, 255), 3)

                        msg = Vector3()
                        x_from_c = - float(cY - frame_cy)
                        y_from_c = - float(cX - frame_cx)

                        cv2.putText(image_detected, f"x: {x_from_c}\ny: {y_from_c}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                        
                        tr_x_from_c = - float(topRight[1] - frame_cy)
                        tr_y_from_c = - float(topRight[0] - frame_cx)

                        br_x_from_c = - float(bottomRight[1] - frame_cy)
                        br_y_from_c = - float(bottomRight[0] - frame_cx)

                        alpha = math.atan2(tr_y_from_c - br_y_from_c, tr_x_from_c - br_x_from_c)

                        msg.x = (angle_by_pixel_x * x_from_c)
                        msg.y = (angle_by_pixel_y * y_from_c) 
                        msg.z = alpha
                        
                        self.coordinates_pub.publish(msg)

                image_detected = cv2.putText(image_detected, self.str_average, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 3, cv2.LINE_AA) 
                imgmsg = self.bridge.cv2_to_imgmsg(image_detected)
                raw_imgmsg = self.bridge.cv2_to_imgmsg(image)
                self.camera_pub.publish(raw_imgmsg)
                self.detection_pub.publish(imgmsg)
                self.camera_info_pub.publish(self.cam_info)

                # cv2.imwrite(f"/home/{user}/Pictures/land.jpg",image_detected)
            else:
                print('Nowhere to land')
                
def main(args=None):
    rclpy.init(args=args)

    landing_publisher = LandingPublisher()

    rclpy.spin(landing_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    landing_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



