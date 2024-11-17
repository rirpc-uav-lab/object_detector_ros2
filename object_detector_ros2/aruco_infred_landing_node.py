import rclpy
from rclpy.node import Node
# qos_profile_sensor_data is a default QoS profile for sensor data
# It is defined in rclpy as a combination of the following settings:
# - history: KEEP_LAST
# - depth: 7
# - reliability: BEST_EFFORT
# - durability: VOLATILE
from rclpy.qos import qos_profile_sensor_data


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
        self.robot_is_simulated = try_get(lambda: self.get_parameter("robot_is_simulated"), False)
        self.camera_port = try_get(lambda: self.get_parameter("camera_port"), "/dev/video0")
        self.calibration_config_path = try_get(lambda: self.get_parameter("calibration_config_path"), None)

        #####################################################
        #                 Sim or Real setup                 #
        #####################################################
        self.camera_pub = None
        if not self.robot_is_simulated:
            self.cam = cv2.VideoCapture(self.camera_port)
        
            if self.calibration_config_path is not None:
                with open(calibration_config_path, 'r') as file:
                    calib_config = yaml.safe_load(file)
                    print(calib_config)
                    # TODO: Add camera calibration config logic
                    raise Exception('Camera info is not set - NOT IMPLEMENTED')
            else:
                raise Exception("No calibration config path provided")

            self.str_average = ""
            self.inimg = None

            self.light_is_bright = False
            timer_period = 0.1  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
            # self.light_timer = self.create_timer(10, self.light_timer)
            self.camera_pub = self.create_publisher(Image, "camera/image", 10)
        else:
            self.cam_sub = self.create_subscription(Image, 'camera', self.aruco_callback, 10)
            self.cam_sub = self.create_subscription(CameraInfo, 'camera/camera_info', self.cam_info_cb, 10)
            self.camera_pub = self.create_publisher(Image, "camera/image", 10)


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
                    if markerID != self.landing_marker_id:
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
                    
                    # x = ( angle_by_pixel_x * float(cX - frame_cx) )
                        
                    # if x > 0.022: 
                    #     msg.x = 0.022
                    
                    # elif x < -0.022:
                    #     msg.x = -0.022

                    # else:
                    #     msg.x = x


                    # y = ( angle_by_pixel_y * float(cY - frame_cy) ) 

                    # if y > 0.022: 
                    #     msg.y = 0.022
                    
                    # elif y < -0.022:
                    #     msg.y = -0.022

                    # else:
                    #     msg.y = y

                    self.coordinates_pub.publish(msg)

            # cv2.imwrite(f"/home/{user}/Pictures/land.jpg",image_detected)
            imgmsg = self.bridge.cv2_to_imgmsg(image_detected)
            self.detection_pub.publish(imgmsg)
            raw_imgmsg = self.bridge.cv2_to_imgmsg(image)
            self.camera_pub.publish(raw_imgmsg)

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
                        if markerID != self.landing_marker_id:
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

                # cv2.imwrite(f"/home/{user}/Pictures/land.jpg",image_detected)
            else:
                print('Nowhere to land')
                
def main(args=None):
    rclpy.init(args=args)

    landing_publisher = None

    print(sys.argv)


  #  if len(sys.argv) > 1:
  #      if sys.argv[1] != None:
  #          landing_publisher = LandingPublisher(username=sys.argv[1])
  #      else:
  #          landing_publisher = LandingPublisher()
  #  else:
  #      landing_publisher = LandingPublisher()

    landing_publisher = LandingPublisher()

    rclpy.spin(landing_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    landing_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



