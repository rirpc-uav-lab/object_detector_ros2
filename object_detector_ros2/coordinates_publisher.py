import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import numpy as np
import cv2
import math
# from pymavlink import mavutil
# from dronekit import connect #, VehicleMode, LocationGlobalRelative, LocationGlobal
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist



class ArucoNode(Node):

    def __init__(self):

        super().__init__('Aruco_Node')
        # self.aruco_pub = self.create_publisher(Twist, '/aruco_result', 10)
        self.aruco_sub = self.create_subscription(Image, '/camera', self.Aruco_callback, 10)
        self.coordinates_pub = self.create_publisher(Vector3, '/camera/landing_position', 10)
        self.detection_pub = self.create_publisher(Image, '/camera/detected_markers', 10)

        # self.vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
        # self.vehicle.parameters['PLND_ENABLED'] = 1
        # self.vehicle.parameters['PLND_TYPE'] = 1
        # self.vehicle.parameters['PLND_EST_TYPE'] = 0
        # self.vehicle.parameters['LAND_SPEED'] = 20
        # self.vehicle = None

    def Aruco_callback(self, sensor_msgs: Image):
        msg = sensor_msgs

        
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
        
        frame_cx = int(image.shape[1] / 2)
        frame_cy = int(image.shape[0] / 2)
        angle_by_pixel = 1.047 / image.shape[1]
        

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

        args = {'type': 'DICT_4X4_1000'}
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

        arucoParams = cv2.aruco.DetectorParameters()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
            parameters=arucoParams)

        image_detected = image.copy()
        cv2.circle(image_detected, (frame_cx, frame_cy), 3, (255, 0, 0), 3)

        # print(f"All: {(corners, ids, rejected)}")

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                if markerID != 688:
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
                msg.x = angle_by_pixel * float(cX - frame_cx)
                msg.y = angle_by_pixel * float(cY - frame_cy)

                # msg.x = float(cX - frame_cx)
                # msg.y = float(cY - frame_cy)

                self.coordinates_pub.publish(msg)

        imgmsg = bridge.cv2_to_imgmsg(image_detected)
        self.detection_pub.publish(imgmsg)

def main(args=None):
    rclpy.init(args=args)
    Aruco_Node = ArucoNode()
    rclpy.spin(Aruco_Node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
        