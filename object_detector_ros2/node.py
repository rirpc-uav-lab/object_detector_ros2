import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import numpy as np
import cv2
import math
from pymavlink import mavutil
from dronekit import connect #, VehicleMode, LocationGlobalRelative, LocationGlobal
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

        self.vehicle = connect('/dev/ttyUSB0', wait_ready=True, baud=115200)
        # self.vehicle.parameters['PLND_ENABLED'] = 1
        # self.vehicle.parameters['PLND_TYPE'] = 1
        # self.vehicle.parameters['PLND_EST_TYPE'] = 0
        # self.vehicle.parameters['LAND_SPEED'] = 20

    def Aruco_callback(self, sensor_msgs: Image):
        result = Twist()
        result.linear.x = 0.0
        result.angular.x = 0.0

        msg = sensor_msgs
        
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
        

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

        # print(f"All: {(corners, ids, rejected)}")

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                x, y = 0, 0

                if cX < 320:
                    x = -(320-cX)
                else:
                    x = (cX - 320)

                if cY < 240:
                    y = -(240-cY) 

                else:
                    y = (cY - 240)
                
                if markerID == 688:
                    msg = self.vehicle.message_factory.landing_target_encode(
                        0,
                        0,
                        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                        cX,
                        cY,
                        0,0,0
                    )

                    print(msg)
                    print(type(msg))

                    self.vehicle.send_mavlink(msg)
                    self.vehicle.flush()
            #result.angular.x = float(markerID)
                # result.linear.x = float(x)
                # result.linear.y = float(y)

                # self.aruco_pub.publish(result)



def main(args=None):
    rclpy.init(args=args)

    Aruco_Node = ArucoNode()

    rclpy.spin(Aruco_Node)

    rclpy.shutdown()

        

    if __name__ == "__main__":
        main()
        