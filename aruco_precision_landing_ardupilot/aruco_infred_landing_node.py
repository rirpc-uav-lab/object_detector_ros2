import rclpy
from rclpy.node import Node

import sys
from getpass import getuser
import numpy as np
import cv2
import math
from copy import copy

from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist

import platform

class LandingPublisher(Node):

    def __init__(self, username=None):
        
        super().__init__('landing_publisher')
        self.coordinates_pub = self.create_publisher(Vector3, '/camera/landing_position', 10)
        self.detection_pub = self.create_publisher(Image, "camera/detected_markers", 10)
        self.bridge = CvBridge()
        
        user = username
        if username == None:
            user = getuser()

        if user == "firefly":
            self.cam = cv2.VideoCapture("/dev/video40")
            # self.res, self.inimg = self.cam.read()
            self.inimg = None
            self.light_is_bright = False
            timer_period = 0.1  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
            self.light_timer = self.create_timer(timer_period, self.light_timer)
        else:
            self.cam_sub = self.create_subscription(Image, '/camera', self.aruco_callback, 10)



    def aruco_callback(self, img_in):
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
        angle_by_pixel = 1.047 / image.shape[1]

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

                self.coordinates_pub.publish(msg)

        imgmsg = self.bridge.cv2_to_imgmsg(image_detected)
        self.detection_pub.publish(imgmsg)

    def light_timer(self):
        if self.inimg is not None:
            img = cv2.cvtColor(self.inimg,cv2.COLOR_BGR2HSV)
            sum = 0

            for i in range(0,480,4):
                for j in range(0,640,4):
                        sum += img[i][j][2]
            
            average = sum/(480*640/16)
            if average > 140:
                self.light_is_bright = True
        

    def timer_callback(self):
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
        self.inimg = image
        if result==1:       
            image_detected = copy(image)
            cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
            frame_cx = int(image.shape[1] / 2)
            frame_cy = int(image.shape[0] / 2)
            angle_by_pixel = 1.047 / image.shape[1]

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
                    if markerID != 3:
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

                    self.coordinates_pub.publish(msg)
            else:    
                if not self.light_is_bright:
                    kernel = np.ones((5, 5), np.uint8)
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    lower = np.array([0, 0, 250])
                    upper = np.array([150, 255, 255])

                    ############################################################################################
                    mask = cv2.inRange(hsv, lower, upper)
                    # cv2.imshow('mask',mask)
                    # cv2.waitKey(0)
                    ####################################################################################################
                    #mask = cv2.erode(mask, kernel, iterations=5) 
                    mask = cv2.dilate(mask, kernel, iterations=4) 
                    # cv2.imshow('erode_mask',mask)
                    # key = cv2.waitKey(30)
                    # if key == ord('q') or key == 27:
                    #         break
                    #############################################################################################

                    contours, hierarchies = cv2.findContours(
                    mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    blank = np.zeros(mask.shape[:2], 
                                    dtype='uint8')
                    cv2.drawContours(blank, contours, -1, 
                                    (255, 0, 0), 1)
                    if len(contours) > 0:
                        i = max(contours, key = cv2.contourArea)
                        
                        cv2.drawContours(image_detected, [i], 0, (0,255,0), 3)
                        M = cv2.moments(i)
                        if M['m00'] != 0:
                                    cX = int(M['m10']/M['m00'])
                                    cY = int(M['m01']/M['m00'])
                                    cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
                                    # cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
                                    msg = Vector3()
                                    msg.x = angle_by_pixel * float(cX - frame_cx)
                                    msg.y = angle_by_pixel * float(cY - frame_cy)    
                                    cv2.circle(image_detected, (cX, cY), 3, (0, 0, 255), 3)
                                    self.coordinates_pub.publish(msg)

            imgmsg = self.bridge.cv2_to_imgmsg(image_detected)
            self.detection_pub.publish(imgmsg)

            cv2.imwrite("/home/firefly/Pictures/land.jpg",image_detected)
        else:
            print('Bei hvostatyx')
            
def main(args=None):
    rclpy.init(args=args)

    landing_publisher = None

    print(sys.argv)


    if len(sys.argv) > 1:
        if sys.argv[1] != None:
            landing_publisher = LandingPublisher(username=sys.argv[1])
        else:
            landing_publisher = LandingPublisher()
    else:
        landing_publisher = LandingPublisher()


    rclpy.spin(landing_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    landing_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



