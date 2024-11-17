import cv2
import numpy as np
import glob
import os

def save_coefficients(mtx, dist, path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()

def load_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_matrix

def main():
    # Ask for chessboard parameters
    intersections_x = int(input("Enter the number of inner corners along the width: "))
    intersections_y = int(input("Enter the number of inner corners along the height: "))
    square_size = float(input("Enter the size of each square in meters: "))

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
    objp = np.zeros((intersections_y * intersections_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:intersections_x, 0:intersections_y].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Create a directory to save the images
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')

    # Start video capture
    cap = cv2.VideoCapture('/dev/video2')  # Change the argument to the desired video port

    img_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (intersections_x, intersections_y), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (intersections_x, intersections_y), corners2, ret)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27:  # ESC key to exit
            break
        elif k == 32:  # SPACE key to save the image
            img_name = f"calibration_images/calibration_{img_counter}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved!")
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the camera matrix and the distortion coefficients to a file
    save_coefficients(mtx, dist, "calibration_data.yaml")

    print("Camera matrix : \n", mtx)
    print("dist : \n", dist)

if __name__ == "__main__":
    main()
