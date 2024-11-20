import cv2
import numpy as np
import glob
import os
import time


def measure_fps(cap, duration=5):
    """
    Measures the FPS of the video capture stream.

    Parameters:
    cap (cv2.VideoCapture): The video capture object.
    duration (int): The duration in seconds to measure the FPS.

    Returns:
    float: The measured FPS.
    """
    start_time = time.time()
    frame_count = 0

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    return fps

def measure_fps_for_resolutions(cap, supported_resolutions, duration=5):
    """
    Measures the FPS for each supported resolution and returns the resolution with the greatest R2FPS value.

    Parameters:
    cap (cv2.VideoCapture): The video capture object.
    supported_resolutions (list of tuples): List of supported resolutions (width, height).
    duration (int): The duration in seconds to measure the FPS.

    Returns:
    tuple: The resolution with the greatest R2FPS value (width, height).
    """
    best_resolution = None
    best_r2fps = 0

    for width, height in supported_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width == width and actual_height == height:
            fps = measure_fps(cap, duration)
            r2fps = fps*width*height / 100000
            print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, R2FPS: {r2fps:.2f}")
            if r2fps > best_r2fps:
                best_r2fps = r2fps
                best_resolution = (width, height)

    return best_resolution


def get_supported_resolutions(cap):
    # Define possible values for height and width
    possible_resolutions = [
        (640, 480), (800, 600), (1024, 768), (1280, 720), (1280, 960),
        (1600, 1200), (1920, 1080), (2048, 1536), (2592, 1944), (3840, 2160)
    ]

    supported_resolutions = []

    for width, height in possible_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width == width and actual_height == height:
            supported_resolutions.append((width, height))

    return supported_resolutions



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

    supported_resolutions = get_supported_resolutions(cap)

    max_width = max(supported_resolutions, key=lambda x: x[0])[0]
    max_height = max(supported_resolutions, key=lambda x: x[1])[1]

    # print(f"Supported resolutions: {supported_resolutions}")
    print(f"Max resolution: {max_width}x{max_height}")
    # print(f"FPS: {measure_fps(cap)}")
    best_width, best_height = measure_fps_for_resolutions(cap, supported_resolutions, duration=10)
    print(f"Best resolution: {best_width}x{best_height}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)

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
