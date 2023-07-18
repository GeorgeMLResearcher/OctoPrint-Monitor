import cv2
import numpy as np
import glob

def capture_image():
    number_image_total = 0
    cap = cv2.VideoCapture("http://192.168.0.17:8080/video")

    while True:
        ret, frame = cap.read()

        # press esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retu, corners = cv2.findChessboardCorners(gray, (9,8), None)

        if retu:
            if cv2.waitKey(1) & 0xFF == ord('s'):
                number_image_total += 1
                cv2.imwrite(f'images/chessboard/chessboard{number_image_total}.png', frame)
                print(f"Image {number_image_total} saved.")
            frame = cv2.drawChessboardCorners(frame, (9, 8), corners, ret)

        cv2.imshow("frame", frame)


    cap.release()
    cv2.destroyAllWindows()


def camera_calibration():

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_size = (12 , 8)
    chessboard_grid_size = 20 
    object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    object_points *= chessboard_grid_size
    
    # Initialize lists to store 3D and 2D pointse
    object_points_list = []
    image_points_list = []

    image_folder_location = r'resources\camera_calibration\second\chessboard*.jpg'

    for image in glob.glob(image_folder_location)[0:15]:
        image = cv2.imread(image)
        image = cv2.resize(image, (1920, 1080))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            print(f"Image {len(object_points_list)} detected.")
            object_points_list.append(object_points)
            corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)

            image_points_list.append(corners)

    # Release all windows and camera devices
    cv2.destroyAllWindows()

    # Calibrate camera using detected 3D and 2D points
    ret, camera_matrix, distortion_matrix, rotation_vector, translation_vector = cv2.calibrateCamera(
        object_points_list, image_points_list, gray.shape[::-1], None, None)

    print(f"reprojection error: {ret} pixels")
    # Save camera calibration parameters to file
    np.savez(image_folder_location.replace(image_folder_location.split('\\')[-1], "camera_calibration"), mtx=camera_matrix, dist=distortion_matrix)

    # Compute mean reprojection error
    mean_error = 0
    error_list = []
    for i in range(len(object_points_list)):
        image_points_projected, _ = cv2.projectPoints(
            object_points_list[i], rotation_vector[i], translation_vector[i], camera_matrix, distortion_matrix)
        image_points_actual = image_points_list[i].reshape(-1, 2)
        error = cv2.norm(image_points_actual, np.squeeze(image_points_projected, axis=1), cv2.NORM_L2)/len(image_points_projected)
            # erorr in percentage two decimal places
        
        error_list.append(error)
        mean_error += error

    # Print mean reprojection error
    print(f"error_list: {error_list}")
    print(f"Reprojection error: {mean_error/len(object_points_list):.2f} pixels")


camera_calibration()
