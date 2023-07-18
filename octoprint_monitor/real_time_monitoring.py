import numpy as np
import cv2
from rembg import remove


class GcodeParser:
    def __init__(self, gcode_file_path, offset=[-30, -30, 0]):
        self.offset = offset
        self.gcode_file_path = gcode_file_path
        self.gcode_parser()

    def gcode_parser(self):
        with open(self.gcode_file_path, "r") as file:
            gcode = file.readlines()

        self.coordinates = []  # store the 3D self.points
        for line in gcode:
            tokens = line.strip().split(" ")  # split the line into separate tokens
            if tokens[0] == "G1":  # if the line is a "G1" command
                # if the next token starts with "Z"
                if tokens[1].startswith("Z"):
                    z = float(tokens[1][1:])  # extract the z-coordinate
                # if the next token starts with "X"
                elif tokens[1].startswith("X"):
                    x = float(tokens[1][1:])  # extract the x-coordinate
                    y = float(tokens[2][1:])  # extract the y-coordinate
                    self.coordinates.append(
                        (x, y, z)
                    )  # append the coordinates to our list

        self.coordinates += np.array(self.offset)

        # convert the list to a numpy array and exclude the first two and the last elements
        self.coordinates = np.array(self.coordinates)[2:-1]

        self.coordinates_dic = {
            z: self.coordinates[self.coordinates[:, 2] == z]
            for z in np.unique(self.coordinates[:, 2])
        }

        # return self.coordinates


class PerspectiveProjection:
    def __init__(self, image_file_location, vertices_list):
        self.image = cv2.imread(image_file_location)
        self.camera_calibration_file = r"C:\Users\georg\OneDrive - RMIT University\Master by Research\Assets\Images\Geometric Distortion Detection\Camera Calibration\camera_calibration.npz"
        self.intrinsic_matrix, self.distortion_matrix = np.load(
            self.camera_calibration_file
        ).values()
        self.vertices_list = vertices_list
        # self.faces_list = faces_list
        self.load_image()
        self.pose_estimation()
        self.project_virtual_object()

    def pose_estimation(self):
        # Load camera calibration and distortion matrices
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        marker_size = 39.6
        corners, ids, _ = cv2.aruco.detectMarkers(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), dictionary
        )
        (
            self.rotation_vector,
            self.translation_vector,
            _,
        ) = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, self.intrinsic_matrix, self.distortion_matrix
        )

    def load_image(self):
        if self.image.shape != (1080, 1920):
            self.image = cv2.resize(self.image, (1920, 1080))

    def project_virtual_object(self):
        self.image_points, _ = cv2.projectPoints(
            self.vertices_list,
            self.rotation_vector,
            self.translation_vector,
            self.intrinsic_matrix,
            self.distortion_matrix,
        )
        self.mask = np.zeros_like(self.image)
        self.mask = cv2.drawFrameAxes(
            self.mask,
            self.intrinsic_matrix,
            self.distortion_matrix,
            self.rotation_vector,
            self.translation_vector,
            50,
            thickness=10,
        )

        for i in range(len(self.image_points) - 1):
            cv2.line(
                self.mask,
                tuple(self.image_points[i].ravel().astype(int)),
                tuple(self.image_points[i + 1].ravel().astype(int)),
                (255, 255, 255),
                2,
            )


class ImageProcessing:
    def __init__(self, image, mask, image_points):
        self.image = image
        self.mask = mask
        self.image_points = image_points

        self.image_segmentation_bouding_box()
        self.image_segmentation()
        self.overlay_image()
        self.image_differencing()

    def overlay_image(self):
        self.image_overlay = cv2.addWeighted(self.image, 1, self.mask, 1, 1)

    def image_segmentation_bouding_box(self):
        self.image_points = np.array(self.image_points, dtype=np.float32)
        self.image_points = self.image_points.reshape((-1, 2))
        x, y, w, h = cv2.boundingRect(self.image_points)
        x -= 25
        y -= 25
        w += 50
        h += 50

        self.mask_ = self.mask.copy()
        self.mask_bounding_box = cv2.rectangle(
            self.mask, (x, y), (x + w, y + h), (0, 0, 0), 2
        )

        self.image_crop = self.image[y: y + h, x: x + w]
        self.mask_crop = self.mask[y: y + h, x: x + w]

    def image_segmentation(self):
        self.image_crop_rembg = remove(self.image_crop)

        self.image_crop_gray = cv2.cvtColor(
            self.image_crop_rembg, cv2.COLOR_BGR2GRAY)
        self.mask_crop_gray = cv2.cvtColor(self.mask_crop, cv2.COLOR_BGR2GRAY)

        self.image_crop_bi = cv2.threshold(
            self.image_crop_gray, 0, 255, cv2.THRESH_BINARY
        )[1]
        self.mask_crop_bi = cv2.threshold(
            self.mask_crop_gray, 0, 255, cv2.THRESH_BINARY
        )[1]

    def image_differencing(self):
        self.image_difference = cv2.absdiff(
            self.mask_crop_bi, self.image_crop_bi)
        self.image_difference_percentage = (
            cv2.countNonZero(self.image_difference)
            / cv2.countNonZero(self.mask_crop_bi)
        ) * 100
        self.image_difference_percentage = round(
            float(self.image_difference_percentage), 2
        )


# image_file_location = r"C:\Users\georg\OneDrive - RMIT University\Master by Research\Assets\Images\Geometric Distortion Detection\Frustum\frustum_5.jpg"
# gcode_file_location = r"C:\Users\georg\OneDrive - RMIT University\Master by Research\Assets\CAD\Frustum\frustum-20_1h3m_0.20mm_200C_PLA_ENDER3NEO.gcode"
# gp = GcodeParser(gcode_file_location)


# current_layer = gp.coordinates_dic[0.2]
# pp_model = PerspectiveProjection(image_file_location, gp.coordinates)
# for key, value in gp.coordinates_dic.items():
#     pp = PerspectiveProjection(image_file_location, current_layer)
#     ip = ImageProcessing(pp.image, pp.mask, pp_model.image_points)
#     cv2.imshow("Image", ip.image_crop_rembg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # print(ip.image_difference_percentage)
#     # cv2.imshow("Image Difference", ip.image_difference)
#     # cv2.waitKey(1)
#     current_layer = np.concatenate((current_layer, value), axis=0)
# # cv2.destroyAllWindows()