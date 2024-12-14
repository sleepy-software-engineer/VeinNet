import cv2
import numpy as np


class ContourProcessor:
    @staticmethod
    def find_defects(largest_contour):
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)
        return defects

    @staticmethod
    def get_far_points(defects, largest_contour, num_points=4):
        defects = sorted(defects, key=lambda x: x[0, 3], reverse=True)
        far_points = [
            tuple(largest_contour[defects[i][0][2]][0])
            for i in range(min(num_points, len(defects)))
        ]
        return sorted(far_points, key=lambda point: point[1])

    @staticmethod
    def compute_midpoint_and_direction(point1, point2):
        midpoint = (
            int((point1[0] + point2[0]) // 2),
            int((point1[1] + point2[1]) // 2),
        )
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length != 0:
            dx /= length
            dy /= length
        return midpoint, dx, dy, length

    @staticmethod
    def compute_perpendicular_point(midpoint, dx, dy, distance=50):
        x_perpendicular = int(midpoint[0] + distance * dy)
        y_perpendicular = int(midpoint[1] - distance * dx)
        return x_perpendicular, y_perpendicular

    @staticmethod
    def calculate_length(first_defect_far, third_defect_far):
        return int(
            np.sqrt(
                (third_defect_far[0] - first_defect_far[0]) ** 2
                + (third_defect_far[1] - first_defect_far[1]) ** 2
            )
        )

    @staticmethod
    def calculate_midpoint(midpoint):
        return (int(midpoint[0]), int(midpoint[1]))

    @staticmethod
    def generate_square_vertices(perpendicular_point, length, offset=50):
        x, y = perpendicular_point
        vertices = [
            (x + offset, y),
            (x + offset, y - length),
            (x + offset + length, y - length),
            (x + offset + length, y),
        ]
        return vertices

    @staticmethod
    def process_perpendicular_point(midpoint, perpendicular_point):
        x, y = perpendicular_point
        dx = x - midpoint[0]
        dy = y - midpoint[1]
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length
        return dx, dy

    @staticmethod
    def rotate_square(vertices, midpoint, angle):
        rotation_matrix = cv2.getRotationMatrix2D(midpoint, angle, scale=1)
        rotated_vertices = (
            cv2.transform(np.array([vertices], dtype=np.float32), rotation_matrix)
            .squeeze()
            .astype(np.int32)
        )
        return rotated_vertices

    @staticmethod
    def translate_square(vertices, start_point):
        return vertices + start_point

    @staticmethod
    def process_translation(dx, dy, vertices_translated):
        translation = (int(50 * dx), int(50 * dy))
        translated = np.array(vertices_translated + translation, dtype=np.float32)
        return translated

    @staticmethod
    def compute_transform(vertices, length):
        rectified_order = np.array(
            [[0, 0], [length, 0], [length, length], [0, length]], dtype=np.float32
        )
        vertices = np.array(vertices, dtype=np.float32)
        return cv2.getPerspectiveTransform(vertices, rectified_order)


class DataProcessor:
    @staticmethod
    def load_and_crop_image(image_path):
        """
        Load the image in grayscale and crop the unnecessary part.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cropped_image = image[:, : image.shape[1] - 120]
        return cropped_image

    @staticmethod
    def preprocess_contours(image):
        """
        Blur the image, threshold it, and find the largest contour.
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    @staticmethod
    def compute_rectified_transform(largest_contour):
        defects = ContourProcessor.find_defects(largest_contour)

        far_points = ContourProcessor.get_far_points(defects, largest_contour)

        first_defect_far, third_defect_far = far_points[0], far_points[2]

        midpoint, dx, dy, length = ContourProcessor.compute_midpoint_and_direction(
            first_defect_far, third_defect_far
        )

        perpendicular_point = ContourProcessor.compute_perpendicular_point(
            midpoint, dx, dy
        )

        length = ContourProcessor.calculate_length(first_defect_far, third_defect_far)

        square_vertices = ContourProcessor.generate_square_vertices(
            perpendicular_point, length
        )

        midpoint = ContourProcessor.calculate_midpoint(midpoint)

        angle = np.arctan2(-dy, dx) * 180 / np.pi
        rotated_vertices = ContourProcessor.rotate_square(
            square_vertices, midpoint, angle
        )

        start_point = (
            first_defect_far[0] - rotated_vertices[0][0],
            first_defect_far[1] - rotated_vertices[0][1],
        )
        translated_vertices = ContourProcessor.translate_square(
            rotated_vertices, start_point
        )

        dx, dy = ContourProcessor.process_perpendicular_point(
            midpoint, perpendicular_point
        )

        translated_vertices = ContourProcessor.process_translation(
            dx, dy, translated_vertices
        )

        transform_matrix = ContourProcessor.compute_transform(
            translated_vertices, length
        )

        return transform_matrix, length

    @staticmethod
    def apply_perspective_transform(image, transform_matrix, length):
        rectified_image = cv2.warpPerspective(image, transform_matrix, (length, length))
        return cv2.equalizeHist(rectified_image)

    @staticmethod
    def apply_gabor_filter(image):
        g_kernel_size, g_sigma, g_theta, g_lambda, g_gamma, g_psi = (
            5,
            2.5,
            np.pi / 3,
            8.0,
            0.4,
            0.0,
        )
        gabor_kernel = cv2.getGaborKernel(
            (g_kernel_size, g_kernel_size),
            g_sigma,
            g_theta,
            g_lambda,
            g_gamma,
            g_psi,
            ktype=cv2.CV_32F,
        )
        filtered_image = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
        return cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    @staticmethod
    def enhance_contrast(image):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        enhanced_image = clahe.apply(image)
        enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        for tile_size in [(4, 4), (8, 8), (10, 10)]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
            enhanced_image = clahe.apply(enhanced_image)
        return enhanced_image

    @staticmethod
    def binarize_image(image):
        _, binary_image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        return binary_image

    def preprocess_image(self, image_path):
        cropped_image = self.load_and_crop_image(image_path)

        largest_contour = self.preprocess_contours(cropped_image)

        transform_matrix, length = self.compute_rectified_transform(largest_contour)

        rectified_image = self.apply_perspective_transform(
            cropped_image, transform_matrix, length
        )

        gabor_filtered = self.apply_gabor_filter(rectified_image)

        enhanced_image = self.enhance_contrast(gabor_filtered)

        binary_image = self.binarize_image(enhanced_image)

        output = cv2.resize(binary_image, (128, 128), interpolation=cv2.INTER_AREA)

        return output
