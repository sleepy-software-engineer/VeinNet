import cv2
import numpy as np

class ContourProcessor:
    @staticmethod
    def find_defects(largest_contour):
        """
        Find convexity defects from the largest contour.
        """
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)
        return defects

    @staticmethod
    def get_far_points(defects, largest_contour, num_points=4):
        """
        Extract farthest points from convexity defects.
        """
        if defects is None:
            return None
        defects = sorted(defects, key=lambda x: x[0, 3], reverse=True)
        far_points = [tuple(largest_contour[defects[i][0][2]][0]) for i in range(min(num_points, len(defects)))]
        return sorted(far_points, key=lambda point: point[1])  # Sort by y-coordinate

    @staticmethod
    def compute_midpoint_and_direction(point1, point2):
        """
        Compute the midpoint and directional vector between two points.
        """
        try:
            midpoint = (int((point1[0] + point2[0]) // 2), int((point1[1] + point2[1]) // 2))
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length != 0:
                dx /= length
                dy /= length
            return midpoint, dx, dy, length
        except Exception as e:
            print(f"Error computing midpoint and direction: {e}")
            raise

    @staticmethod
    def compute_perpendicular_point(midpoint, dx, dy, distance=50):
        """
        Compute a point perpendicular to the line between two points.
        """
        x_perpendicular = int(midpoint[0] + distance * dy)
        y_perpendicular = int(midpoint[1] - distance * dx)
        return x_perpendicular, y_perpendicular

    @staticmethod
    def generate_square_vertices(perpendicular_point, length, offset=50):
        """
        Generate vertices for a square based on a perpendicular point and length.
        """
        x, y = perpendicular_point
        vertices = [
            (x + offset, y),
            (x + offset, y - length),
            (x + offset + length, y - length),
            (x + offset + length, y)
        ]
        return vertices

    @staticmethod
    def rotate_square(vertices, midpoint, angle):
        """
        Rotate the square vertices around a midpoint.
        """
        if not isinstance(midpoint, tuple) or len(midpoint) != 2:
            raise ValueError(f"Invalid midpoint: {midpoint}")
        if not all(isinstance(coord, (int, float)) for coord in midpoint):
            raise TypeError(f"Midpoint coordinates must be int or float, got: {midpoint}")

        rotation_matrix = cv2.getRotationMatrix2D(midpoint, angle, scale=1)
        rotated_vertices = cv2.transform(np.array([vertices], dtype=np.float32), rotation_matrix).squeeze().astype(np.int32)
        return rotated_vertices

    @staticmethod
    def translate_square(vertices, start_point):
        """
        Translate the square vertices by a given point.
        """
        return vertices + start_point

    @staticmethod
    def compute_transform(vertices, length):
        """
        Compute the perspective transform matrix for the square vertices.
        """
        rectified_order = np.array([[0, 0], [length, 0], [length, length], [0, length]], dtype=np.float32)
        vertices = np.array(vertices, dtype=np.float32)
        return cv2.getPerspectiveTransform(vertices, rectified_order)

class VeinImageProcessor:
    @staticmethod
    def load_and_crop_image(image_path):
        """
        Load the image in grayscale and crop the unnecessary part.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cropped_image = image[:, :image.shape[1] - 120]
        return cropped_image

    @staticmethod
    def preprocess_contours(image):
        """
        Blur the image, threshold it, and find the largest contour.
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    @staticmethod
    def compute_rectified_transform(largest_contour):
        """
        Compute the perspective transform matrix based on the largest contour.
        """
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)

        # Step 1: Find convexity defects
        defects = ContourProcessor.find_defects(largest_contour)

        # Step 2: Get farthest points
        far_points = ContourProcessor.get_far_points(defects, largest_contour)
        if far_points is None or len(far_points) < 3:
            return None, None  # Insufficient points to compute transform

        first_defect_far, third_defect_far = far_points[0], far_points[2]

        # Step 3: Compute midpoint and directional vector
        midpoint, dx, dy, length = ContourProcessor.compute_midpoint_and_direction(first_defect_far, third_defect_far)

        # Step 4: Compute perpendicular point
        perpendicular_point = ContourProcessor.compute_perpendicular_point(midpoint, dx, dy)

        # Step 5: Generate square vertices
        square_vertices = ContourProcessor.generate_square_vertices(perpendicular_point, length)

        # Step 6: Rotate square vertices
        angle = np.arctan2(-dy, dx) * 180 / np.pi
        rotated_vertices = ContourProcessor.rotate_square(square_vertices, midpoint, angle)

        # Step 7: Translate square vertices
        start_point = (first_defect_far[0] - rotated_vertices[0][0], first_defect_far[1] - rotated_vertices[0][1])
        translated_vertices = ContourProcessor.translate_square(rotated_vertices, start_point)

        # Step 8: Compute perspective transform
        transform_matrix = ContourProcessor.compute_transform(translated_vertices, int(length))

        return transform_matrix, int(length)
    
    @staticmethod
    def apply_perspective_transform(image, transform_matrix, length):
        """
        Apply the perspective transform to rectify the image.
        """
        rectified_image = cv2.warpPerspective(image, transform_matrix, (length, length))
        return cv2.equalizeHist(rectified_image)

    @staticmethod
    def apply_gabor_filter(image):
        """
        Apply Gabor filter to enhance vein structures.
        """
        g_kernel_size, g_sigma, g_theta, g_lambda, g_gamma, g_psi = 5, 2.5, np.pi/3, 8.0, 0.4, 0.0
        gabor_kernel = cv2.getGaborKernel((g_kernel_size, g_kernel_size), g_sigma, g_theta, g_lambda, g_gamma, g_psi, ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
        return cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    @staticmethod
    def enhance_contrast(image):
        """
        Apply multiple CLAHE passes and Gaussian blur for contrast enhancement.
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        enhanced_image = clahe.apply(image)
        enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        for tile_size in [(4, 4), (8, 8), (10, 10)]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
            enhanced_image = clahe.apply(enhanced_image)
        return enhanced_image

    @staticmethod
    def binarize_image(image):
        """
        Threshold the image to produce a binary image.
        """
        _, binary_image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        return binary_image

    def preprocess_image(self, image_path):
        """
        Main preprocessing pipeline.
        """
        # Step 1: Load and crop the image
        cropped_image = self.load_and_crop_image(image_path)

        # Step 2: Find the largest contour
        largest_contour = self.preprocess_contours(cropped_image)

        # Step 3: Compute rectification transform
        transform_matrix, length = self.compute_rectified_transform(largest_contour)

        if transform_matrix is not None:
            # Step 4: Rectify and equalize the image
            rectified_image = self.apply_perspective_transform(cropped_image, transform_matrix, length)
        else:
            rectified_image = cropped_image

        # Step 5: Apply Gabor filter
        gabor_filtered = self.apply_gabor_filter(rectified_image)

        # Step 6: Enhance contrast
        enhanced_image = self.enhance_contrast(gabor_filtered)

        # Step 7: Binarize the image
        binary_image = self.binarize_image(enhanced_image)

        return binary_image