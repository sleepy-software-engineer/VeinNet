import cv2


class ContourProcessor:
    """
    A class used to process contours and find convexity defects.
    """

    @staticmethod
    def find_defects(largest_contour):
        """
        Finds the convexity defects in the largest contour.
        """
        # Find the convex hull of the largest contour
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        # Find the convexity defects of the largest contour using the convex hull
        defects = cv2.convexityDefects(largest_contour, hull)
        return defects

    @staticmethod
    def get_far_points(defects, largest_contour, num_points=4):
        """
        Retrieves the farthest points from the convexity defects.
        """
        # Sort defects based on the distance to the farthest point (in descending order)
        defects = sorted(defects, key=lambda x: x[0, 3], reverse=True)
        # Extract the farthest points from the sorted defects
        far_points = [
            tuple(largest_contour[defects[i][0][2]][0])
            for i in range(min(num_points, len(defects)))
        ]
        # Sort the farthest points by their y-coordinates
        return sorted(far_points, key=lambda point: point[1])

    @staticmethod
    def load_and_crop_image(image_path):
        """
        Loads an image from the specified path and crops it by removing 120 pixels from the right side.
        """
        # Load the image in grayscale mode
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Crop the image by removing 120 pixels from the right side
        cropped_image = image[:, : image.shape[1] - 120]
        # Return the cropped image
        return cropped_image

    @staticmethod
    def preprocess_contours(image):
        """
        Preprocesses the image to find and return the largest contour.
        """
        # Apply Gaussian blur to the image to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Apply binary thresholding to the blurred image
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Find the largest contour based on contour area
        largest_contour = max(contours, key=cv2.contourArea)
        # Return the largest contour
        return largest_contour
