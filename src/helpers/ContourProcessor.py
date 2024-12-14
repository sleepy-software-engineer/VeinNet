import cv2

class ContourProcessor:
    """
    A class used to process contours and find convexity defects.
    """

    @staticmethod
    def find_defects(largest_contour: list) -> list:
        """
        Finds the convexity defects in the largest contour.
        """
        # Find the convex hull of the largest contour
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        # Find the convexity defects of the largest contour using the convex hull
        defects = cv2.convexityDefects(largest_contour, hull)
        return defects

    @staticmethod
    def get_far_points(defects: list, largest_contour: list, num_points: int = 4) -> list:
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
    def load_and_crop_image(image_path: str) -> any:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cropped_image = image[:, : image.shape[1] - 120]
        return cropped_image

    @staticmethod
    def preprocess_contours(image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour