import cv2
import numpy as np

class HandGeometryProcessor:
    @staticmethod
    def extract_features(image_path):
        # Read and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cropped_image = image[:, :image.shape[1] - 120]
        blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract geometric features
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)

        convex_hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(convex_hull)
        solidity = contour_area / hull_area if hull_area != 0 else 0

        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h != 0 else 0

        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull_indices) if hull_indices is not None else None
        num_defects = len(defects) if defects is not None else 0
        avg_defect_depth = (
            sum([d[0][3] for d in defects]) / num_defects if defects is not None and num_defects > 0 else 0
        )

        moments = cv2.moments(largest_contour)
        centroid_x = moments["m10"] / moments["m00"] if moments["m00"] != 0 else 0
        centroid_y = moments["m01"] / moments["m00"] if moments["m00"] != 0 else 0

        return np.array(
            [
                contour_area, contour_perimeter, hull_area, solidity, w, h, aspect_ratio,
                num_defects, avg_defect_depth, centroid_x, centroid_y
            ],
            dtype=np.float32,
        )