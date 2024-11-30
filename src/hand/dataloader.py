import glob
import os
import cv2
import numpy as np

class HandGeometryDataLoader:
    def __init__(self, dataset_dir, batch_size=1):
        """
        Initialize the data loader with the dataset directory and batch size.
        Args:
            dataset_dir (str): Directory containing the hand images.
            batch_size (int): Number of samples per batch.
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def _load_images(self):
        """
        Load all image paths from the dataset directory.
        Returns:
            list: List of image file paths.
        """
        return glob.glob(os.path.join(self.dataset_dir, "*.jpg"))

    def _extract_features(self, image_path):
        """
        Extract hand geometry features from a single image.
        Args:
            image_path (str): Path to the hand image.
        Returns:
            np.ndarray: Array of extracted features.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cropped_image = image[:, :image.shape[1] - 120]
        blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Contour features
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)

        # Convex hull features
        convex_hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(convex_hull)
        solidity = contour_area / hull_area if hull_area != 0 else 0

        # Bounding rectangle features
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h != 0 else 0

        # Convexity defects
        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull_indices) if hull_indices is not None else None
        num_defects = len(defects) if defects is not None else 0
        avg_defect_depth = (
            sum([d[0][3] for d in defects]) / num_defects if defects is not None and num_defects > 0 else 0
        )

        # Moments and Hu moments
        moments = cv2.moments(largest_contour)
        centroid_x = moments["m10"] / moments["m00"] if moments["m00"] != 0 else 0
        centroid_y = moments["m01"] / moments["m00"] if moments["m00"] != 0 else 0

        # Combine all features
        features = [
            contour_area, contour_perimeter, hull_area, solidity, w, h, aspect_ratio,
            num_defects, avg_defect_depth, centroid_x, centroid_y
        ]
        return np.array(features, dtype=np.float32)

    def generate_batches(self):
        """
        Generate batches of hand geometry features.
        Yields:
            np.ndarray: Batch of feature arrays.
        """
        images = self._load_images()
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            features_batch = [self._extract_features(img) for img in batch]
            yield np.stack(features_batch, axis=0)
