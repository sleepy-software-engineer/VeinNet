import glob
import os
import cv2
import numpy as np

DATASET_PATH = "/home/lucian/University/MSc-Courses/BiometricSystems/data/"
PATIENTS = [f"{i:03}" for i in range(1, 101)] 
HAND = "l"  
SPECTRUM = "940" 
BATCH_SIZE = 1
OUTPUT_DIR = "/home/lucian/University/MSc-Courses/BiometricSystems/src/hand/out/" 

class HandGeometryDataLoader:
    def __init__(self, dataset_dir, batch_size=1):
        """
        Initialize the data loader with the dataset directory and batch size.
        Args:
            dataset_dir (str): Directory containing the hand images.
            batch_size (int): Number of patients per batch.
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def _load_images(self, person_id, hand, spectrum):
        """
        Load images for a specific patient, hand, and spectrum.
        Args:
            person_id (str): ID of the patient.
            hand (str): Hand type ('l' for left, 'r' for right).
            spectrum (str): Spectrum type (e.g., '940').
        Returns:
            list: List of image file paths for the patient.
        """
        pattern = f"{person_id}_{hand}_{spectrum}_*.jpg"
        return glob.glob(os.path.join(self.dataset_dir, pattern))

    def _extract_features(self, image_path):
        """
        Extract hand geometry features from a single image.
        Args:
            image_path (str): Path to the hand image.
        Returns:
            np.ndarray: Array of extracted features.
        """
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

    def generate_batches(self, hand, spectrum):
        """
        Generate batches of hand geometry features for all patients.
        Yields:
            Tuple[np.ndarray, List[str]]: (Batch of feature arrays, list of patient IDs).
        """
        patient_batches = []
        patient_ids = []

        for person_id in PATIENTS:
            images = self._load_images(person_id, hand, spectrum)
            if len(images) == 6:  # Ensure exactly 6 images per patient
                # Extract features for the patient's 6 images
                features = [self._extract_features(img) for img in images]
                patient_batches.append(np.stack(features, axis=0))  # Shape: (6, num_features)
                patient_ids.append(person_id)

                # Yield the batch if it reaches the specified batch size
                if len(patient_batches) == self.batch_size:
                    yield np.stack(patient_batches, axis=0), patient_ids
                    patient_batches = []
                    patient_ids = []

        # Yield the last batch if it exists
        if patient_batches:
            yield np.stack(patient_batches, axis=0), patient_ids
            
def test_hand_geometry_dataloader():
    """
    Test function for HandGeometryDataLoader.
    Saves features to a text file for verification.
    """
    geometry_loader = HandGeometryDataLoader(dataset_dir=DATASET_PATH, batch_size=BATCH_SIZE)
    output_dir = os.path.join(OUTPUT_DIR, "hand_geometry_test")
    os.makedirs(output_dir, exist_ok=True)

    batch_generator = geometry_loader.generate_batches(hand=HAND, spectrum=SPECTRUM)

    for features_batch, patient_ids in batch_generator:
        print(f"Patient IDs: {patient_ids}")
        print(f"Batch shape: {features_batch.shape}")  # Expected: (batch_size, 6, num_features)

        for i, patient_id in enumerate(patient_ids):
            # Save features for each patient
            save_path = os.path.join(output_dir, f"features_{patient_id}.txt")
            np.savetxt(save_path, features_batch[i], header=f"Features for Patient {patient_id}")
            print(f"Saved: {save_path}")

        break  # Test only the first batch


if __name__ == "__main__":
    test_hand_geometry_dataloader()
