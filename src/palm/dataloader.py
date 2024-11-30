import glob
import cv2
import numpy as np
import os

class VeinImageDataLoader:
    def __init__(self, dataset_dir, batch_size=1):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def _load_images(self, person_id, hand, spectrum):
        pattern = f"{person_id}_{hand}_{spectrum}_*.jpg"
        matching_files = glob.glob(os.path.join(self.dataset_dir, pattern))
        return matching_files

    def _preprocess_image(self, image_path):
        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Crop and blur the image
        cropped_image = image[:, :image.shape[1] - 120]
        blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)
        
        # Threshold the image
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Contour and rectify the image
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)
        
        if defects is not None:
            defects = sorted(defects, key=lambda x: x[0, 3], reverse=True)
            far_points = [tuple(largest_contour[defects[i][0][2]][0]) for i in range(min(4, len(defects)))]
            far_points = sorted(far_points, key=lambda point: point[1])
            first_defect_far, third_defect_far = far_points[0], far_points[2]

            length = int(np.sqrt((third_defect_far[0] - first_defect_far[0])**2 +
                                 (third_defect_far[1] - first_defect_far[1])**2))
            rectified_order = np.array([[0, 0], [length, 0], [length, length], [0, length]], dtype=np.float32)
            transform_matrix = cv2.getPerspectiveTransform(
                np.float32([first_defect_far, third_defect_far, (0, 0), (0, 0)]), rectified_order
            )
            rectified_image = cv2.warpPerspective(image, transform_matrix, (length, length))
        else:
            rectified_image = cropped_image

        rectified_image_equalized = cv2.equalizeHist(rectified_image)

        # Apply Gabor filtering
        g_kernel_size, g_sigma, g_theta, g_lambda, g_gamma, g_psi = 5, 2.5, np.pi/3, 8.0, 0.4, 0.0
        gabor_kernel = cv2.getGaborKernel((g_kernel_size, g_kernel_size), g_sigma, g_theta, g_lambda, g_gamma, g_psi, ktype=cv2.CV_32F)
        filtered_veins = cv2.filter2D(rectified_image_equalized, cv2.CV_32F, gabor_kernel)
        filtered_veins = cv2.normalize(filtered_veins, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # CLAHE and thresholding
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        clahe_veins = clahe.apply(filtered_veins)
        _, binary_veins = cv2.threshold(clahe_veins, 110, 255, cv2.THRESH_BINARY)

        return binary_veins

    def generate_batches(self, person_id, hand, spectrum):
        images = self._load_images(person_id, hand, spectrum)
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size]
            preprocessed_batch = [self._preprocess_image(img) for img in batch]
            yield np.stack(preprocessed_batch, axis=0)
