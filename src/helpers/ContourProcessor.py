import cv2


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
    def load_and_crop_image(image_path):
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
