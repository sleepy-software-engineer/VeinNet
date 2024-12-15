from helpers.ContourProcessor import ContourProcessor
from helpers.RoiProcessor import RoiProcessor
from helpers.SquareProcessor import SquareProcessor


class DataProcessor:
    """
    DataProcessor class provides static methods to preprocess images for biometric systems.
    The preprocessing steps include loading and cropping the image, contour processing,
    square area calculation, perspective transformation, Gabor filtering, contrast enhancement,
    binarization, and resizing.
    """

    @staticmethod
    def preprocess_image(image_path: str) -> any:
        """
        Preprocesses the image located at the given path by performing a series of image processing steps.
        """
        # Load and crop the image using ContourProcessor
        cropped_image = ContourProcessor.load_and_crop_image(image_path)

        # Preprocess contours to find the largest contour in the cropped image
        largest_contour = ContourProcessor.preprocess_contours(cropped_image)

        # Calculate the square area and its side length from the largest contour
        square_area, length = SquareProcessor.square_area(largest_contour)

        # Apply perspective transformation to the cropped image using the square area and length
        rectified_image = RoiProcessor.apply_perspective_transform(
            cropped_image, square_area, length
        )

        # Apply Gabor filter to the rectified image for texture enhancement
        gabor_filtered = RoiProcessor.apply_gabor_filter(rectified_image)

        # Enhance the contrast of the Gabor filtered image
        enhanced_image = RoiProcessor.enhance_contrast(gabor_filtered)

        # Binarize the contrast-enhanced image
        binary_image = RoiProcessor.binarize_image(enhanced_image)

        # Resize the binarized image to a standard size
        output = RoiProcessor.resize_image(binary_image)

        return output


if __name__ == "__main__":
    # Test the DataProcessor class
    image_path = (
        "/Users/ldcrainic/University/MSc-Courses/BiometricSystems/data/001_l_940_05.jpg"
    )
    preprocessed_image = DataProcessor.preprocess_image(image_path)
    # save imag
    import cv2

    cv2.imwrite("preprocessed_image.jpg", preprocessed_image)
