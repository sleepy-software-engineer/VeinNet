from helpers.ContourProcessor import ContourProcessor
from helpers.RoiProcessor import RoiProcessor
from helpers.SquareProcessor import SquareProcessor


class DataProcessor:
    def preprocess_image(image_path: str) -> any:
        cropped_image = ContourProcessor.load_and_crop_image(image_path)

        largest_contour = ContourProcessor.preprocess_contours(cropped_image)

        square_area, length = SquareProcessor.square_area(largest_contour)

        rectified_image = RoiProcessor.apply_perspective_transform(
            cropped_image, square_area, length
        )

        gabor_filtered = RoiProcessor.apply_gabor_filter(rectified_image)

        enhanced_image = RoiProcessor.enhance_contrast(gabor_filtered)

        binary_image = RoiProcessor.binarize_image(enhanced_image)

        output = RoiProcessor.resize_image(binary_image)

        return output
