import cv2
import numpy as np


class RoiProcessor:
    """
    A class to process Regions of Interest (ROI) in images using various image processing techniques.
    """

    @staticmethod
    def apply_perspective_transform(image, transform_matrix, length):
        """
        Apply a perspective transform to the input image using the given transformation matrix and length.
        """
        # Apply perspective transformation to the input image
        rectified_image = cv2.warpPerspective(image, transform_matrix, (length, length))
        # Equalize the histogram of the rectified image
        return cv2.equalizeHist(rectified_image)

    @staticmethod
    def apply_gabor_filter(image):
        """
        Apply a Gabor filter to the input image to enhance texture features.
        """
        # Define Gabor filter parameters
        g_kernel_size, g_sigma, g_theta, g_lambda, g_gamma, g_psi = (
            5,  # Kernel size
            2.5,  # Sigma
            np.pi / 3,  # Theta
            8.0,  # Lambda
            0.4,  # Gamma
            0.0,  # Psi
        )
        # Create Gabor kernel with the specified parameters
        gabor_kernel = cv2.getGaborKernel(
            (g_kernel_size, g_kernel_size),
            g_sigma,
            g_theta,
            g_lambda,
            g_gamma,
            g_psi,
            ktype=cv2.CV_32F,
        )
        # Apply the Gabor filter to the input image
        filtered_image = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
        # Normalize the filtered image to the range [0, 255]
        return cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    @staticmethod
    def enhance_contrast(image):
        """
        Enhance the contrast of the given image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        and Gaussian blur.
        """
        # Create a CLAHE object with a clip limit of 3.0 and a tile grid size of (2, 2)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        # Apply CLAHE to the input image
        enhanced_image = clahe.apply(image)
        # Apply Gaussian blur to the enhanced image with a kernel size of (5, 5)
        enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        # Iterate over different tile sizes [(4, 4), (8, 8), (10, 10)]
        for tile_size in [(4, 4), (8, 8), (10, 10)]:
            # Create a CLAHE object with a clip limit of 2.0 and the current tile size
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
            # Apply CLAHE to the enhanced image
            enhanced_image = clahe.apply(enhanced_image)
        # Return the final enhanced image
        return enhanced_image

    @staticmethod
    def binarize_image(image):
        """
        Convert the input image to a binary image using a fixed threshold.
        """
        # Apply a fixed threshold to the input image
        _, binary_image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        # Return the binarized image
        return binary_image

    @staticmethod
    def resize_image(image):
        """
        Resize the input image to 128x128 pixels using INTER_AREA interpolation.
        """
        # Resize the image to 128x128 pixels using INTER_AREA interpolation
        return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
