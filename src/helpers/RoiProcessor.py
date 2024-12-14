import cv2
import numpy as np

class RoiProcessor:
    
    @staticmethod
    def apply_perspective_transform(image, transform_matrix, length):
        rectified_image = cv2.warpPerspective(image, transform_matrix, (length, length))
        return cv2.equalizeHist(rectified_image)

    @staticmethod
    def apply_gabor_filter(image):
        g_kernel_size, g_sigma, g_theta, g_lambda, g_gamma, g_psi = (
            5,
            2.5,
            np.pi / 3,
            8.0,
            0.4,
            0.0,
        )
        gabor_kernel = cv2.getGaborKernel(
            (g_kernel_size, g_kernel_size),
            g_sigma,
            g_theta,
            g_lambda,
            g_gamma,
            g_psi,
            ktype=cv2.CV_32F,
        )
        filtered_image = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
        return cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    @staticmethod
    def enhance_contrast(image):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        enhanced_image = clahe.apply(image)
        enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        for tile_size in [(4, 4), (8, 8), (10, 10)]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
            enhanced_image = clahe.apply(enhanced_image)
        return enhanced_image

    @staticmethod
    def binarize_image(image):
        _, binary_image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        return binary_image
    
    @staticmethod
    def resize_image(image):
        return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)