import cv2
import numpy as np

from helpers.ContourProcessor import ContourProcessor
from helpers.PointsProcessor import PointsProcessor


class SquareProcessor:
    """
    A class used to process squares in an image.
    """

    @staticmethod
    def generate_square_vertices(perpendicular_point, length, offset=50):
        """
        Generate the vertices of a square given a perpendicular point, length, and offset.
        """
        # Unpack the perpendicular point into x and y coordinates
        x, y = perpendicular_point
        # Define the vertices of the square based on the given length and offset
        vertices = [
            (x + offset, y),
            (x + offset, y - length),
            (x + offset + length, y - length),
            (x + offset + length, y),
        ]
        # Return the list of vertices
        return vertices

    @staticmethod
    def rotate_square(vertices, midpoint, angle):
        """
        Rotate the square vertices around a midpoint by a given angle.
        """
        # Get the rotation matrix for the given midpoint and angle
        rotation_matrix = cv2.getRotationMatrix2D(midpoint, angle, scale=1)
        # Apply the rotation matrix to the vertices and convert to integer type
        rotated_vertices = (
            cv2.transform(np.array([vertices], dtype=np.float32), rotation_matrix)
            .squeeze()
            .astype(np.int32)
        )
        # Return the rotated vertices
        return rotated_vertices

    @staticmethod
    def translate_square(vertices, start_point):
        """
        Translate the square vertices by a given start point.
        """
        # Translate the vertices by adding the start point
        return vertices + start_point

    @staticmethod
    def process_translation(dx, dy, vertices_translated):
        """
        Process the translation of the square vertices by given dx and dy.
        """
        # Calculate the translation vector based on dx and dy
        translation = (int(50 * dx), int(50 * dy))
        # Apply the translation to the vertices and convert to float type
        translated = np.array(vertices_translated + translation, dtype=np.float32)
        # Return the translated vertices
        return translated

    @staticmethod
    def compute_transform(vertices, length):
        """
        Compute the perspective transform matrix for the square vertices.
        """
        # Define the rectified order of the square vertices
        rectified_order = np.array(
            [[0, 0], [length, 0], [length, length], [0, length]], dtype=np.float32
        )
        # Convert the vertices to float type
        vertices = np.array(vertices, dtype=np.float32)
        # Compute the perspective transform matrix
        return cv2.getPerspectiveTransform(vertices, rectified_order)

    @staticmethod
    def square_area(largest_contour):
        """
        Calculate the area of the square based on the largest contour.
        """
        # Find the defects in the largest contour
        defects = ContourProcessor.find_defects(largest_contour)

        # Get the far points from the defects
        far_points = ContourProcessor.get_far_points(defects, largest_contour)

        # Extract the first and third far points
        first_defect_far, third_defect_far = far_points[0], far_points[2]

        # Compute the midpoint, direction, and length between the first and third far points
        midpoint, dx, dy = PointsProcessor.compute_midpoint_and_direction(
            first_defect_far, third_defect_far
        )

        # Compute the perpendicular point from the midpoint and direction
        perpendicular_point = PointsProcessor.compute_perpendicular_point(
            midpoint, dx, dy
        )

        # Calculate the length between the first and third far points
        length = PointsProcessor.calculate_length(first_defect_far, third_defect_far)

        # Generate the vertices of the square
        square_vertices = SquareProcessor.generate_square_vertices(
            perpendicular_point, length
        )

        # Calculate the angle of rotation based on the direction
        angle = np.arctan2(-dy, dx) * 180 / np.pi
        # Rotate the square vertices around the midpoint by the calculated angle
        rotated_vertices = SquareProcessor.rotate_square(
            square_vertices, midpoint, angle
        )

        # Calculate the start point for translation
        start_point = (
            first_defect_far[0] - rotated_vertices[0][0],
            first_defect_far[1] - rotated_vertices[0][1],
        )
        # Translate the rotated vertices by the start point
        translated_vertices = SquareProcessor.translate_square(
            rotated_vertices, start_point
        )

        # Process the perpendicular point translation
        dx, dy = PointsProcessor.process_perpendicular_point(
            midpoint, perpendicular_point
        )

        # Apply the translation to the vertices
        translated_vertices = SquareProcessor.process_translation(
            dx, dy, translated_vertices
        )

        # Compute the perspective transform matrix for the translated vertices
        square_area = SquareProcessor.compute_transform(translated_vertices, length)

        # Return the perspective transform matrix and the length of the square sides
        return square_area, length
