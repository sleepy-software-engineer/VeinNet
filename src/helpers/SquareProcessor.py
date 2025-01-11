import cv2
import numpy as np

from helpers.ContourProcessor import ContourProcessor
from helpers.PointsProcessor import PointsProcessor


class SquareProcessor:
    @staticmethod
    def generate_square_vertices(perpendicular_point, length, offset=50):
        x, y = perpendicular_point
        vertices = [
            (x + offset, y),
            (x + offset, y - length),
            (x + offset + length, y - length),
            (x + offset + length, y),
        ]
        return vertices

    @staticmethod
    def rotate_square(vertices, midpoint, angle):
        rotation_matrix = cv2.getRotationMatrix2D(midpoint, angle, scale=1)
        rotated_vertices = (
            cv2.transform(np.array([vertices], dtype=np.float32), rotation_matrix)
            .squeeze()
            .astype(np.int32)
        )
        return rotated_vertices

    @staticmethod
    def translate_square(vertices, start_point):
        return vertices + start_point

    @staticmethod
    def process_translation(dx, dy, vertices_translated):
        translation = (int(50 * dx), int(50 * dy))
        translated = np.array(vertices_translated + translation, dtype=np.float32)
        return translated

    @staticmethod
    def compute_transform(vertices, length):
        rectified_order = np.array(
            [[0, 0], [length, 0], [length, length], [0, length]], dtype=np.float32
        )
        vertices = np.array(vertices, dtype=np.float32)
        return cv2.getPerspectiveTransform(vertices, rectified_order)

    @staticmethod
    def square_area(largest_contour):
        defects = ContourProcessor.find_defects(largest_contour)
        far_points = ContourProcessor.get_far_points(defects, largest_contour)
        first_defect_far, third_defect_far = far_points[0], far_points[2]
        midpoint, dx, dy = PointsProcessor.compute_midpoint_and_direction(
            first_defect_far, third_defect_far
        )
        perpendicular_point = PointsProcessor.compute_perpendicular_point(
            midpoint, dx, dy
        )
        length = PointsProcessor.calculate_length(first_defect_far, third_defect_far)
        square_vertices = SquareProcessor.generate_square_vertices(
            perpendicular_point, length
        )
        angle = np.arctan2(-dy, dx) * 180 / np.pi
        rotated_vertices = SquareProcessor.rotate_square(
            square_vertices, midpoint, angle
        )
        start_point = (
            first_defect_far[0] - rotated_vertices[0][0],
            first_defect_far[1] - rotated_vertices[0][1],
        )
        translated_vertices = SquareProcessor.translate_square(
            rotated_vertices, start_point
        )
        dx, dy = PointsProcessor.process_perpendicular_point(
            midpoint, perpendicular_point
        )
        translated_vertices = SquareProcessor.process_translation(
            dx, dy, translated_vertices
        )
        square_area = SquareProcessor.compute_transform(translated_vertices, length)
        return square_area, length
