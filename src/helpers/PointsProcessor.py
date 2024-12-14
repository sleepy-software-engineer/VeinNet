import numpy as np

class PointsProcessor:
    @staticmethod
    def compute_midpoint_and_direction(point1, point2):
        midpoint = (
            int((point1[0] + point2[0]) // 2),
            int((point1[1] + point2[1]) // 2),
        )
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length != 0:
            dx /= length
            dy /= length
        return midpoint, dx, dy, length

    @staticmethod
    def compute_perpendicular_point(midpoint, dx, dy, distance=50):
        x_perpendicular = int(midpoint[0] + distance * dy)
        y_perpendicular = int(midpoint[1] - distance * dx)
        return x_perpendicular, y_perpendicular

    @staticmethod
    def calculate_length(first_defect_far, third_defect_far):
        return int(
            np.sqrt(
                (third_defect_far[0] - first_defect_far[0]) ** 2
                + (third_defect_far[1] - first_defect_far[1]) ** 2
            )
        )

    @staticmethod
    def calculate_midpoint(midpoint):
        return (int(midpoint[0]), int(midpoint[1]))
    
    @staticmethod
    def process_perpendicular_point(midpoint, perpendicular_point):
        x, y = perpendicular_point
        dx = x - midpoint[0]
        dy = y - midpoint[1]
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length
        return dx, dy