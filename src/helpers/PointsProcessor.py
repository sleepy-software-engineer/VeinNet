import numpy as np


class PointsProcessor:
    """
    A class used to process points and compute various geometric properties.
    """

    @staticmethod
    def compute_midpoint_and_direction(point1, point2):
        """
        Compute the midpoint and direction vector between two points.
        """
        # Calculate the midpoint between point1 and point2
        midpoint = (
            int((point1[0] + point2[0]) // 2),
            int((point1[1] + point2[1]) // 2),
        )
        # Calculate the difference in x and y coordinates
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        # Calculate the length of the vector
        length = np.sqrt(dx**2 + dy**2)
        # Normalize the direction vector if the length is not zero
        if length != 0:
            dx /= length
            dy /= length
        # Return the midpoint, direction vector, and length
        return midpoint, dx, dy

    @staticmethod
    def compute_perpendicular_point(midpoint, dx, dy, distance=50):
        """
        Compute a point perpendicular to the direction vector at a given distance from the midpoint.
        """
        # Calculate the x coordinate of the perpendicular point
        x_perpendicular = int(midpoint[0] + distance * dy)
        # Calculate the y coordinate of the perpendicular point
        y_perpendicular = int(midpoint[1] - distance * dx)
        # Return the coordinates of the perpendicular point
        return x_perpendicular, y_perpendicular

    @staticmethod
    def calculate_length(first_defect_far, third_defect_far):
        """
        Calculate the length between two points.
        """
        # Calculate the length between first_defect_far and third_defect_far
        return int(
            np.sqrt(
                (third_defect_far[0] - first_defect_far[0]) ** 2
                + (third_defect_far[1] - first_defect_far[1]) ** 2
            )
        )

    @staticmethod
    def process_perpendicular_point(midpoint, perpendicular_point):
        """
        Process a perpendicular point to compute the direction vector.
        """
        # Extract the x and y coordinates of the perpendicular point
        x, y = perpendicular_point
        # Calculate the difference in x and y coordinates from the midpoint
        dx = x - midpoint[0]
        dy = y - midpoint[1]
        # Calculate the length of the vector
        length = np.sqrt(dx**2 + dy**2)
        # Normalize the direction vector
        dx /= length
        dy /= length
        # Return the normalized direction vector
        return dx, dy
