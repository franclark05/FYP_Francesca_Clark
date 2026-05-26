import numpy as np


def line_to_abc(x1, y1, x2, y2):
    #Convert two points into line equation ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - y2 * x1
    return a, b, c


def point_line_distance(point, line_abc):
    #Compute perpendicular distance from point to line.
    x, y = point
    a, b, c = line_abc
    return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)


def is_point_on_segment(point, segment, tolerance=5):
    #check if projection lies within segment bounds.
    (x1, y1), (x2, y2) = segment
    px, py = point

    return (
        min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance
        and
        min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance
    )

def line_from_points(x1, y1, x2, y2):
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return (a, b, c)
