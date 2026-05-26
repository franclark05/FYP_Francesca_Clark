import numpy as np

LINE_DISTANCE_THRESHOLD = 3 


def point_to_segment_distance(point, segment):
    #Compute distance from a point to a LINE SEGMENT (not infinite line)
    px, py = point
    (x1, y1), (x2, y2) = segment

    line_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])

    line_len = np.dot(line_vec, line_vec)

    if line_len == 0:
        return np.linalg.norm(point_vec)

    #Project point onto segment, clamp between 0 and 1
    t = np.dot(point_vec, line_vec) / line_len
    t = max(0, min(1, t))

    projection = np.array([x1, y1]) + t * line_vec

    return np.linalg.norm(np.array([px, py]) - projection)


def check_line_fault(foot_keypoints, court_lines):
    """
    court_lines: list of (segment, abc)
    segment = ((x1,y1),(x2,y2))
    """

    if not foot_keypoints or not court_lines:
        return False

    for foot in foot_keypoints:
        for part in ["heel", "toe"]:
            pt = foot[part]

            if pt is None:
                continue

            px, py = pt

            for (segment, _) in court_lines:

                (x1, y1), (x2, y2) = segment

                #filter 1: X-range constraint avoids infinite extension sideways
                if px < min(x1, x2) - 20 or px > max(x1, x2) + 20:
                    continue

                #filter 2 Y-position constraint avoids line passing "through body"
                if py < min(y1, y2) - 10:
                    continue

                # Distance to segment
                dist = point_to_segment_distance(pt, segment)

                if dist < LINE_DISTANCE_THRESHOLD:
                    return True

    return False