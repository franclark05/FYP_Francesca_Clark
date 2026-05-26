from config import SERVICE_HEIGHT_PIXELS


def check_shuttle_height(shuttle_centroid):
    """
    Checks whether the shuttle is above the legal service height.

    Parameters
    ----------
    shuttle_centroid : tuple
        (cx, cy) coordinates of shuttle.

    Returns
    -------
    fault : bool
        True if shuttle is above legal height.
    """

    if shuttle_centroid is None:
        return False

    _, cy = shuttle_centroid

    # In images, smaller y = higher in frame
    if cy < SERVICE_HEIGHT_PIXELS:
        return True

    return False