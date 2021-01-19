import cv2

def draw_faces(img, faces):
    """
    Draw faces on image
    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
    faces : List of face coordinates
        Coordinates of faces to draw
    Returns
    -------
    None.
    """
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)