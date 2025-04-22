import cv2
import numpy as np

def resize_image(image, size):
    """Resize image to the given size (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def convert_to_grayscale(image):
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def overlay_text(image, text, position=(10, 30), color=(0, 255, 0), scale=0.7, thickness=2):
    """Overlay text on the image at a specific position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, scale, color, thickness)
    return image
