import cv2

class Visualizer:
    """Handles visualization of detection and tracking results."""
    
    def __init__(self):
        """Initialize visualizer."""
        pass
    
    def draw_tracking_info(self, image, distance=0, angle=0):
        """
        Add distance and angle information to image.
        
        Args:
            image: Input image
            distance: Distance to target (meters)
            angle: Angle to target (degrees)
            
        Returns:
            numpy.ndarray: Image with added text
        """
        if image is None or image.size == 0:
            return image
            
        # Convert distance and angle to strings
        distance_text = f"Distance: {distance:.2f} m"
        angle_text = f"Angle: {angle:.2f} deg"
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Set text position (middle-lower part of the image)
        text_x = width // 2 - 100  # Centered horizontally
        text_y = height - 30      # 30 pixels from the bottom
        
        # Set font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 255, 0)  # Green color
        thickness = 2
        
        # Put distance text
        cv2.putText(image, distance_text, (text_x, text_y - 30), 
                font, font_scale, font_color, thickness)
        
        # Put angle text
        cv2.putText(image, angle_text, (text_x, text_y), 
                font, font_scale, font_color, thickness)
                
        return image