import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')

        # Set up QoS profile to match the publisher
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Match BEST_EFFORT policy
            history=QoSHistoryPolicy.KEEP_LAST,          # Keep last 10 messages
            depth=10
        )

        # Create subscriber with the custom QoS profile
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile=qos_profile
        )
        self.subscription  # Prevent unused variable warning
        self.bridge = CvBridge()

        # Publisher for /cmd_vel to control the robot's rotation
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Load YOLO model for person detection (use YOLOv4-tiny for better performance)
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()

        # Handle different OpenCV versions for getUnconnectedOutLayers()
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        if unconnected_out_layers.ndim == 2:  # OpenCV returns a 2D array
            self.output_layers = [self.layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:  # OpenCV returns a 1D array
            self.output_layers = [self.layer_names[i - 1] for i in unconnected_out_layers]

        # Camera parameters (update these based on your camera)
        self.fov = 70  # Field of view in degrees (approximate, adjust based on your camera)
        self.known_person_height = 1.7  # Average person height in meters
        self.focal_length = 600  # Focal length in pixels (approximate, adjust based on your camera)

        # Variables to store the latest distance and angle
        self.latest_distance = 0.0
        self.latest_angle = 0.0
        self.person_detected = False  # Flag to track if a person is detected

        # Timer for text updates
        self.last_update_time = time.time()
        self.update_interval = 0.5  # Update text every 0.5 seconds

    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return  # Skip processing if interval hasn't passed
        
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Detect person and calculate distance/angle
        self.detect_person(cv_image)

        # Display the image with distance and angle
        self.display_info(cv_image)

        # Rotate the robot to center the person
        self.rotate_robot()

        # Refresh the display window
        cv2.imshow("Camera Image", cv_image)
        cv2.waitKey(1)

    def detect_person(self, image):
        height, width, channels = image.shape
        self.person_detected = False  # Reset detection flag

        # Detect objects using YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # Reduced resolution
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] == 'person':
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Draw bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate distance and angle
                    self.latest_distance = (self.known_person_height * self.focal_length) / h
                    offset_x = center_x - (width / 2)
                    self.latest_angle = (offset_x / (width / 2)) * (self.fov / 2)

                    # Set person detected flag
                    self.person_detected = True

    def display_info(self, image):
        # Display distance and angle at the lower middle of the screen
        height, width, _ = image.shape
        text_x = width // 2 - 100
        text_y = height - 50

        # Clear previous text by drawing a filled rectangle
        cv2.rectangle(image, (text_x - 10, text_y - 30), (text_x + 200, text_y + 10), (0, 0, 0), -1)

        # Display distance and angle
        cv2.putText(image, f"Distance: {self.latest_distance:.2f} m", (text_x, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"Angle: {self.latest_angle:.2f} deg", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def rotate_robot(self):
        # Create a Twist message to rotate the robot
        twist_msg = Twist()

        if self.person_detected:
            # If the person is detected, rotate to center them
            if abs(self.latest_angle) > 6:  # Adjust the threshold as needed
                twist_msg.angular.z = -0.25 if self.latest_angle > 0 else 0.25 # Increase rotation speed
            else:
                twist_msg.angular.z = 0.0  # Stop rotating when centered
        else:
            # If no person is detected, stop rotating
            twist_msg.angular.z = 0.0

        # Publish the Twist message
        self.cmd_vel_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    image_viewer = ImageViewer()
    rclpy.spin(image_viewer)
    image_viewer.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()