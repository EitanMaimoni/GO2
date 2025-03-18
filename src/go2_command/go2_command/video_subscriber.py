#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from unitree_go.msg import Go2FrontVideoData


class MyNode(Node):

    def __init__(self):
        super().__init__("video_subscriber")
        self.get_logger().info("Video subscriber node started.")
        
        # Subscription to the /frontvideostream topic
        self.create_subscription(
            Go2FrontVideoData,
            "/frontvideostream",  # Correct topic name
            self.video_callback,
            10  # QoS history depth
        )

    def video_callback(self, msg: Go2FrontVideoData):
        # Log information about the video data received
        self.get_logger().info(f"Received video data with frame number: {msg.frame_num}, "
                               f"frame width: {msg.frame_width}, frame height: {msg.frame_height}")


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
