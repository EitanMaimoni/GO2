#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # Assuming the message type is PoseStamped


class RobotPoseSubscriber(Node):
    def __init__(self):
        super().__init__("robot_pose_subscriber")
        self.get_logger().info("Robot Pose Subscriber Node Started.")
        
        # Subscribe to the topic
        self.create_subscription(
            PoseStamped,
            "/utlidar/robot_pose",
            self.pose_callback,
            10  # QoS history depth
        )

    def pose_callback(self, msg: PoseStamped):
        # Extract x, y, z, w
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        w = msg.pose.orientation.w

        # Print with padding to align values under each other
        self.get_logger().info(
            f"Position:\n"
            f"  x = {x:.6f}\n"
            f"  y = {y:.6f}\n"
            f"  z = {z:.6f}\n"
            f"  w = {w:.6f}\n"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RobotPoseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
