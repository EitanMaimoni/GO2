import signal
import sys
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

class RobotMovement:
    def __init__(self):
        # Initialize ObstaclesAvoidClient for obstacle avoidance
        self.obstacles_avoid_client = ObstaclesAvoidClient()
        self.obstacles_avoid_client.SetTimeout(10.0)
        self.obstacles_avoid_client.Init()
        self.obstacles_avoid_client.UseRemoteCommandFromApi(True)

        print("Enabling obstacle avoidance mode...")
        self.obstacles_avoid_client.SwitchSet(True)
        self.obstacles_avoid_client.Move(0, 0, 0)

        # Margin of error for angle (in degrees)
        self.angle_margin = 5.0

        # Distance threshold to stop walking (in meters)
        self.distance_threshold = 1.0  # Stop when the robot is 1 meter away from the person

        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """
        Handle Ctrl+C signal.
        """
        print("\nCtrl+C detected. Stopping all movement and exiting...")
        self.stop()
        sys.exit(0)

    def walk_towards_target(self, angle, distance):
        """
        Walk towards the target (person) based on the angle and distance.
        """
        if abs(angle) > self.angle_margin:
            print("Rotating to face the person...")
            self.rotate(angle)
        elif distance > self.distance_threshold:
            print("Walking towards the person...")
            self.walk_forward(distance)
        else:
            print("Reached the person. Stopping all movement.")
            self.obstacles_avoid_client.Move(0, 0, 0)  # Stop obstacle avoidance movement

    def rotate(self, angle):
        """
        Rotate the robot based on the detected angle.
        """
        # Rotate left or right based on the angle
        if angle > 0:
            print("Rotating left...")
            self.obstacles_avoid_client.Move(0, 0, -0.5)  # Rotate left (negative angular velocity)
        else:
            print("Rotating right...")
            self.obstacles_avoid_client.Move(0, 0, 0.5)  # Rotate right (positive angular velocity)

    def walk_forward(self, distance):
        """
        Walk forward.
        """
        print("Walking forward...")
        # Move forward based on the distance.
        speed = distance * 0.5 
        self.obstacles_avoid_client.Move(speed, 0, 0)
    
    def stop(self):
        """
        Stop all movement.
        """
        print("Stopping all movement.")
        self.obstacles_avoid_client.Move(0, 0, 0)  # Stop obstacle avoidance movement
        self.obstacles_avoid_client.UseRemoteCommandFromApi(False)
