import signal
import sys
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

class RobotController:
    """Controls robot movement based on tracking information."""
    
    def __init__(self, robot_params):
        """
        Initialize robot controller.
        
        Args:
            robot_params: Robot parameters object
        """
        # Initialize ObstaclesAvoidClient for obstacle avoidance
        self.obstacles_avoid_client = ObstaclesAvoidClient()
        self.obstacles_avoid_client.SetTimeout(10.0)
        self.obstacles_avoid_client.Init()
        self.obstacles_avoid_client.UseRemoteCommandFromApi(True)

        print("Enabling obstacle avoidance mode...")
        self.obstacles_avoid_client.SwitchSet(True)
        self.obstacles_avoid_client.Move(0, 0, 0)

        # Robot parameters
        self.params = robot_params

        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """
        Handle Ctrl+C signal.
        """
        print("\nCtrl+C detected. Stopping all movement and exiting...")
        self.stop()
        sys.exit(0)

    def follow_target(self, angle, distance):
        """
        Follow target based on angle and distance.
        
        Args:
            angle: Angle to target (degrees)
            distance: Distance to target (meters)
        """
        # if abs(angle) > self.params.angle_threshold and distance > self.params.distance_threshold:
        #     print("Rotating to face the person and walking towards them...")
        #     self._rotate_and_walk_forward(angle, distance)
        # elif abs(angle) > self.params.angle_threshold:
        #     print("Rotating to face the person...")
        #     self._rotate(angle)
        # elif distance > self.params.distance_threshold:
        #     print("Walking towards the person...")
        #     self._walk_forward(distance)
        # else:
        #     print("Reached the person. Stopping all movement.")
        #     self.obstacles_avoid_client.Move(0, 0, 0)

    def _rotate(self, angle):
        """
        Rotate the robot based on the detected angle.
        
        Args:
            angle: Angle to rotate (degrees)
        """
        # Rotate left or right based on the angle
        if angle > 0:
            print("Rotating left...")
            self.obstacles_avoid_client.Move(0, 0, -self.params.rotation_speed)  # Rotate left (negative angular velocity)
        else:
            print("Rotating right...")
            self.obstacles_avoid_client.Move(0, 0, self.params.rotation_speed)  # Rotate right (positive angular velocity)

    def _walk_forward(self, distance):
        """
        Walk forward.
        
        Args:
            distance: Distance to target (meters)
        """
        print("Walking forward...")
        # Move forward based on the distance.
        speed = min(distance * self.params.forward_speed_factor, 0.5)  # Cap max speed
        self.obstacles_avoid_client.Move(speed, 0, 0)
    
    def _rotate_and_walk_forward(self, angle, distance):
        """
        Rotate and walk forward simultaneously.
        
        Args:
            angle: Angle to target (degrees)
            distance: Distance to target (meters)
        """
        speed = min(distance * self.params.forward_speed_factor, 0.5)  # Lower max speed when combined

        if angle > 0:
            print("Rotating left and moving forward...")
            self.obstacles_avoid_client.Move(speed, 0, -self.params.rotation_speed)
        else:
            print("Rotating right and moving forward...")
            self.obstacles_avoid_client.Move(speed, 0, self.params.rotation_speed)

    def stop(self):
        """Stop all movement."""
        print("Stopping all movement.")
        self.obstacles_avoid_client.Move(0, 0, 0)  # Stop obstacle avoidance movement
        self.obstacles_avoid_client.UseRemoteCommandFromApi(False)