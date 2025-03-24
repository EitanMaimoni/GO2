import time
from unitree_sdk2py.go2.sport.sport_client import SportClient

class RobotMovement:
    def __init__(self):
        # Initialize SportClient for robot movement
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        # Margin of error for angle (in degrees)
        self.angle_margin = 5.0

        # Distance threshold to stop walking (in meters)
        self.distance_threshold = 1.0  # Stop when the robot is 1 meter away from the person

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
            self.sport_client.StopMove()

    def rotate(self, angle):
        """
        Rotate the robot based on the detected angle.
        """
        # Rotate left or right based on the angle
        if angle > 0:
            print("Rotating left...")
            self.sport_client.Move(0, 0, -0.5)  # Rotate left (negative angular velocity)
        else:
            print("Rotating right...")
            self.sport_client.Move(0, 0, 0.5)  # Rotate right (positive angular velocity)

    def walk_forward(self, distance):
        """
        Walk forward.
        """
        print("Walking forward...")
        # Move forward based on the distance.
        speed = distance * 0.5 
        self.sport_client.Move(speed, 0, 0)
    
    def stop(self):
        """
        Stop all movement.
        """
        print("Stopping all movement.")
        self.sport_client.StopMove()

    def stand_up(self):
        """
        Make the robot stand up.
        """
        print("Standing up...")
        self.sport_client.StandUp()

    def stand_down(self):
        """
        Make the robot stand down.
        """
        print("Standing down...")
        self.sport_client.StandDown()