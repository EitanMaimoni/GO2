# system/person_following.py

class PersonFollowingSystem:
    def __init__(self):
        self.settings = self.init_settings()
        self.camera = self.init_camera()
        self.robot = self.init_robot()
        self.detector = self.init_detector()
        self.feature_extractor = self.init_feature_extractor()
        self.model_manager = self.init_model_manager()
        self.recognizer = self.init_recognizer()
        self.visualizer = self.init_visualizer()
        self.person_follower = self.init_person_follower()
        self.ui = self.init_ui()

    def init_settings(self):
        from config.settings import Settings
        from config.robot_params import RobotParams
        settings = Settings()
        settings.robot_params = RobotParams()
        return settings

    def init_camera(self):
        from go2_interface.camera import Camera
        return Camera(self.settings)

    def init_robot(self):
        from go2_interface.robot import RobotController
        return RobotController(self.settings.robot_params)

    def init_detector(self):
        from core.detection import PersonDetector
        return PersonDetector(self.settings)

    def init_feature_extractor(self):
        from core.feature_extractor import FeatureExtractor
        return FeatureExtractor()
    
    def init_recognizer(self):
        from core.recognition import PersonRecognition
        return PersonRecognition(self.feature_extractor, self.settings)
    
    def init_model_manager(self):
        from models.model_manager import ModelManager
        return ModelManager(self.settings, self.camera, self.detector, self.feature_extractor)

    def init_visualizer(self):
        from core.visualization import Visualizer
        return Visualizer()
    
    def init_person_follower(self):
        from core.follower import PersonFollower
        return PersonFollower(self.robot, self.recognizer, self.visualizer, self.camera, self.detector, self.model_manager)

    def init_ui(self):
        from ui.cli import CLIInterface
        self.ui = CLIInterface(self)
        return self.ui

    def cleanup(self):
        if self.robot:
            self.robot.stop()
