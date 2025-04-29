# system/person_following.py

class PersonFollowingSystem:
    def __init__(self):
        self.settings = None
        self.camera = None
        self.robot = None
        self.detector = None
        self.feature_extractor = None
        self.model_manager = None
        self.tracker = None
        self.visualizer = None
        self.gui = None

    def initialize(self):
        self.settings = self.init_settings()
        self.camera = self.init_camera()
        self.robot = self.init_robot()
        self.detector = self.init_detector()
        self.feature_extractor = self.init_feature_extractor()
        self.model_manager = self.init_model_manager()
        self.tracker = self.init_tracker()
        self.visualizer = self.init_visualizer()

    def init_settings(self):
        from config.settings import Settings
        from config.robot_params import RobotParams
        settings = Settings()
        settings.robot_params = RobotParams()
        return settings

    def init_camera(self):
        from hardware.camera import Camera
        return Camera(self.settings)

    def init_robot(self):
        from hardware.robot import RobotController
        return RobotController(self.settings.robot_params)

    def init_detector(self):
        from core.detection import PersonDetector
        return PersonDetector(self.settings)

    def init_feature_extractor(self):
        from core.feature_extractor import FeatureExtractor
        return FeatureExtractor(self.settings)

    def init_model_manager(self):
        from models.model_manager import ModelManager
        manager = ModelManager(self.settings)
        manager.set_feature_extractor(self.feature_extractor)
        return manager
    
    def init_tracker(self):
        from core.tracking import PersonTracker
        return PersonTracker(self.detector, self.feature_extractor, self.settings.similarity_threshold)

    def init_visualizer(self):
        from core.visualization import Visualizer
        return Visualizer()

    def attach_gui(self):
        from ui.cli import CLIInterface
        self.ui = CLIInterface(self)
        return self.ui

    def cleanup(self):
        if self.camera:
            self.camera.release()
        if self.robot:
            self.robot.stop()
