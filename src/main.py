from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# DDS must be initialized BEFORE any Unitree SDK component
ChannelFactoryInitialize(0)

from config.settings import Settings
from config.robot_params import RobotParams

from core.detection import PersonDetector
from core.recognition import FeatureExtractor, PersonRecognizer
from core.tracking import PersonTracker
from core.visualization import Visualizer

from hardware.camera import Camera
from hardware.robot import RobotController

from models.model_manager import ModelManager

from ui.gui import GUIInterface

class PersonFollowingSystem:
    def __init__(self):
        self.settings = Settings()
        self.settings.robot_params = RobotParams()

        self.camera = Camera(self.settings)
        self.detector = PersonDetector(self.settings)

        self.feature_extractor = FeatureExtractor(self.settings)
        self.model_manager = ModelManager(self.settings)
        self.model_manager.set_feature_extractor(self.feature_extractor)

        self.recognizer = PersonRecognizer(self.feature_extractor, self.model_manager)
        self.tracker = PersonTracker(self.detector, self.recognizer)
        self.visualizer = Visualizer()

        self.robot = RobotController(self.settings.robot_params)

    def cleanup(self):
        self.camera.release()
        self.robot.stop()

if __name__ == "__main__":
    system = PersonFollowingSystem()
    gui = GUIInterface(system)
    gui.start()
from config.settings import Settings
from config.robot_params import RobotParams

from core.detection import PersonDetector
from core.recognition import FeatureExtractor, PersonRecognizer
from core.tracking import PersonTracker
from core.visualization import Visualizer

from hardware.camera import Camera
from hardware.robot import RobotController

from models.model_manager import ModelManager

from ui.gui import CommandLineInterface

class PersonFollowingSystem:
    def __init__(self):
        self.settings = Settings()
        self.settings.robot_params = RobotParams()

        self.camera = Camera(self.settings)
        self.detector = PersonDetector(self.settings)

        self.feature_extractor = FeatureExtractor(self.settings)
        self.model_manager = ModelManager(self.settings)
        self.model_manager.set_feature_extractor(self.feature_extractor)

        self.recognizer = PersonRecognizer(self.feature_extractor, self.model_manager)
        self.tracker = PersonTracker(self.detector, self.recognizer)
        self.visualizer = Visualizer()

        self.robot = RobotController(self.settings.robot_params)

    def cleanup(self):
        self.camera.release()
        self.robot.stop()

if __name__ == "__main__":
    system = PersonFollowingSystem()
    gui = CommandLineInterface(system)
    gui.start()
