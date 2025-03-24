import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from video_client import VideoProcessor

if __name__ == "__main__":
    # Initialize DDS communication
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])  # Pass network interface as argument
    else:
        ChannelFactoryInitialize(0)  # Use default network interface

    # Paths to YOLO files
    weights_path = "../model/yolov4.weights"
    cfg_path = "../model/yolov4.cfg"
    names_path = "../model/coco.names"

    # Initialize video processor
    processor = VideoProcessor(weights_path, cfg_path, names_path)

    # Process video
    processor.process_video()