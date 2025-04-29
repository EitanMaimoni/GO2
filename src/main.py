from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from system.person_following import PersonFollowingSystem

def main():
    """
    Main function to initialize and run the person following system.
    """
    # Initialize the channel factory
    ChannelFactoryInitialize(0)
    
    # Create an instance of the PersonFollowingSystem
    system = PersonFollowingSystem()
    
    # Initialize the system components
    system.initialize()
    
    # Attach UI for user interaction
    ui = system.attach_gui()
    
    # Start the UI loop
    ui.start()

if __name__ == "__main__":
    main()
