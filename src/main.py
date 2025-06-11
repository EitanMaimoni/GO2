# TODO: Remove statics method after testing
import cProfile
import pstats
import io
from system.person_following import PersonFollowingSystem
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

def main():
    ChannelFactoryInitialize(0)
    system = PersonFollowingSystem()
    system.ui.start()

if __name__ == "__main__":
    main()

