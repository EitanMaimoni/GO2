# TODO: Remove statics method after testing
import cProfile
import pstats
import io
from system.person_following import PersonFollowingSystem
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

def main():
    ChannelFactoryInitialize(0)
    system = PersonFollowingSystem()
    system.initialize()
    ui = system.attach_ui()
    ui.start()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    # Output profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats("src")  
    print(s.getvalue())
