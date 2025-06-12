import cv2
import numpy as np
import time

class CLIInterface:
    def __init__(self, system):
        self.system = system

    # TODO: ADD option to delete model
    def start(self):
        while True:
            print("\n===== Person Tracker CLI =====")
            print("1. Create Person Model")
            print("2. Follow Person")
            print("3. Delete Model")
            print("4. Exit")

            choice = input("Select an option: ").strip()

            if choice == "1":
                self.system.model_manager.create_model()
            elif choice == "2":
                self.system.person_follower.follow_person()
            elif choice == "3":
                self.system.model_manager.delete_model()
            elif choice == "4":
                self.system.cleanup()
                print("Goodbye.")
                break
            else:
                print("Invalid choice.")        

        