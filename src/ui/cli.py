from models.finetune_osnet import FinetuneOSNet

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
            print("4. OSNET Training")
            print("5. Exit")

            choice = input("Select an option: ").strip()

            if choice == "1":
                self.system.model_manager.create_model()
            elif choice == "2":
                self.system.person_follower.follow_person()
            elif choice == "3":
                self.system.model_manager.delete_model()
            elif choice == "4":
                finetune_osnet = FinetuneOSNet(self.system)
                finetune_osnet.run()
            elif choice == "5":
                self.system.cleanup()
                print("Goodbye.")
                break
            else:
                print("Invalid choice.")        

        