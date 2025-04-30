import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import io
import sys
import numpy as np
import time

class GUIInterface:
    """Unified full-screen GUI for the person-following system with embedded control, preview, and console."""

    def __init__(self, system):
        self.system = system
        self.root = tk.Tk()
        self.root.title("AI-Powered Person Tracker")
        self.root.configure(bg="#1f1f2e")
        self.root.attributes('-fullscreen', True)

        self._build_layout()
        self.capture_mode = False
        self.latest_image = None
        self.latest_person = None

    def _build_layout(self):
        self.main_frame = tk.Frame(self.root, bg="#1f1f2e")
        self.main_frame.pack(fill="both", expand=True)

        self.preview_label = tk.Label(self.main_frame, bg="black")
        self.preview_label.place(relx=0, rely=0, relwidth=0.7, relheight=0.8)

        self.console = tk.Text(self.main_frame, bg="#0e0e13", fg="lime", font=("Consolas", 10))
        self.console.place(relx=0, rely=0.8, relwidth=0.7, relheight=0.2)

        self.control_frame = tk.Frame(self.main_frame, bg="#2a2a3d")
        self.control_frame.place(relx=0.7, rely=0, relwidth=0.3, relheight=1.0)

        ttk.Label(self.control_frame, text="Person Tracker", font=("Segoe UI", 20)).pack(pady=40)
        ttk.Button(self.control_frame, text="Create Person Model", command=self.show_create_panel).pack(pady=10)
        ttk.Button(self.control_frame, text="Follow Person", command=self.show_follow_panel).pack(pady=10)
        ttk.Button(self.control_frame, text="Exit", command=self.cleanup).pack(pady=40)

        self.input_frame = tk.Frame(self.control_frame, bg="#2a2a3d")
        self.input_frame.pack(fill="both", expand=True)

        sys.stdout = TextRedirector(self.console, "stdout")
        sys.stderr = TextRedirector(self.console, "stderr")

    def start(self):
        self.root.mainloop()

    def show_create_panel(self):
        self.clear_input_frame()
        ttk.Label(self.input_frame, text="Enter person name:", background="#2a2a3d", foreground="white").pack(pady=10)
        name_entry = ttk.Entry(self.input_frame)
        name_entry.pack(pady=5)

        def start_capture():
            name = name_entry.get().strip()
            if name:
                print(f"[INFO] Starting capture for: {name}")
                print("[INSTRUCTION] Press ENTER to save image, ESC to finish.")
                self.create_model(name)

        ttk.Button(self.input_frame, text="Start Capture", command=start_capture).pack(pady=10)

    def show_follow_panel(self):
        self.clear_input_frame()
        models = self.system.model_manager.list_models()
        if not models:
            print("[INFO] No trained models available.")
            return

        ttk.Label(self.input_frame, text="Select person to follow:", background="#2a2a3d", foreground="white").pack(pady=10)
        selected = tk.StringVar()
        dropdown = ttk.Combobox(self.input_frame, textvariable=selected, values=models)
        dropdown.pack(pady=5)

        def start_follow():
            name = selected.get()
            if name:
                print(f"[INFO] Tracking {name}...")
                self.follow_person(name)

        ttk.Button(self.input_frame, text="Start Following", command=start_follow).pack(pady=10)

    def clear_input_frame(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()

    def create_model(self, name):
        self.system.model_manager.create_dataset(name)
        self.capture_mode = True
        self.capture_count = 0

        def loop():
            while self.capture_mode:
                frame = self.system.camera.get_frame()
                if frame is None:
                    continue

                person_img, _ = self.system.detector.get_first_person(frame)
                self.latest_image = frame
                self.latest_person = person_img

                display = person_img if person_img is not None else frame
                self._update_preview(display)

        def on_key_press(event):
            if event.keysym == 'Return':
                if self.latest_person is not None:
                    self.system.model_manager.save_image(name, self.latest_person)
                    self.capture_count += 1
                    print(f"[INFO] Saved image {self.capture_count}")
                else:
                    print("[WARNING] No person detected — image not saved.")
            elif event.keysym == 'Escape':
                self.capture_mode = False
                print("[INFO] Training model... please wait.")
                try:
                    self.system.model_manager.train_model_osnet(name)
                    print(f"[SUCCESS] Model trained successfully for {name}.")
                except Exception as e:
                    print(f"[ERROR] Training failed: {e}")
                self.root.unbind("<Key>")

        self.root.bind("<Key>", on_key_press)
        threading.Thread(target=loop, daemon=True).start()

    def follow_person(self, person_name):
        try:
            gallery_path = f"{self.system.model_manager.base_dir}/{person_name}/gallery/features.npy"
            gallery_features = np.load(gallery_path)
        except Exception as e:
            print(f"[ERROR] Failed to load features: {e}")
            return

        print("[INFO] Press ESC to stop following.")
        self.capture_mode = True

        def loop():
            last_seen_time = time.time()

            while self.capture_mode:
                frame = self.system.camera.get_frame()
                if frame is None:
                    continue

                person_img, target_info = self.system.recognizer.recognize_target(frame, gallery_features)

                if person_img is not None:
                    if target_info is not None:
                        annotated = self.system.visualizer.draw_tracking_info(
                            person_img,
                            distance=target_info.get("distance", 0),
                            angle=target_info.get("angle", 0),
                            confidence=target_info.get("confidence", 0)
                        )
                        self._update_preview(annotated)
                        self.system.robot.follow_target(target_info["angle"], target_info["distance"])
                        last_seen_time = time.time()
                    else:
                        self._update_preview(person_img)
                        self.system.robot.follow_target(0, 0)
                else:
                    self._update_preview(frame)
                    self.system.robot.follow_target(0, 0)

                if time.time() - last_seen_time >= 3.0:
                    self.system.robot.follow_target(0, 0)

        def on_key_press(event):
            if event.keysym == 'Escape':
                self.capture_mode = False
                self.system.robot.follow_target(0, 0)
                print("[INFO] Stopped following.")
                self.root.unbind("<Key>")

        self.root.bind("<Key>", on_key_press)
        threading.Thread(target=loop, daemon=True).start()

    def _update_preview(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        label_width = self.preview_label.winfo_width()
        label_height = self.preview_label.winfo_height()
        if label_width > 0 and label_height > 0:
            img = img.resize((label_width, label_height), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.latest_image = image
        self.preview_label.imgtk = imgtk
        self.preview_label.configure(image=imgtk)

    def cleanup(self):
        print("[INFO] Cleaning up and exiting application.")
        self.system.cleanup()
        self.root.destroy()


class TextRedirector(object):
    def __init__(self, widget, tag):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.insert("end", str)
        self.widget.see("end")

    def flush(self):
        pass
