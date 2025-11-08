import customtkinter
from customtkinter import CTkImage, CTkLabel, CTkButton, CTkFrame, CTkFont, CTkToplevel, CTkTextbox
from PIL import Image, ImageTk
import cv2
import threading
import os
import time
import numpy as np 

# --- IMPORT OUR REAL MODEL AND REPORT FILES ---
import model 
import report

# --- App Configuration ---
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 750
IMAGE_DISPLAY_SIZE = 400

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

class DermOscanApp(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        self.title("DermOscan - AI Skin Detection")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # --- Set Window Icon ---
        try:
            self.iconbitmap("logo.ico")
        except Exception:
            pass

        self.current_image_path = None
        self.report_data = None
        self.camera_window = None
        self.cap = None
        self.last_camera_frame = None
        self.slideshow_images = []
        self.current_slide = 0

        # --- Load Assets ---
        self.load_slideshow_images()
        
        # --- Load Logo for Home Screen ---
        try:
            self.logo_image_ctk = CTkImage(Image.open("logo.png"), size=(300, 300))
        except FileNotFoundError:
            print("Warning: logo.png not found for home screen.")
            self.logo_image_ctk = None

        # --- Create Frames ---
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.home_frame = CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid(row=0, column=0, sticky="nsew")

        self.scan_frame = CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.scan_frame.grid(row=0, column=0, sticky="nsew")

        self.report_frame = CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.report_frame.grid(row=0, column=0, sticky="nsew")

        # --- Create Widgets ---
        self.create_home_widgets()
        self.create_scan_widgets()
        self.create_report_widgets()

        # --- Start App ---
        self.show_frame("home")

    def show_frame(self, frame_name):
        """Helper function to switch between frames."""
        if frame_name == "home":
            self.home_frame.tkraise()
            self.start_slideshow()
        else:
            self.stop_slideshow()
            if frame_name == "scan":
                self.scan_frame.tkraise()
            elif frame_name == "report":
                self.report_frame.tkraise()

    # --- Home Screen ---

    def load_slideshow_images(self):
        """Loads the 6 DOS images for the slideshow."""
        self.slideshow_images = []
        slide_image_names = [f"DOS{i}.jpg" for i in range(1, 7)] 
        
        for img_name in slide_image_names:
            try:
                if os.path.exists(img_name):
                    img = Image.open(img_name)
                    img_ctk = CTkImage(light_image=img, dark_image=img, size=(700, 350))
                    self.slideshow_images.append(img_ctk)
                else:
                     print(f"Warning: Slideshow image '{img_name}' not found.")
            except Exception as e:
                print(f"Error loading {img_name}: {e}")

        if not self.slideshow_images:
            img = Image.new('RGB', (700, 350), color="#2B2B2B")
            img_ctk = CTkImage(light_image=img, dark_image=img, size=(700, 350))
            self.slideshow_images.append(img_ctk)

    def create_home_widgets(self):
        self.home_frame.grid_rowconfigure(6, weight=1) 
        self.home_frame.grid_columnconfigure(0, weight=1)

        # 1. Logo (Row 0) - CENTERED
        if self.logo_image_ctk:
            logo_label = CTkLabel(self.home_frame, text="", image=self.logo_image_ctk)
            logo_label.grid(row=0, column=0, padx=20, pady=(40, 10))

        # 2. Title (Row 1)
        title_label = CTkLabel(self.home_frame, text="DermOscan", 
                               font=CTkFont(size=48, weight="bold"))
        title_label.grid(row=1, column=0, padx=20, pady=(5, 5), sticky="s")

        # 3. Subtitle with "Made by AYUSH" (Row 2)
        subtitle_label = CTkLabel(self.home_frame, 
                                  text="AI-Driven Skin Cancer Detection\nMade by AYUSH", 
                                  font=CTkFont(size=20))
        subtitle_label.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="n")

        # 4. Info Text (Row 3)
        info_label = CTkLabel(self.home_frame, 
                              text="Utilizing advanced computer vision for preliminary analysis of skin lesions.\n"
                                   "This is not a substitute for professional medical advice.",
                              font=CTkFont(size=14),
                              text_color="gray")
        info_label.grid(row=3, column=0, padx=20, pady=(0, 10))

        # 5. Slideshow (Row 4)
        self.slideshow_label = CTkLabel(self.home_frame, text="", 
                                        image=self.slideshow_images[0],
                                        fg_color="transparent")
        self.slideshow_label.grid(row=4, column=0, padx=20, pady=10)

        # 6. Start Button (Row 5)
        start_button = CTkButton(self.home_frame, text="Get Started", 
                                 font=CTkFont(size=18, weight="bold"),
                                 corner_radius=10,
                                 command=lambda: self.show_frame("scan"),
                                 height=50, width=200)
        start_button.grid(row=5, column=0, padx=20, pady=(20, 40), sticky="n")

    def start_slideshow(self):
        if len(self.slideshow_images) > 1:
            self.current_slide = (self.current_slide + 1) % len(self.slideshow_images)
            self.slideshow_label.configure(image=self.slideshow_images[self.current_slide])
            self.slideshow_job = self.after(3000, self.start_slideshow)

    def stop_slideshow(self):
        if hasattr(self, 'slideshow_job'):
            self.after_cancel(self.slideshow_job)

    # --- Scan Screen ---

    def create_scan_widgets(self):
        self.scan_frame.grid_columnconfigure(0, weight=1)
        self.scan_frame.grid_rowconfigure(0, weight=1)

        main_container = CTkFrame(self.scan_frame, fg_color="transparent")
        main_container.grid(row=0, column=0, pady=40, padx=60, sticky="nsew")
        
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        controls_frame = CTkFrame(main_container, fg_color="transparent")
        controls_frame.grid(row=0, column=0, padx=(0, 40), sticky="n")

        back_button = CTkButton(controls_frame, text="< Back to Home", 
                                command=self.go_home, fg_color="transparent", 
                                text_color=("gray20", "gray80"), hover=False)
        back_button.pack(anchor="w", pady=(0, 30))

        upload_button = CTkButton(controls_frame, text="Upload Photo", 
                                  command=self.upload_photo, height=40, 
                                  font=CTkFont(size=16))
        upload_button.pack(fill="x", pady=10)

        camera_button = CTkButton(controls_frame, text="Open Camera", 
                                  command=self.open_camera, height=40, 
                                  font=CTkFont(size=16))
        camera_button.pack(fill="x", pady=10)
        
        self.scan_button = CTkButton(controls_frame, text="Quick Scan", 
                                     command=self.start_scan_thread, height=50, 
                                     font=CTkFont(size=18, weight="bold"),
                                     state="disabled")
        self.scan_button.pack(fill="x", pady=(30, 10))

        image_frame = CTkFrame(main_container, fg_color=("gray90", "gray20"), 
                               width=IMAGE_DISPLAY_SIZE, height=IMAGE_DISPLAY_SIZE)
        image_frame.grid(row=0, column=1, sticky="nsew")
        image_frame.grid_propagate(False) 

        self.image_display_label = CTkLabel(image_frame, text="Upload or capture an image\nto begin analysis.", 
                                            font=CTkFont(size=18), text_color="gray")
        self.image_display_label.pack(expand=True, fill="both", padx=10, pady=10)

    def upload_photo(self):
        file_path = customtkinter.filedialog.askopenfilename(
            title="Select a skin lesion image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif")]
        )
        if file_path:
            self.load_image_to_display(file_path)
            self.current_image_path = file_path
            self.scan_button.configure(state="normal")

    def load_image_to_display(self, image_path):
        try:
            pil_image = Image.open(image_path)
            img_ctk = CTkImage(light_image=pil_image, dark_image=pil_image, 
                               size=(IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE))
            self.image_display_label.configure(image=img_ctk, text="")
        except Exception as e:
            print(f"Error loading image: {e}")
            self.image_display_label.configure(image=None, text="Error: Failed to load image.")

    def open_camera(self):
        if self.camera_window and self.camera_window.winfo_exists():
            self.camera_window.focus()
            return

        self.camera_window = CTkToplevel(self)
        self.camera_window.title("Camera")
        self.camera_window.geometry("660x580")
        self.camera_window.resizable(False, False)
        self.camera_window.transient(self)

        self.camera_label = CTkLabel(self.camera_window, text="")
        self.camera_label.pack(pady=10)

        capture_button = CTkButton(self.camera_window, text="Capture", 
                                   command=self.capture_image, height=40, 
                                   font=CTkFont(size=16))
        capture_button.pack(pady=10)
        
        self.camera_window.protocol("WM_DELETE_WINDOW", self.close_camera)

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("Cannot open webcam")
            self.update_camera_feed()
        except Exception as e:
            print(f"Camera Error: {e}")
            self.camera_label.configure(text=f"Error: Unable to open webcam.\n{e}")
            capture_button.configure(state="disabled")

    def update_camera_feed(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_camera_frame = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                img_ctk = CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
                self.camera_label.configure(image=img_ctk)
                self.camera_window.after(10, self.update_camera_feed)
            else:
                self.close_camera()
        
    def capture_image(self):
        if self.last_camera_frame is not None:
            self.current_image_path = "temp_capture.jpg"
            cv2.imwrite(self.current_image_path, self.last_camera_frame)
            self.load_image_to_display(self.current_image_path)
            self.scan_button.configure(state="normal")
            self.close_camera()
        else:
             print("Error: No frame to capture.")

    def close_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None

    def start_scan_thread(self):
        if not self.current_image_path:
            return

        self.popup = CTkToplevel(self)
        self.popup.title("Processing")
        self.popup.geometry("300x150")
        self.popup.transient(self)
        self.popup.grab_set()
        
        popup_label = CTkLabel(self.popup, text="Be patient, your report\nis getting ready...", 
                               font=CTkFont(size=16))
        popup_label.pack(expand=True, pady=20)
        
        self.scan_button.configure(state="disabled")
        
        self.scan_thread = threading.Thread(
            target=self.run_scan_processing,
            args=(self.current_image_path, self.popup),
            daemon=True
        )
        self.scan_thread.start()

    # --- Skin Detection Helper ---
    def is_skin_present(self, image_path):
        """Checks if the image contains a reasonable amount of skin-colored pixels."""
        try:
            img = cv2.imread(image_path)
            if img is None: return False

            # Convert to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define lower and upper bounds for typical skin tones
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # Create mask and count pixels
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_pixels = cv2.countNonZero(mask)
            total_pixels = img.shape[0] * img.shape[1]
            skin_percentage = (skin_pixels / total_pixels) * 100

            return skin_percentage > 15
        except Exception as e:
            print(f"Skin detection error: {e}")
            return True 

    def run_scan_processing(self, image_path, popup_window):
        # --- 1. PERFORM SKIN CHECK FIRST ---
        if not self.is_skin_present(image_path):
             self.report_data = {
                "status": "Error",
                "prediction": "No Skin Detected",
                "confidence": "N/A",
                "report_id": "N/A",
                "factors": {"Analysis": "The uploaded image does not appear to contain enough visible skin area for analysis."},
                "recommendation": "Please upload a clear, well-lit photo of a skin lesion.",
                "image_path": image_path
            }
             self.after(0, self.show_report, popup_window)
             return

        # --- 2. IF SKIN IS PRESENT, RUN MODEL ---
        try:
            self.report_data = model.get_real_prediction(image_path)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            self.report_data = {
                "status": "Error",
                "prediction": "Analysis Failed",
                "confidence": "N/A",
                "report_id": "N/A",
                "factors": {"Error": str(e)},
                "recommendation": "An error occurred. Please try again.",
                "image_path": image_path
            }
        self.after(0, self.show_report, popup_window)

    def show_report(self, popup_window):
        popup_window.destroy()
        self.populate_report_widgets()
        self.show_frame("report")

    # --- Report Screen ---

    def create_report_widgets(self):
        self.report_frame.grid_columnconfigure(0, weight=2)
        self.report_frame.grid_columnconfigure(1, weight=3)
        self.report_frame.grid_rowconfigure(1, weight=1)
        
        header_frame = CTkFrame(self.report_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, padx=40, pady=(40, 20), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)
        
        self.report_title_label = CTkLabel(header_frame, text="Report:", 
                                           font=CTkFont(size=28, weight="bold"))
        self.report_title_label.grid(row=0, column=0, sticky="w")
        
        self.report_confidence_label = CTkLabel(header_frame, text="Confidence: ", 
                                                font=CTkFont(size=18))
        self.report_confidence_label.grid(row=1, column=0, sticky="w")
        
        button_frame = CTkFrame(header_frame, fg_color="transparent")
        button_frame.grid(row=0, column=1, rowspan=2, sticky="e")
        
        self.download_button = CTkButton(button_frame, text="Download as PDF", 
                                         command=self.download_pdf, height=40)
        self.download_button.pack(side="left", padx=10)
        
        self.new_report_button = CTkButton(button_frame, text="New Report", 
                                           command=self.go_home, height=40, 
                                           fg_color="gray", hover_color="gray30")
        self.new_report_button.pack(side="left")

        self.report_image_frame = CTkFrame(self.report_frame, fg_color=("gray90", "gray20"))
        self.report_image_frame.grid(row=1, column=0, padx=(40, 10), pady=(0, 40), sticky="nsew")
        
        self.report_image_label = CTkLabel(self.report_image_frame, text="Image Preview", 
                                           font=CTkFont(size=14))
        self.report_image_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.report_details_textbox = CTkTextbox(self.report_frame, 
                                                 font=CTkFont(size=14), 
                                                 wrap="word",
                                                 state="disabled")
        self.report_details_textbox.grid(row=1, column=1, padx=(10, 40), pady=(0, 40), sticky="nsew")

    def populate_report_widgets(self):
        if not self.report_data:
            return

        self.report_title_label.configure(text=f"Prediction: {self.report_data['prediction']}")
        self.report_confidence_label.configure(
            text=f"Model Confidence: {self.report_data['confidence']}"
        )
        
        if self.report_data["status"] == "Warning":
            self.report_title_label.configure(text_color="#F96161")
        elif self.report_data["status"] == "Clear":
            self.report_title_label.configure(text_color="#5DF1A2")
        else:
            self.report_title_label.configure(text_color=("gray20", "gray80"))

        report_string = f"REPORT ID: {self.report_data['report_id']}\n"
        report_string += f"STATUS: {self.report_data['status']}\n\n"
        report_string += "--- KEY FACTORS & ANALYSIS ---\n"
        for key, value in self.report_data['factors'].items():
            report_string += f"  • {key}: {value}\n"
        report_string += "\n--- RECOMMENDATION ---\n"
        report_string += self.report_data['recommendation']
        report_string += "\n\n--- DISCLAIMER ---\n"
        report_string += ("This is an AI-generated report for informational purposes only. "
                          "It is not a substitute for a professional medical diagnosis. "
                          "Always consult a qualified dermatologist for any health concerns.")
        report_string += "\n\n[ Application Developed by AYUSH ]"

        self.report_details_textbox.configure(state="normal")
        self.report_details_textbox.delete("1.0", "end")
        self.report_details_textbox.insert("1.0", report_string)
        self.report_details_textbox.configure(state="disabled")

        try:
            pil_image = Image.open(self.report_data['image_path'])
            img_ctk = CTkImage(light_image=pil_image, dark_image=pil_image, 
                               size=(IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE))
            self.report_image_label.configure(image=img_ctk, text="")
        except Exception as e:
            print(f"Error showing report image: {e}")
            self.report_image_label.configure(image=None, text="Image Error")

    def download_pdf(self):
        if self.report_data:
            report.generate_report_pdf(self.report_data)

    def go_home(self):
        self.current_image_path = None
        self.report_data = None
        self.scan_button.configure(state="disabled")
        
        # --- ROBUST FIX FOR IMAGE CLEARING BUG ---
        try:
            self.image_display_label.configure(image=None, text="Upload or capture an image\nto begin analysis.")
        except Exception:
            # Gracefully ignore if CTk fails to clear the old image reference
            pass

        self.report_details_textbox.configure(state="normal")
        self.report_details_textbox.delete("1.0", "end")
        self.report_details_textbox.configure(state="disabled")
        self.report_title_label.configure(text="Report:")
        self.report_confidence_label.configure(text="Confidence: ")
        
        # --- ROBUST FIX FOR REPORT IMAGE TOO ---
        try:
            self.report_image_label.configure(image=None, text="Image Preview")
        except Exception:
            pass
            
        self.show_frame("home")


if __name__ == "__main__":
    if os.path.exists("temp_capture.jpg"):
        os.remove("temp_capture.jpg")
        
    app = DermOscanApp()
    app.mainloop()

    if os.path.exists("temp_capture.jpg"):
        os.remove("temp_capture.jpg")