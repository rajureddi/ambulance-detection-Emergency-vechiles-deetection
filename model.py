import cv2
import threading
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/rajub/Downloads/ambulance_model.pt")

# Global controls
cap = None
running = False

# ---------------- IMAGE DETECTION ---------------- #

def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    frame = cv2.imread(file_path)

    results = model(frame, conf=0.3)
    annotated = results[0].plot()

    show_frame(annotated)


# ---------------- VIDEO DETECTION ---------------- #

def upload_video():
    global cap, running

    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if not file_path:
        return

    stop_video()

    cap = cv2.VideoCapture(file_path)
    running = True

    thread = threading.Thread(target=process_video)
    thread.start()


def process_video():
    global running

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)
        annotated = results[0].plot()

        show_frame(annotated)

    stop_video()


def stop_video():
    global running, cap
    running = False
    if cap:
        cap.release()


# ---------------- FRAME DISPLAY ---------------- #

def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame)
    img = img.resize((800, 600))

    imgtk = ImageTk.PhotoImage(image=img)

    panel.imgtk = imgtk
    panel.config(image=imgtk)


# ---------------- GUI ---------------- #

root = Tk()
root.title("YOLO Detection System")
root.geometry("1000x700")
root.configure(bg="black")

panel = Label(root)
panel.pack(pady=20)

btn_frame = Frame(root, bg="black")
btn_frame.pack()

Button(
    btn_frame,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 14),
    bg="violet",
    fg="white",
    width=15
).grid(row=0, column=0, padx=10)

Button(
    btn_frame,
    text="Upload Video",
    command=upload_video,
    font=("Arial", 14),
    bg="violet",
    fg="white",
    width=15
).grid(row=0, column=1, padx=10)

Button(
    btn_frame,
    text="Stop Video",
    command=stop_video,
    font=("Arial", 14),
    bg="red",
    fg="white",
    width=15
).grid(row=0, column=2, padx=10)

root.mainloop()