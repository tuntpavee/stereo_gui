import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# --- Variables ---
frame_width = 320
frame_height = 240
full_res_width = 1280
full_res_height = 720
recording = False
frame_count = 0
goal_frames = 0
save_path = ""

# --- GUI Setup ---
window = tk.Tk()
window.title("OpenCV GUI with Hello Kitty Background")
window.geometry("1000x600")

# Load and display background image
bg_image_path = "image.png"
bg_img = Image.open(bg_image_path)
bg_img = bg_img.resize((1000, 600))  # Resize as needed
bg_photo = ImageTk.PhotoImage(bg_img)

# Canvas with background image
canvas = tk.Canvas(window, width=1000, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Frames
left_frame = tk.Frame(window, bg='white')
right_frame = tk.Frame(window, bg='white')
canvas.create_window(20, 20, anchor="nw", window=left_frame)
canvas.create_window(700, 20, anchor="nw", window=right_frame)

# --- Left (Camera Views + Progress Bar) ---
progress_label = tk.Label(left_frame, text="0%", bg='white')
progress_label.pack()

progress = ttk.Progressbar(left_frame, length=frame_width * 2)
progress.pack(pady=(0, 10))

# Camera Feed Labels
cam0_label = tk.Label(left_frame)
cam0_label.pack(side="left", padx=5)

cam1_label = tk.Label(left_frame)
cam1_label.pack(side="right", padx=5)

# --- Right Frame: Inputs ---
label_style = {'fg': '#d63384', 'bg': 'white', 'font': ('Helvetica', 12, 'bold')}
entry_style = {'font': ('Helvetica', 12), 'width': 30}

tk.Label(right_frame, text="How many full-res frames:", **label_style).pack(anchor='w', padx=7, pady=(7, 0))
frame_count_entry = tk.Entry(right_frame, **entry_style)
frame_count_entry.pack(padx=5, pady=(0, 10))

tk.Label(right_frame, text="Folder name to create:", **label_style).pack(anchor='w', padx=7, pady=(7, 0))
subfolder_entry = tk.Entry(right_frame, **entry_style)
subfolder_entry.pack(padx=5, pady=(0, 10))

tk.Label(right_frame, text="Base path to save:", **label_style).pack(anchor='w', padx=7, pady=(7, 0))
folder_entry = tk.Entry(right_frame, **entry_style)
folder_entry.pack(padx=5, pady=(0, 10))

start_button = tk.Button(right_frame, text="Start Recording", command=lambda: start_recording(), bg="#d63384", fg="white")
start_button.pack(pady=10)

# --- Functions ---
def update_frame():
    global frame_count

    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if ret0:
        preview0 = cv2.resize(frame0, (frame_width, frame_height))
        frame_rgb0 = cv2.cvtColor(preview0, cv2.COLOR_BGR2RGB)
        img0 = Image.fromarray(frame_rgb0)
        imgtk0 = ImageTk.PhotoImage(image=img0)
        cam0_label.imgtk = imgtk0
        cam0_label.configure(image=imgtk0)

    if ret1:
        preview1 = cv2.resize(frame1, (frame_width, frame_height))
        frame_rgb1 = cv2.cvtColor(preview1, cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(frame_rgb1)
        imgtk1 = ImageTk.PhotoImage(image=img1)
        cam1_label.imgtk = imgtk1
        cam1_label.configure(image=imgtk1)

    if recording and ret0 and frame_count < goal_frames:
        full_res = cv2.resize(frame0, (full_res_width, full_res_height))
        filename = os.path.join(save_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, full_res)
        frame_count += 1
        progress['value'] = (frame_count / goal_frames) * 100
        progress_label.config(text=f"{int(progress['value'])}%")

    window.after(30, update_frame)

def start_recording():
    global recording, goal_frames, save_path, frame_count

    try:
        goal_frames = int(frame_count_entry.get())
        subfolder_name = subfolder_entry.get().strip()
        base_path = folder_entry.get().strip()

        if not subfolder_name or not base_path:
            raise ValueError("Both subfolder and base path must be provided.")

        save_path = os.path.join(base_path, subfolder_name)
        os.makedirs(save_path, exist_ok=True)

        frame_count = 0
        progress['value'] = 0
        progress_label.config(text="0%")
        recording = True
    except Exception as e:
        print(f"[Error] {e}")

# --- Open Cameras ---
cap0 = cv2.VideoCapture(0)  # First webcam
cap1 = cv2.VideoCapture(1)  # Second webcam
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, full_res_width)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, full_res_height)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, full_res_width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, full_res_height)

# --- Main Loop ---
update_frame()
window.mainloop()

# --- Cleanup ---
cap0.release()
cap1.release()
cv2.destroyAllWindows()
