import tkinter as tk
from tkinter import filedialog
import cv2
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image, ImageTk
import numpy as np
import torch.nn as nn
import os
from datetime import datetime

# Create Result directory if it doesn't exist
if not os.path.exists('Result'):
    os.makedirs('Result')

# GUI setup
root = tk.Tk()
root.title("Attendance System")

# Create a ResNet model instance
model = models.resnet18(pretrained=False)

# Fine-tuning the fully connected layer (fc)
num_ftrs = model.fc.in_features
num_classes = 4  # 'ok' and 'ng'
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the trained model
model.load_state_dict(torch.load('script/save_model.pth'))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize video capture
cap = cv2.VideoCapture(0)

def update_image():
    global cap
    ret, frame = cap.read()
    if ret:
        # Convert to PIL Image and apply transformations
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_tensor = transform(image_pil)
        image_tensor = image_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)

        # Display result on GUI
        if preds.item() == 0:  # Assuming 0 is the class index for 'OK'
            label_result.config(text="OK", fg="green")
        else:
            label_result.config(text="NG", fg="red")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = f"Result/NG_{timestamp}.jpg"
            cv2.imwrite(save_path, frame)

        # Convert frame to ImageTk.PhotoImage and update label
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image=image)
        label_image.config(image=image)
        label_image.image = image

    root.after(50, update_image)

# Create and place label for displaying video feed
label_image = tk.Label(root)
label_image.pack()

# Create and place label for displaying inference result
label_result = tk.Label(root, text="", font=("Helvetica", 48))
label_result.pack()

# Start the image update function
update_image()

root.mainloop()
