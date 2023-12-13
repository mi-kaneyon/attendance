import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import queue
import threading
import torchvision.models as models

# CUDA check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_terminated = False
# ResNet50 model instance
model = models.resnet50()
# layer output change
model.fc = nn.Linear(model.fc.in_features, 2)
# model move to device
model = model.to(device)
# trained model loading
model.load_state_dict(torch.load('script/model_best.pth', map_location=device))
# model inference mode
model.eval()

# Preprocessing Steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the camera
cap = cv2.VideoCapture(0)
# setting of frame ratio
cap.set(cv2.CAP_PROP_FPS, 30)

# resolution setting
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MJPG format by camera setting 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# Check if the camera opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize two queues to store frames and inference results
frame_queue = queue.Queue(maxsize=10)  # for raw frames from the camera
result_queue = queue.Queue(maxsize=10)  # for frames with inference results

# defenition
def capture_frames():
    global cap
    global is_terminated
    while not is_terminated:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera. Retrying...")
            cap.release()
            cv2.destroyAllWindows()
            cap = cv2.VideoCapture(0)  # Try to reinitialize the camera
            continue

        if not frame_queue.full():
            # Only put the new frame if the queue is not full
            frame_queue.put(frame)
        else:
            # Skip the frame if the queue is full
            continue

        if is_terminated:
            break

capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

def perform_inference():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_batch)

                # Softmax
                probabilities = F.softmax(output, dim=1)
                # best class and percentage
                top_p, top_class = probabilities.topk(1, dim=1)
                label = "OK" if top_class.item() == 1 else "NG"
                probability_formatted = f"{top_p.item():.2f}"

                # Store the result in the result queue
                result_queue.put((frame, f"{label} - {probability_formatted}"))
            except Exception as e:
                print(f"Failed to perform inference: {e}")

        if is_terminated:
            break

inference_thread = threading.Thread(target=perform_inference, daemon=True)
inference_thread.start()

try:
    while True:
        # Display the frame with inference results
        if not result_queue.empty():
            processed_frame, text = result_queue.get()
            cv2.putText(processed_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Inference Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_terminated = True
            break

finally:
    while capture_thread.is_alive() or inference_thread.is_alive():
        # Wait for both threads to finish
        pass

    cap.release()
    cv2.destroyAllWindows()

