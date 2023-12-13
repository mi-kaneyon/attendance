import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import threading
import torchvision.models as models

# CUDAが利用可能かどうかを確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet50 モデルのインスタンスを作成
model = models.resnet50()
# 全結合層の出力サイズを変更
model.fc = nn.Linear(model.fc.in_features, 2)
# モデルをデバイスに移動
model = model.to(device)
# 訓練済みの重みを読み込み
model.load_state_dict(torch.load('script/model_best.pth', map_location=device))
# モデルを評価モードに設定
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

# Check if the camera opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize a condition variable to synchronize the threads
cv = threading.Condition()

# Initialize a variable to store the current frame
current_frame = None

def capture_frames():
    global cap, current_frame  # Add this line to treat 'cap' and 'current_frame' as global variables
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera. Retrying...")
            cap.release()
            cv2.destroyAllWindows()
            cap = cv2.VideoCapture(0)  # Try to reinitialize the camera
            continue

        with cv:
            current_frame = frame
            cv.notify()

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

def perform_inference():
    global current_frame  # Add this line to treat 'current_frame' as a global variable
    while True:
        with cv:
            cv.wait()
            frame = current_frame

        try:
            input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_batch)

            # Softmax関数を使用して確率を計算
            probabilities = F.softmax(output, dim=1)
            # 最も確率の高いクラスとその確率を取得
            top_p, top_class = probabilities.topk(1, dim=1)
            label = "OK" if top_class.item() == 1 else "NG"
            probability_formatted = f"{top_p.item():.2f}"

            # Display the frame with inference results
            cv2.putText(frame, f"{label} - {probability_formatted}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Inference Frame', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Failed to perform inference: {e}")

inference_thread = threading.Thread(target=perform_inference)
inference_thread.start()

try:
    capture_thread.join()
    inference_thread.join()
finally:
    cap.release()
    cv2.destroyAllWindows()
