import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image

# モデルの定義（ダミーのResNet18を使用）
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
model.fc = nn.Linear(512, 4)  # 出力クラス数を4に変更（保存されたモデルと一致させる）

# 保存されたモデルの状態をロード
model.load_state_dict(torch.load('script/save_model.pth'))
model.eval()  # 評価モード

# カメラの初期化
cap = cv2.VideoCapture(0)

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

while True:
    ret, frame = cap.read()
    
    # 画像をPyTorchテンソルに変換
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # 推論を実行
    with torch.no_grad():
        output = model(input_batch)
    
    # "OK" または "NG" の予測
    _, predicted = torch.max(output.data, 1)
    label = 'OK' if predicted.item() == 0 else 'NG'

    # ラベルを画像に表示
    color = (0, 255, 0) if label == 'OK' else (0, 0, 255)
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # 画像を表示
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
