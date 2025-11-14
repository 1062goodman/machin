import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import numpy as np 

# --- 1. 하이퍼파라미터 정의 ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001 
NUM_EPOCHS = 20 # (MNIST는 10 에포크면 충분히 98% 이상 나옵니다)

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용할 장치: {DEVICE}")

# --- 2. 데이터셋 및 데이터 로더 준비 (★수정 1★) ---
transform = transforms.Compose([
    transforms.ToTensor(), 
    # (★수정★) MNIST는 1채널이므로 (0.5,)를 사용합니다.
    transforms.Normalize((0.5,), (0.5,)) 
])

# 훈련 데이터셋 (MNIST)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 테스트 데이터셋 (MNIST)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. CNN 모델 아키텍처 정의 (★수정 2, 3★) ---
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # (★수정★) MNIST는 1채널이므로 in_channels=1
        # (28x28 입력 -> conv1(5x5) -> 24x24 -> pool(2x2) -> 12x12)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # (12x12 입력 -> conv2(5x5) -> 8x8 -> pool(2x2) -> 4x4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        
        # (★수정★) 최종 특징 맵 크기: 32채널 * 4x4 = 512
        self.fc1 = nn.Linear(32 * 4 * 4, 128) 
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10) # MNIST는 10개 클래스

    def forward(self, x):
        # x의 초기 shape: [batch, 1, 28, 28]
        
        # 합성곱 -> ReLU -> 풀링
        x = self.pool(F.relu(self.conv1(x))) # [batch, 1, 28, 28] -> [batch, 16, 12, 12]
        x = self.pool(F.relu(self.conv2(x))) # [batch, 16, 12, 12] -> [batch, 32, 4, 4]
        
        # (★수정★) Flatten (MLP에 넣기 위해 1차원 벡터로 변환)
        x = x.view(-1, 32 * 4 * 4) # [batch, 32, 4, 4] -> [batch, 512]
        
        # 완전 연결 층 (MLP) -> ReLU
        x = F.relu(self.fc1(x))    # [batch, 512] -> [batch, 128]
        x = F.relu(self.fc2(x))    # [batch, 128] -> [batch, 84]
        x = self.fc3(x)            # [batch, 84] -> [batch, 10] (최종 Logits)
        
        return x

# --- 4. 모델 인스턴스 생성, 손실 함수, 옵티마이저 정의 ---
model = MNIST_CNN().to(DEVICE) # 모델 이름을 MNIST_CNN으로 변경
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 

# --- 결과를 저장할 '빈 리스트' 생성 ---
train_loss_history = []
train_accuracy_history = []
test_accuracy_history = []

# --- 5. 훈련 루프 ---
print("\n--- 훈련 시작 ---")
for epoch in range(NUM_EPOCHS):
    model.train() 
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        images = images.to(DEVICE) 
        labels = labels.to(DEVICE) 
        
        optimizer.zero_grad() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    print(f"\nEpoch {epoch+1} 훈련 손실: {avg_train_loss:.4f}")

    # --- 6. 1 에포크마다 모델 평가 (훈련 셋 및 테스트 셋) ---
    model.eval() 
    
    # --- 훈련 셋 정확도 평가 (과대적합 확인용) ---
    train_correct = 0
    train_total = 0
    with torch.no_grad(): 
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    train_accuracy_history.append(train_accuracy)
    print(f"  훈련 셋 정확도: {train_accuracy:.2f}% ({train_correct}/{train_total})")

    # --- 테스트 셋 정확도 평가 ---
    test_correct = 0
    test_total = 0
    with torch.no_grad(): 
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    test_accuracy = 100 * test_correct / test_total
    test_accuracy_history.append(test_accuracy)
    print(f"  테스트 셋 정확도: {test_accuracy:.2f}% ({test_correct}/{test_total})")
    
print("\n--- 훈련 완료 ---")

# --- 7. Matplotlib를 사용해 로스 커브 그리기 ---
plt.figure(figsize=(12, 5))

# 그래프 1: 훈련 손실 (Loss Curve)
plt.subplot(1, 2, 1) 
plt.plot(train_loss_history, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 그래프 2: 훈련/테스트 정확도 (Accuracy Curve)
plt.subplot(1, 2, 2) 
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')
plt.title('Training & Test Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)') 
plt.legend()

plt.tight_layout() 
plt.show() 

# --- 8. 최종 이미지 예측 예시 (★수정 4, 5★) ---
print("\n--- 예측 예시 ---")
model.eval()
data_iter = iter(test_loader)
images, labels = next(data_iter)
images_show = images.to(DEVICE)
outputs = model(images_show)
_, predicted = torch.max(outputs.data, 1)

# (★수정★) MNIST용 시각화 함수
def imshow(img):
    img = img / 2 + 0.5     # 정규화 해제 (-1~1 -> 0~1)
    npimg = img.cpu().numpy() # GPU 텐서를 CPU numpy 배열로 변환
    # (★수정★) 흑백 이미지는 (H, W)로 표시, cmap='gray' 사용
    plt.imshow(npimg.squeeze(), cmap='gray') 
    plt.show()

# (★수정★) MNIST는 .classes 속성이 없으므로, 라벨 숫자(텐서)를 직접 사용
print('실제값: ', ' '.join(f'{labels[j].item():5d}' for j in range(4)))
print('예측값: ', ' '.join(f'{predicted.cpu()[j].item():5d}' for j in range(4))) 
imshow(torchvision.utils.make_grid(images.cpu()[:4]))