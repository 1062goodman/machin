import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F # ReLU 같은 함수를 위해 필요
from tqdm import tqdm # 훈련 진행 상황을 아름다운 프로그레스 바로 표시
import matplotlib.pyplot as plt # (★ 1. 그래프(Plot) 라이브러리 임포트 ★)
import numpy as np # (이미지 시각화를 위해 임포트)

# --- 1. 하이퍼파라미터 정의 ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001 
NUM_EPOCHS = 20

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용할 장치: {DEVICE}")

# --- 2. 데이터셋 및 데이터 로더 준비 ---
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. CNN 모델 아키텍처 정의 (nn.Module 상속) ---
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 5 * 5, 128) 
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 32 * 5 * 5) 
        x = F.relu(self.fc1(x))    
        x = F.relu(self.fc2(x))    
        x = self.fc3(x)            
        return x

# --- 4. 모델 인스턴스 생성, 손실 함수, 옵티마이저 정의 ---
model = CIFAR10_CNN().to(DEVICE) 
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 

# --- (★ 2. 결과를 저장할 '빈 리스트' 생성 ★) ---
train_loss_history = []
train_accuracy_history = []
test_accuracy_history = []

# --- 5. 훈련 루프 ---
print("\n--- 훈련 시작 ---")
for epoch in range(NUM_EPOCHS):
    model.train() 
    running_loss = 0.0
    
    # 훈련 루프 (tqdm 사용)
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        images = images.to(DEVICE) 
        labels = labels.to(DEVICE) 
        
        optimizer.zero_grad() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # (★ 3. 에포크 종료 시, '훈련 손실'을 리스트에 추가 ★)
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
    # (★ 3. '훈련 정확도'를 리스트에 추가 ★)
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
    # (★ 3. '테스트 정확도'를 리스트에 추가 ★)
    test_accuracy_history.append(test_accuracy)
    print(f"  테스트 셋 정확도: {test_accuracy:.2f}% ({test_correct}/{test_total})")
    
print("\n--- 훈련 완료 ---")

# --- (★ 4. Matplotlib를 사용해 로스 커브 그리기 ★) ---
# 2개의 그래프를 나란히 그리기
plt.figure(figsize=(12, 5))

# 그래프 1: 훈련 손실 (Loss Curve)
plt.subplot(1, 2, 1) # (1행 2열 중 1번째 그래프)
plt.plot(train_loss_history, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 그래프 2: 훈련/테스트 정확도 (Accuracy Curve)
plt.subplot(1, 2, 2) # (1행 2열 중 2번째 그래프)
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')
plt.title('Training & Test Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)') # 단위가 %임
plt.legend()

plt.tight_layout() # 그래프가 겹치지 않게 함
plt.show() # 그래프 창을 화면에 띄움

# --- 7. 최종 이미지 예측 예시 (선택 사항) ---
print("\n--- 예측 예시 ---")
model.eval()
data_iter = iter(test_loader)
images, labels = next(data_iter)
images_show = images.to(DEVICE) # 시각화용 이미지
outputs = model(images_show)
_, predicted = torch.max(outputs.data, 1)

def imshow(img):
    img = img / 2 + 0.5     # 정규화 해제 (-1~1 -> 0~1)
    npimg = img.cpu().numpy() # GPU 텐서를 CPU numpy 배열로 변환
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # (C, H, W) -> (H, W, C)
    plt.show()

print('실제값: ', ' '.join(f'{test_dataset.classes[labels[j]]:5s}' for j in range(4)))
print('예측값: ', ' '.join(f'{test_dataset.classes[predicted.cpu()[j]]:5s}' for j in range(4))) # .cpu() 추가
imshow(torchvision.utils.make_grid(images.cpu()[:4])) # .cpu() 추가