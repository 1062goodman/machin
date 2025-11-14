import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn



transform = transforms.Compose([ #이미지 데이터는 [높이, 너비, 채널] 파이토치는 [채널, 높이, 너비]
    transforms.ToTensor(), # 이미지 데이터를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,)) # 데이터를 -1 ~ 1 범위로 정규화
])

#하이퍼 파라미터
batch =64; l_rate = 0.01; epoch=50

#train 데이터셋
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
data_iter= iter(train_loader)
#test 데이터셋
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)



# 가중치1 집합. 데이터의 크기가 28 * 28이니까 입력도 784. 128개의 은닉층
random_velues= ((torch.rand(784, 128))*2 -1) * 0.08
weight_1 = nn.Parameter(random_velues, requires_grad=True)
bias_1 = nn.Parameter(torch.zeros(128), requires_grad=True)

#가중치2 집합
random_velues= ((torch.rand(128,10))*2 -1) * 0.2
weight_2 = nn.Parameter(random_velues, requires_grad=True)
bias_2 = nn.Parameter(torch.zeros(10), requires_grad=True)

loss_curve = []
accur_curve = []

for x in range(epoch):
    a=0
    losssum=0
    for images, labels in train_loader:
        #images 평탄화. [batch][1][28][28]에서 [batch][1][784]
        images_flat= images.reshape(-1,784)
        

        #은닉층 계산
        z1=images_flat @ weight_1 + bias_1 #shpae [batch][128]
        a1 = torch.relu(z1)

        #출력층 계산
        osum=a1 @ weight_2 + bias_2 
        criterion=nn.CrossEntropyLoss()
        loss = criterion(osum,labels)
        losssum+= loss.item()
        loss.backward()
    

        with torch.no_grad():
            weight_1-= l_rate * weight_1.grad
            bias_1 -= l_rate * bias_1.grad
            weight_2-= l_rate * weight_2.grad
            bias_2 -= l_rate * bias_2.grad

        weight_1.grad.zero_()
        bias_1.grad.zero_()
        weight_2.grad.zero_()
        bias_2.grad.zero_()
        a +=1
        print(f"{a}번 학습 / {len(train_loader)} epoch: {x}", end="\r")
    print()
    print(x+1,"번째.\n테스트중... /loss: ",loss)
    loss_curve.append(losssum/len(train_loader))
        #트레인 집합 테스트
    with torch.no_grad():
        total_correct=0
        total_samples=0

        for images_test, labels_test in train_loader:
            images_test_flat= images_test.reshape(-1,784)

            z1_test=images_test_flat @ weight_1 + bias_1 #shpae [batch][128]
            a1_test = torch.relu(z1_test)

            osum_test=a1_test @ weight_2+ bias_2

            predicted_labels = torch.argmax(osum_test, dim=1) # 값들중 가장 큰값(예측값)의 인덱스
            total_correct += (predicted_labels == labels_test).sum().item()
            total_samples += labels_test.size(0)

    print(" 타겟 집합 정확도: ",total_correct," / ", total_samples,
    "\n", (total_correct/total_samples))
        
        
        #테스트 집합 테스트
    total_correct1=0
    total_samples1=0
    for images_test, labels_test in test_loader:
        images_test_flat= images_test.reshape(-1,784)

        z1_test=images_test_flat @ weight_1 + bias_1 #shpae [batch][128]
        a1_test = torch.relu(z1_test)

        osum_test=a1_test @ weight_2+ bias_2

        predicted_labels = torch.argmax(osum_test, dim=1) # 값들중 가장 큰값(예측값)의 인덱스
        total_correct1 += (predicted_labels == labels_test).sum().item()
        total_samples1 += labels_test.size(0)
    print(" 타겟 집합 정확도: ",total_correct1," / ", total_samples1,
    "\n", (total_correct1/total_samples1))
    
    accur_curve.append( [round(total_correct/total_samples,5),round(total_correct1/total_samples1,5)])
    print()


print(loss_curve)


    

import matplotlib.pyplot as plt

train_acc_history = [item[0] for item in accur_curve]
test_acc_history = [item[1] for item in accur_curve]


plt.figure(figsize=(12, 5)) #가로 12인치, 세로 5인치 크기


plt.subplot(1, 2, 1) # (1행 2열 중 1번째 그래프)
plt.plot(loss_curve, label='Training Loss') 
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend() 


plt.subplot(1, 2, 2) 
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.title('Training & Test Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend() 


plt.tight_layout() 
plt.show() 



