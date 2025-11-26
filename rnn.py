
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt



#하이퍼 파라미터
l_rate = 0.01; num_epoch=50


# 입력층 가중치. 입력은 1. 64개의 은닉층
random_velues= ((torch.rand(1, 64))*2 -1) * 0.1 #정규화
weight_in = nn.Parameter(random_velues, requires_grad=True)
b_in = nn.Parameter(torch.zeros(64), requires_grad=True)

#엣지 집합
random_velues= ((torch.rand(64,64))*2 -1) * 0.1
edge = nn.Parameter(random_velues, requires_grad=True)


# 출력층 가중치. 입력은 64. 1개의 출력과 바이어스
random_velues= ((torch.rand(64, 1))*2 -1) * 0.1 #정규화
weight_out = nn.Parameter(random_velues, requires_grad=True)
b_out = nn.Parameter(torch.zeros(1), requires_grad=True)


#데이터셋 뽑기

import FinanceDataReader as fdr

df = fdr.DataReader('005930', '2000')
x = list(df.Close)

df.to_csv('samsung_stock.csv') # csv
df_reset = df.reset_index()

min_val = min(x)
max_val = max(x)

x_normalized = [(d - min_val) / (max_val - min_val) for d in x] #정규화


#학습데이터 뽑기
import random

def random_batch(data, max=40):
   
    total_len = len(data)    
    start_idx = random.randint(0, total_len - 21) 
    space = total_len - start_idx
    
    limit = min(space-1, max)

    if limit <10:
        seq_len = space-1
    else:
        seq_len = random.randint(10, limit)
    
    
    
    x_seq = data[start_idx : start_idx + seq_len] # 입력값
    y_seq = data[start_idx + 1 : start_idx + seq_len + 1] #입력값+1 정답
    
    return x_seq, y_seq


optimizer = torch.optim.Adam([weight_in, b_in, edge, weight_out, b_out], lr=l_rate)
criterion = nn.MSELoss()

loss_history=[]



for epoch in range(num_epoch):
    input_data, label = random_batch(x_normalized)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
    target_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(1)

    seq_len = input_tensor.size(0)
    h = torch.zeros(1, 64)
    model_outputs = []

    for t in range(seq_len):
        
        x_t = input_tensor[t].unsqueeze(0) 
        
        h = torch.tanh( (x_t @ weight_in + b_in) + (h @ edge) )
    
        y_t = h @ weight_out + b_out
        
        # 예측값 저장
        model_outputs.append(y_t)
    
    final_outputs = torch.stack(model_outputs).squeeze(1)
    loss = criterion(final_outputs, target_tensor)
    loss_history.append(loss.item())


    optimizer.zero_grad() # 기울기 초기화
    loss.backward()       # 기울기 계산
    optimizer.step()      # 가중치 수정
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")




def analyze_edge(edge_param, title="RNN Edge Weights"):
    # 텐서에서 데이터 꺼내기 (Gradients는 backward 후에만 존재함)
    weights = edge_param.detach().numpy()
    
    print(f"[{title} 분석]")
    print(f"1. 가중치 평균 (Mean): {weights.mean():.6f}")
    print(f"2. 가중치 절댓값 평균 (Abs Mean): {abs(weights).mean():.6f}")
    print(f"3. 가중치 최댓값 (Max): {weights.max():.6f}")
    print(f"4. 가중치 최솟값 (Min): {weights.min():.6f}")
    
    if edge_param.grad is not None:
        grads = edge_param.grad.detach().numpy()
        print(f"5. 기울기(Gradient) 절댓값 평균: {abs(grads).mean():.6f}")
        print("   (이 값이 0에 가까우면 '기울기 소실', 너무 크면 '기울기 폭주'입니다)")
    else:
        print("5. 기울기 정보 없음 (backward() 실행 전임)")

    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    # cmap='coolwarm' : 파란색(-), 흰색(0), 빨간색(+)
    plt.imshow(weights, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f'{title} Heatmap (64x64)')
    plt.xlabel('From Hidden Node (h_t-1)')
    plt.ylabel('To Hidden Node (h_t)')
    plt.show()

# --- 실행 ---
# 사용 중인 edge 파라미터를 넣어주세요
analyze_edge(edge)

print("\n학습 완료! 마지막 배치 결과 확인")
prediction_price = final_outputs[-1].item() * (max_val - min_val) + min_val
real_price = target_tensor[-1].item() * (max_val - min_val) + min_val

print(f"모델 예측값: {prediction_price:.0f}원")
print(f"실제 정답값: {real_price:.0f}원")



plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='red')
plt.title('Training Process (Loss History)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.xlim(0, num_epoch)
plt.show()
    

pred_viz = final_outputs.detach().numpy()
real_viz = target_tensor.detach().numpy()
plt.figure(figsize=(10, 5))
plt.plot(real_viz, label='Real', marker='.', color='blue')
plt.plot(pred_viz, label='RNN Prediction', marker='.', color='orange', linestyle='--')
plt.title('Real vs RNN Prediction (Normalized)')
plt.legend()
plt.grid(True)
plt.show()






