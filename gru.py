import torch
import torch.nn as nn
import FinanceDataReader as fdr
import random
import matplotlib.pyplot as plt
import math

# --- 1. 하이퍼 파라미터 ---
l_rate = 0.01
num_epoch = 1000
hidden_size = 64

# --- 2. GRU 모델 파라미터 초기화 ---
# GRU는 게이트가 3개 필요합니다: Reset(r), Update(z), New(n)
# 따라서 가중치 크기는 Hidden Size * 3 입니다. (64 * 3 = 192)

# [입력 -> 3개의 게이트] (1 -> 192)
weight_in = nn.Parameter(((torch.rand(1, hidden_size * 3)) * 2 - 1) / math.sqrt(hidden_size), requires_grad=True)
b_in = nn.Parameter(torch.zeros(hidden_size * 3), requires_grad=True)

# [은닉 -> 3개의 게이트] (64 -> 192)
edge = nn.Parameter(((torch.rand(hidden_size, hidden_size * 3)) * 2 - 1) / math.sqrt(hidden_size), requires_grad=True)
b_hh = nn.Parameter(torch.zeros(hidden_size * 3), requires_grad=True)

# [출력층] (64 -> 1) (동일)
weight_out = nn.Parameter(((torch.rand(hidden_size, 1)) * 2 - 1) / math.sqrt(hidden_size), requires_grad=True)
b_out = nn.Parameter(torch.zeros(1), requires_grad=True)


# --- 3. 데이터 로드 및 전처리 (동일) ---
df = fdr.DataReader('005930', '2000')
x = list(df.Close)
min_val = min(x)
max_val = max(x)
x_normalized = [(d - min_val) / (max_val - min_val) for d in x] 

# --- 4. 배치 함수 (동일) ---
def random_batch(data, max_len=40):
    total_len = len(data)    
    start_idx = random.randint(0, total_len - 21) 
    space = total_len - start_idx
    limit = min(space-1, max_len)
    if limit < 10: seq_len = space-1
    else: seq_len = random.randint(10, limit)
    x_seq = data[start_idx : start_idx + seq_len] 
    y_seq = data[start_idx + 1 : start_idx + seq_len + 1] 
    return x_seq, y_seq

# --- 5. 학습 루프 ---
optimizer = torch.optim.Adam([weight_in, b_in, edge, b_hh, weight_out, b_out], lr=l_rate)
criterion = nn.MSELoss()
loss_history = []

print("GRU 학습 시작...")

for epoch in range(num_epoch):
    input_data, label_data = random_batch(x_normalized)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
    target_tensor = torch.tensor(label_data, dtype=torch.float32).unsqueeze(1)

    seq_len = input_tensor.size(0)

    # ★ GRU는 h(은닉 상태) 하나만 있으면 됩니다! (c 불필요)
    h = torch.zeros(1, hidden_size)
    
    model_outputs = []

    # ==========================================================
    # ★ GRU Forward Loop (수동 구현)
    # ==========================================================
    for t in range(seq_len):
        x_t = input_tensor[t].unsqueeze(0) # (1, 1)

        # 1. 선형 변환 (x와 h의 기여도 계산)
        # 나중에 Reset 게이트 연산을 위해 x_linear와 h_linear를 따로 계산해서 더하는 게 정석입니다.
        x_linear = x_t @ weight_in + b_in
        h_linear = h @ edge + b_hh
        
        # 2. 3등분 (Chunking) -> [Reset(r) | Update(z) | New(n)]
        r_x, z_x, n_x = x_linear.chunk(3, 1)
        r_h, z_h, n_h = h_linear.chunk(3, 1)
        
        # 3. 게이트 계산
        # Reset Gate: 과거의 기억을 얼마나 무시할지 결정
        r = torch.sigmoid(r_x + r_h)
        
        # Update Gate: 과거의 기억을 얼마나 유지할지 결정
        z = torch.sigmoid(z_x + z_h)
        
        # 4. 후보 은닉 상태 (Candidate Hidden State) - GRU의 핵심 수식
        # 여기서 Reset 게이트(r)가 과거의 기억(h)에 곱해집니다.
        n = torch.tanh(n_x + r * n_h)
        
        # 5. 최종 은닉 상태 업데이트
        # 공식: (1-z) * 새로운기억 + z * 옛날기억
        h = (1 - z) * n + z * h
        
        # 6. 출력
        y_t = h @ weight_out + b_out
        model_outputs.append(y_t)
    # ==========================================================
    
    final_outputs = torch.stack(model_outputs).squeeze(1)
    loss = criterion(final_outputs, target_tensor)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([weight_in, b_in, edge, b_hh, weight_out, b_out], 1.0)
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# --- 6. 결과 확인 ---
print("\n학습 완료! 마지막 배치 결과 확인")
prediction_price = final_outputs[-1].item() * (max_val - min_val) + min_val
real_price = target_tensor[-1].item() * (max_val - min_val) + min_val

print(f"GRU 예측값: {prediction_price:.0f}원")
print(f"실제 정답값: {real_price:.0f}원")

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='GRU Training Loss', color='purple')
plt.xlim(0, 300)
plt.title('GRU Training Process')
plt.legend()
plt.grid(True)
plt.show()

pred_viz = final_outputs.detach().numpy()
real_viz = target_tensor.detach().numpy()
plt.figure(figsize=(10, 5))
plt.plot(real_viz, label='Real', marker='.', color='blue')
plt.plot(pred_viz, label='GRU Prediction', marker='.', color='orange', linestyle='--')
plt.title('Real vs GRU Prediction (Normalized)')
plt.legend()
plt.grid(True)
plt.show()