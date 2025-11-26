import torch
import torch.nn as nn
import FinanceDataReader as fdr
import random
import matplotlib.pyplot as plt
import math

# --- 1. 하이퍼 파라미터 (유지) ---
l_rate = 0.01
num_epoch = 1000
hidden_size = 64

# --- 2. LSTM 모델 파라미터 초기화 (핵심 변경!) ---
# LSTM은 4개의 게이트(forget, input, cell, output)를 위해 가중치가 4배 필요합니다.
# 따라서 64 * 4 = 256개의 열을 만듭니다.

# [입력 -> 4개의 게이트] (1 -> 256)
# Xavier 초기화 적용
weight_in = nn.Parameter(((torch.rand(1, hidden_size * 4)) * 2 - 1) / math.sqrt(hidden_size), requires_grad=True)
b_in = nn.Parameter(torch.zeros(hidden_size * 4), requires_grad=True)

# [은닉 -> 4개의 게이트] (64 -> 256)
# RNN의 edge와 같은 역할이지만 크기가 4배입니다.
edge = nn.Parameter(((torch.rand(hidden_size, hidden_size * 4)) * 2 - 1) / math.sqrt(hidden_size), requires_grad=True)
b_hh = nn.Parameter(torch.zeros(hidden_size * 4), requires_grad=True)

# [출력층] (64 -> 1) (RNN과 동일)
# 최종적으로 나가는 건 h 상태 하나이므로 이건 그대로입니다.
weight_out = nn.Parameter(((torch.rand(hidden_size, 1)) * 2 - 1) / math.sqrt(hidden_size), requires_grad=True)
b_out = nn.Parameter(torch.zeros(1), requires_grad=True)


# --- 3. 데이터 로드 및 전처리 (유지) ---
df = fdr.DataReader('005930', '2000')
x = list(df.Close)
min_val = min(x)
max_val = max(x)
x_normalized = [(d - min_val) / (max_val - min_val) for d in x] # 정규화

# --- 4. 배치 함수 (유지) ---
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
# 파라미터가 늘어났으니 optimizer에 다 등록해줍니다.
optimizer = torch.optim.Adam([weight_in, b_in, edge, b_hh, weight_out, b_out], lr=l_rate)
criterion = nn.MSELoss()

loss_history = []

print("LSTM 학습 시작...")

for epoch in range(num_epoch):
    input_data, label_data = random_batch(x_normalized)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
    target_tensor = torch.tensor(label_data, dtype=torch.float32).unsqueeze(1)

    seq_len = input_tensor.size(0)

    # ★ LSTM은 상태가 2개 필요합니다 (h: 은닉 상태, c: 셀 상태)
    h = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)
    
    model_outputs = []

    # ==========================================================
    # ★ LSTM Forward Loop (수동 구현)
    # ==========================================================
    for t in range(seq_len):
        x_t = input_tensor[t].unsqueeze(0) # (1, 1)

        # 1. 모든 게이트의 값을 한 번에 계산 (Linear 연산)
        # gates 크기: (1, 256) -> [i | f | g | o] 순서로 붙어있음
        gates = (x_t @ weight_in + b_in) + (h @ edge + b_hh)
        
        # 2. 4개의 덩어리로 쪼개기 (각각 64개씩)
        # chunk(4, dim=1)은 텐서를 4등분 해줍니다.
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        # 3. 활성화 함수 적용
        i = torch.sigmoid(i_gate) # 입력 게이트
        f = torch.sigmoid(f_gate) # 망각 게이트
        g = torch.tanh(g_gate)    # 데이터 후보 (cell candidate)
        o = torch.sigmoid(o_gate) # 출력 게이트

        # 4. 상태 업데이트 (LSTM의 핵심 공식)
        # C_t = f * C_t-1 + i * g
        c = f * c + i * g
        
        # h_t = o * tanh(C_t)
        h = o * torch.tanh(c)

        # 5. 최종 예측
        y_t = h @ weight_out + b_out
        model_outputs.append(y_t)
    # ==========================================================
    
    final_outputs = torch.stack(model_outputs).squeeze(1)
    loss = criterion(final_outputs, target_tensor)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    
    # LSTM은 기울기 폭주가 일어날 수 있으니 안전장치(Clipping) 추가 추천
    torch.nn.utils.clip_grad_norm_([weight_in, b_in, edge, b_hh, weight_out, b_out], 1.0)
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# --- 6. 결과 확인 및 시각화 ---
print("\n학습 완료! 마지막 배치 결과 확인")
prediction_price = final_outputs[-1].item() * (max_val - min_val) + min_val
real_price = target_tensor[-1].item() * (max_val - min_val) + min_val

print(f"LSTM 예측값: {prediction_price:.0f}원")
print(f"실제 정답값: {real_price:.0f}원")

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='LSTM Training Loss', color='green')
plt.xlim(0, 300) # 300까지만 확대해서 보기
plt.title('LSTM Training Process')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

"""
# 마지막 배치 예측 비교
pred_viz = final_outputs.detach().numpy()
real_viz = target_tensor.detach().numpy()
plt.figure(figsize=(10, 5))
plt.plot(real_viz, label='Real', marker='.', color='blue')
plt.plot(pred_viz, label='LSTM Prediction', marker='.', color='orange', linestyle='--')
plt.title('Real vs LSTM Prediction (Normalized)')
plt.legend()
plt.grid(True)
plt.show()
"""