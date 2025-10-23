import gymnasium
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer

# تنظیم دستگاه
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ساخت محیط
env = gymnasium.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ابرپارامترها
# هایپرپارامترهای پیشنهادی
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 1e-3
batch_size = 128
buffer_capacity = 10000
target_update_freq = 10
num_episodes = 2560

# ساخت شبکه‌ها
q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_capacity)
rewards_history = []

# حلقه آموزش
for episode in range(num_episodes):
    state = env.reset()
    if isinstance(state, tuple):  # برای gymnasium
        state = state[0]

    total_reward = 0

    for t in range(1, 500):  # حداکثر طول اپیزود
        # انتخاب عمل با ε-greedy
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = q_values.argmax().item()

        # اجرای عمل
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result

        if isinstance(next_state, tuple):
            next_state = next_state[0]

        # ذخیره‌سازی تجربه
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # یادگیری
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)

            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                targets = rewards + gamma * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # به‌روزرسانی دوره‌ای شبکه هدف
        if t % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            break

    # کاهش ε
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_history.append(total_reward)

    if (episode + 1) % 200 == 0:
        print(f"Episode {episode + 1}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")

plt.figure(figsize=(12, 5))

# پاداش هر اپیزود
plt.plot(range(len(rewards_history)), rewards_history, label='Episode Reward', color='skyblue', linewidth=1)

# رسم هر 25 اپیزود یک‌بار: هم پاداش هم میانگین
window = 150
episode_indices = []
raw_rewards = []
avg_rewards = []

for i in range(0, len(rewards_history) - window + 1, window):
    reward_block = rewards_history[i:i+window]
    episode_indices.append(i + window)                   # اپیزود انتهایی هر بلوک
    raw_rewards.append(reward_block[-1])                 # آخرین پاداش در بلوک
    avg_rewards.append(np.mean(reward_block))            # میانگین بلوک

# رسم نمودار
plt.figure(figsize=(12, 5))
plt.plot(episode_indices, raw_rewards, label='Episode Reward', color='skyblue', marker='o')
plt.plot(episode_indices, avg_rewards, label='Average Reward', color='orange', marker='x')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training on CartPole-v1")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_training_rewards_sampled.png")
plt.show()
