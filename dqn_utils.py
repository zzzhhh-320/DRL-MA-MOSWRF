import random
import math
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =================================================================================
# DQN 核心组件
# =================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    """一个固定大小的循环缓冲区，用于存储DQN的经验元组。"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个转换。"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """从内存中随机采样一个批次的转换。"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DuelingDQNNet(nn.Module):
    """Dueling DQN 的神经网络模型。"""
    def __init__(self, n_observations, n_actions):
        super(DuelingDQNNet, self).__init__()
        
        # 共享的特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # 状态价值流 (State Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 输出一个标量 V(s)
        )

        # 动作优势流 (Action Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions) # 输出每个动作的优势 A(s, a)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        
        # 分别计算 V(s) 和 A(s, a)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
       
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    """封装了DQN网络和学习逻辑的代理。"""
    def __init__(self, n_observations, n_actions, hyparams):
        self.n_actions = n_actions
        self.hyparams = hyparams
        self.steps_done = 0

        self.policy_net = DuelingDQNNet(n_observations, n_actions).to(device)
        self.target_net = DuelingDQNNet(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=hyparams['LR'], amsgrad=True)
        self.memory = ReplayBuffer(hyparams['MEMORY_SIZE'])

    def select_action(self, state):
        
        sample = random.random()
        eps_threshold = self.hyparams['EPS_END'] + (self.hyparams['EPS_START'] - self.hyparams['EPS_END']) * \
            math.exp(-1. * self.steps_done / self.hyparams['EPS_DECAY'])
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def select_greedy_action(self, state):
        
        with torch.no_grad():
            
            return self.policy_net(state).max(1)[1].view(1, 1)

    def learn(self):
        """从经验回放池中采样，训练DQN网络。"""
        if len(self.memory) < self.hyparams['BATCH_SIZE']:
            return

        transitions = self.memory.sample(self.hyparams['BATCH_SIZE'])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.hyparams['BATCH_SIZE'], device=device)
        # 只有在存在非最终状态时才计算next_state_values
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.cat(non_final_next_states_list)
            with torch.no_grad():
                # --- Double DQN 核心改造 ---
                # 1. 使用 policy_net (策略网络) 为下一个状态选择最佳动作
                best_next_actions = self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
                # 2. 使用 target_net (目标网络) 评估这些被选定动作的Q值
                # 这样就解耦了动作选择和价值评估，避免了Q值的过高估计
                next_q_values = self.target_net(non_final_next_states).gather(1, best_next_actions)
                next_state_values[non_final_mask] = next_q_values.squeeze()
        
        expected_state_action_values = (next_state_values * self.hyparams['GAMMA']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """软更新目标网络。"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.hyparams['TAU'] + target_net_state_dict[key] * (1 - self.hyparams['TAU'])
        self.target_net.load_state_dict(target_net_state_dict)



def encode_routes_to_permutation(routes, num_tasks):
 
    final_permutation = []
    for r in sorted(routes.keys()):
        final_permutation.extend(routes[r])
    
    # Sanity check to ensure all tasks are present
    if len(final_permutation) != num_tasks:
        # This case can happen if REGA fails to re-insert tasks.
        # A simple fix is to append missing tasks.
        all_tasks = set(range(1, num_tasks + 1))
        missing = list(all_tasks - set(final_permutation))
        final_permutation.extend(missing)

    return np.array(final_permutation) - 1 # Return 0-indexed
