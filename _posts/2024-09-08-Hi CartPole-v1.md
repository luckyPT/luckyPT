---
date: 2024-09-08 16:50:48
layout: post
title: Hi CartPole-v1
description: Hi CartPole-v1
image: /post_images/rl/CartPole-v1.png
optimized_image: /post_images/rl/CartPole-v1.png
category: 机器学习
tags:
  - 机器学习
  - 强化学习
  - Q-learning
  - CartPole-v1
author: 沙中世界
---
## Hello Q-learning

python gym的基础知识以及CartPole-v1的环境、状态等信息说明，查阅官网介绍即可；

base规则策略，通常只能坚持30步左右，杆就会失去平衡，达到fail状态；

基于q-learning的参数更新，训练100步以下 基本没什么效果，但是训练到150 ~ 200步左右的时候，大多数时候可以稳定在100步以上；

```
...
epoch =  97 g_reward =  50.0
epoch =  98 g_reward =  59.0
epoch =  99 g_reward =  100.0
epoch =  100 g_reward =  28.0
epoch =  101 g_reward =  19.0
epoch =  102 g_reward =  57.0
epoch =  103 g_reward =  23.0
epoch =  104 g_reward =  17.0
epoch =  105 g_reward =  62.0
...
...
epoch =  182 g_reward =  100.0
epoch =  183 g_reward =  100.0
epoch =  184 g_reward =  100.0
epoch =  185 g_reward =  100.0
epoch =  186 g_reward =  100.0
epoch =  187 g_reward =  100.0
epoch =  188 g_reward =  30.0
epoch =  189 g_reward =  100.0
epoch =  190 g_reward =  100.0
epoch =  191 g_reward =  100.0
epoch =  192 g_reward =  100.0
epoch =  193 g_reward =  100.0
```
原理介绍：
Q-Learning的核心Q-table参数值以及如何计算得到这些参数值；<br>
Q-table是一个二维数组，一个维度表示状态，另一个维度表示动作，值表示在某种状态下，采取某种动作，所得到的回报；<br>
参数计算有多种方案，下面展示了其中一种，有两个核心点：<br>
① 通过随机选择action来探索一定状态下最合适的action；<br>
② 从回报函数的设计，后向的state-action会间接影响前向状态及action，也就是假定后向选择逻辑一定的情况下，当前action的选择如果正确，得到的回报会比较大，如果当前action选择错误，得到的回报会比较小；<br>
  如何实现这种效果呢？是因为当前action执行之后，会返回一个state，这个state对应一个Q-value，即：回报值；实际上是取的这个state对应的所有action中Q value的最大值；<br>
  这个回报值越大，则当前的state-action对应的值增加的越大，


完整可执行代码如下：
```python
import gym
import time
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import time, math, random
from typing import Tuple

# 参考：https://github.com/RJBrooker/Q-learning-demo-Cartpole-V1/blob/master/cartpole.ipynb
env = gym.make('CartPole-v1', render_mode='human')
print(env.observation_space)
print(env.action_space)


# base 策略，只能支撑三四十步左右
def base_policy(obs):
    pos, velocity, angle, angle_velocity = obs
    return 1 if angle > 0 else 0


n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], math.radians(50)]


def discretizer(_, __, angle, pole_velocity) -> Tuple[int, ...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))


Q_table = np.zeros(n_bins + (env.action_space.n,))

# 训练100轮以下，看运气，运气差的时候 基本没什么效果
# 训练200轮以上，基本可以稳定支撑100步以上
def q_policy(state: tuple):
    """Choosing action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])


def new_Q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    """Temperal diffrence for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


train_count = 1000
e = 1
for epoch in range(train_count):
    g_reward = 0
    obs, _ = env.reset()
    current_state = discretizer(*obs)
    env.render()
    for _ in range(100):
        action = q_policy(current_state)
        if np.random.random() < exploration_rate(epoch):
            action = env.action_space.sample()  # explore
        obs, reward, terminated, truncated, info = env.step(action)
        new_state = discretizer(*obs)
        # Update Q-Table
        lr = learning_rate(epoch)
        learnt_value = new_Q_value(reward, new_state)
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value
        current_state = new_state

        g_reward += reward
        env.render()
        if terminated or truncated:
            print('epoch = ', epoch, 'g_reward = ', g_reward)
            break
        if g_reward == 100:
            print('epoch = ', epoch, 'g_reward = ', g_reward)
        # time.sleep(0.01)

env.close()

```



## Hello DQN
大约训练500轮次的时候，能够出现持续300步以上的模型<br>
偶尔可能会陷入局部最优，一直到达不了300步以上，需要多尝试几次

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=16, fc3_units=8):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.to(device)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    # Initialize the DQN agent
    def __init__(self, state_size, action_size, seed, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        self.memory = ReplayBuffer(action_size, buffer_size=int(1e5), batch_size=64, seed=seed)
        self.t_step = 0

    # Choose an action based on the current state
    def act(self, state, eps=0.):
        state_tensor = torch.from_numpy(state[0] if isinstance(state, tuple) else state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return action_values.argmax(dim=1).item()
        else:
            return np.random.randint(self.action_size)

    # Learn from batch of experiences
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


import gym
from collections import deque
import random

# Set up the environment
env = gym.make("CartPole-v1")

# Define training parameters
num_episodes = 2500
max_steps_per_episode = 500
epsilon_start = 1.0
epsilon_end = 0.2
epsilon_decay_rate = 0.99
gamma = 0.9
lr = 0.0005
buffer_size = 10000
buffer = deque(maxlen=buffer_size)
batch_size = 64
update_frequency = 3

# Initialize the DQNAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
new_agent = DQNAgent(input_dim, output_dim, seed=170715, lr=lr)

for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))

    # Run one episode
    for step in range(max_steps_per_episode):
        # Choose and perform an action
        action = new_agent.act(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)

        buffer.append((state[0] if isinstance(state, tuple) else state, action, reward, next_state, done))

        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            # Update the agent's knowledge
            new_agent.learn(batch, gamma)

        state = next_state

        # Check if the episode has ended
        if done:
            break

    if (episode + 1) % update_frequency == 0:
        print(f"Episode {episode + 1}: Finished training,", "step = ", step)
        if step >= 399:
            break

# Visualize the agent's performance
import time

env.close()
env = gym.make("CartPole-v1", render_mode='human')
state = env.reset()
done = False
print('start test... ...')
step_count = 0
while not done:
    env.render()
    action = new_agent.act(state, eps=0.)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    time.sleep(0.01)  # Add a delay to make the visualization easier to follow
    step_count += 1
print("step count = ", step_count)

env.close()

```

将参数更新为下面的参数，运气好的话，可以得到永久平衡的模型：
```python
# Define training parameters
lr = 0.001
buffer_size = 1000
buffer = deque(maxlen=buffer_size)
batch_size = 512
```

## PPO算法
很容易就能达到持久平衡
```Python
import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# Parameters
gamma = 0.99
seed = 1
log_interval = 10

env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('./exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience


def main():
    agent = PPO()
    for i_epoch in range(1000):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        target_steps = 1000
        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            state = next_state

            if done:
                if len(agent.buffer) >= agent.batch_size: agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
            else:
                if t > 0 and t % 100 == 0:
                    print(t)
        if t > target_steps:
            break
    env.close()
    # ===============TEST=============
    new_env = gym.make("CartPole-v1", render_mode='human')
    state = new_env.reset()
    done = False
    print('start test... ...')
    step_count = 0
    while not done:
        new_env.render()
        state = state[0] if isinstance(state, tuple) else state
        action, _ = agent.select_action(state)
        next_state, reward, done, _, _ = new_env.step(action)
        state = next_state
        time.sleep(0.01)  # Add a delay to make the visualization easier to follow
        step_count += 1
    print("step count = ", step_count)

    new_env.close()


if __name__ == '__main__':
    main()
    print("end")

```