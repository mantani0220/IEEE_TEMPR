import numpy as np
import copy
import collections
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Dict

device = torch.device("cpu")


class DDPGagent:
    def __init__(self, state_dim, action_dim, max_action, buffer_len, lookup_step,actor_learning_rate, critic_learning_rate): 
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = actor_learning_rate)

        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = critic_learning_rate)
        
        # Replay Memory
        self.memory = EpisodeBuffer()
        self.max_action  = max_action
        self.lookup_step = lookup_step
        self.noise_cnt = 0
        self.noise_cnt_min = 24*7*50
        
    def get_action(self, state, h, c, noise):
        action, new_h, new_c = self.actor(state, h, c)      
        action = action.cpu().data.numpy().flatten()
        self.policy_update_cnt += 1
        if self.policy_update_cnt < self.policy_update_cnt_min:
           action = self.max_action * (np.random.rand(self.actor.action_dim))
        else:
            noise_value = np.random.normal(0, noise, size=action.shape)
            action = (action + noise_value).clip(-self.max_action, self.max_action)
        return action, new_h, new_c
        
    def get_value(self, state, h, c):
        state  = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, h, c = self.actor(state, h, c)
        return self.critic(state, action).cpu().data.numpy().flatten(), h, c
    

# DDPG Agent class
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.hidden_space = 64
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.Linear1 = nn.Linear(self.state_dim, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first = True)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_dim)
        
    def forward(self, state, h, c):
        x = F.relu(self.Linear1(state))
        x, (new_h, new_c) = self.lstm(x, (h, c)) 
        x =self.Linear2(x)
        action = self.max_action * torch.tanh(x)
        return action, new_h, new_c
   
    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

class Critic(nn.Module):
    def __init__(self, state_dim = None, action_dim = None):
        super(Critic, self).__init__()
        self.hidden_space = 64
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.Linear1 = nn.Linear(self.state_dim + self.action_dim, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first = True)
        self.Linear2 = nn.Linear(self.hidden_space, 1)
        
    def forward(self, state, action, h, c):
        x = torch.cat([state, action],dim=2)
        x = F.relu(self.Linear1(x))
        # LSTM
        x, (new_h, new_c) = self.lstm(x, (h, c))
        q_value = self.Linear2(x)
        return q_value, new_h, new_c
    
    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

###########################################################################
# Reply buffer
###########################################################################

class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)
    
class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                 max_epi_num=100, max_epi_len=700,
                 batch_size=1,
                 lookup_step=None):
        
        self.random_update = random_update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)
        
    def sample(self,random_sample=False, batch_size=None):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update or random_sample:  # Random upodate       

            if batch_size:        
                sampled_episodes = random.sample(self.memory, batch_size)
            else:
                sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                # get minimum step from sampled episodes
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=True, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    # sample buffer with minstep size
                    idx = np.random.randint(0, len(episode)-min_step+1)
                    sample = episode.sample(random_update=True, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)
                    
        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(
                random_update=self.random_update))

        # buffers, sequence_length
        return sampled_buffer, len(sampled_buffer[0]['obs'])

                
    def __len__(self):
        return len(self.memory)

    

def train(actor=None, actor_target=None, critic=None, critic_target = None, episode_memory=None,
          device=None,
          actor_optimizer=None,
          critic_optimizer = None,
          batch_size=1,
          gamma=0.99):
    
    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    observations = [samples[i]["obs"] for i in range(batch_size)]
    actions = [samples[i]["acts"] for i in range(batch_size)]
    rewards = [samples[i]["rews"] for i in range(batch_size)]
    next_observations = [samples[i]["next_obs"] for i in range(batch_size)]
    dones = [samples[i]["done"] for i in range(batch_size)]
    # print(f'observations:{observations_1}')
    
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(
        observations.reshape(batch_size, seq_len, -1)).to(device)
    actions = torch.LongTensor(actions.reshape(
        batch_size, seq_len, -1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(
        batch_size, seq_len, -1)).to(device)
    next_observations = torch.FloatTensor(
        next_observations.reshape(batch_size, seq_len, -1)).to(device)
    dones = torch.FloatTensor(dones.reshape(
        batch_size, seq_len, -1)).to(device)
    
    # アクターのLSTMの初期状態を取得
    h_actor, c_actor = actor.init_hidden_state(batch_size=batch_size, training=True)
    h_actor_target, c_actor_target = actor_target.init_hidden_state(batch_size=batch_size, training=True)

    # クリティックのLSTMの初期状態を取得
    h_critic, c_critic = critic.init_hidden_state(batch_size=batch_size, training=True)
    h_critic_target, c_critic_target = critic_target.init_hidden_state(batch_size=batch_size, training=True)

    next_actions, h_critic_next, c_critic_next = actor(next_observations, h_critic, c_critic)
    
    # ターゲットクリティックネットワークでQ値を計算
    target_q_values, _, _ = critic_target(next_observations, next_actions, h_critic_target, c_critic_target)

    # ターゲット値を計算
    target_q_values = rewards + gamma * target_q_values

    # クリティックネットワークの計算
    current_q_values, _, _ = critic(observations, actions, h_critic_next, c_critic_next)
    
    # クリティックの損失
    critic_loss = F.mse_loss(current_q_values, target_q_values)

    # クリティックの更新
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # アクターの損失計算
    predicted_actions, h_actor_next, c_actor_next = actor(observations, h_actor, c_actor)
    actor_loss = -critic(observations, predicted_actions, h_critic, c_critic)[0].mean()

    # アクターの更新
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
