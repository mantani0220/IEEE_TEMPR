import os
import copy
from   datetime import datetime
import numpy as np
from   tqdm import tqdm  # progress bar
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.tensorboard import SummaryWriter
import random

device = torch.device("cpu")

########################################################################
# Neural Networks (critic)
########################################################################
class Critic(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Critic, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 64
        self.state_space  = state_space
        self.input_space  = state_space + self.hidden_space*2
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.input_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        return x
    
    def sample_action(self, obs, h, c, epsilon):
        output = self.forward(obs, h, c)

        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return output.argmax().item()

###########################################################################
# Reply buffer
###########################################################################

class ReplayMemory(object):
    def __init__(self, state_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # buffers
        self.state      = np.zeros((max_size, state_dim))
        self.action_idx = np.zeros((max_size, 1), dtype=np.int32)
        self.next_state = np.zeros((max_size, state_dim))
        self.reward     = np.zeros((max_size, 1))
        self.not_done   = np.zeros((max_size, 1))

    def add(self, state, action_idx, next_state, reward, done):
        # buffering
        self.state[self.ptr]      = state
        self.action_idx[self.ptr] = action_idx
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        # move pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.LongTensor(self.action_idx[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
            )
    
    def __len__(self):
        return self.size

###############################################################################
# DQN agent
###############################################################################

class DQN:
    def __init__(self, state_dim, action_num, 
                 ## DQN options
                 CRITIC_LEARN_RATE   = 1e-3,
                 DISCOUNT            = 0.99, 
                 REPLAY_MEMORY_SIZE  = 5_000,
                 REPLAY_MEMORY_MIN   = 100,
                 MINIBATCH_SIZE      = 16, 
                 UPDATE_TARGET_EVERY = 5, 
                 EPSILON_INIT        = 1,
                 EPSILON_DECAY       = 0.998, 
                 EPSILON_MIN         = 0.01,
                 ): 

        # Critic
        self.critic              = Critic(state_dim, action_num)
        self.critic_target       = copy.deepcopy(self.critic)
        self.critic_optimizer    = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARN_RATE)
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY

        # Replay Memory
        self.replay_memory     = ReplayMemory(state_dim, REPLAY_MEMORY_SIZE)
        self.REPLAY_MEMORY_MIN = REPLAY_MEMORY_MIN
        self.MINIBATCH_SIZE    = MINIBATCH_SIZE
        
        # DQN Options
        self.actNum        = action_num
        self.DISCOUNT      = DISCOUNT
        self.epsilon       = EPSILON_INIT 
        self.EPSILON_INI   = EPSILON_INIT
        self.EPSILON_MIN   = EPSILON_MIN
        self.EPSILON_DECAY = EPSILON_DECAY
        
        # Initialization of variables
        self.target_update_counter = 0
                

    def get_qs(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  
        return self.critic.forward(state)
    
    
    def get_value(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  
        return self.critic.forward(state).max()
    
    ####################################################################
    # Training loop
    ####################################################################
    def train(self, env, 
              EPISODES      = 50, 
              LOG_DIR       = None,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 1,
              RESTART_EP    = None
              ):
        
        ######################################
        # Prepare log writer 
        ######################################
        if LOG_DIR:
            summary_dir    = LOG_DIR+'/'+datetime.now().strftime('%m%d_%H%M')
            summary_writer = SummaryWriter(log_dir=summary_dir)
        if LOG_DIR and SHOW_PROGRESS:
            print(f'Progress recorded: {summary_dir}')
            print(f'---> $tensorboard --logdir {summary_dir}')        
    
        ##########################################
        # Define iterator for training loop
        ##########################################
        start = 0 if RESTART_EP == None else RESTART_EP
        if SHOW_PROGRESS:    
            iterator = tqdm(range(start, EPISODES), ascii=True, unit='episodes')        
        else:
            iterator = range(start, EPISODES)
    
        if RESTART_EP:
            self.epsilon = max(self.EPSILON_MIN, 
                               self.EPSILON_INI*np.power(self.EPSILON_DECAY,RESTART_EP))
    
        #######################################################################
        # Main loop
        #######################################################################
        for episode in iterator:
            
            ##########################
            # Reset
            ##########################
            state  = env.reset() 
            is_done = False
            episode_reward = 0
            episode_q0     = self.get_value(state)
        
            #######################################################
            # Iterate until episode ends 
            #######################################################
            while not is_done:
                action = self.critic.sample_action(state, self.epsilon)
                
                # make a step
                next_state, reward, is_done = env.step(action)
                episode_reward += reward
        
                # store experience and train Q network
                self.replay_memory.add(state, action, next_state, reward, is_done)
                if len(self.replay_memory) > self.REPLAY_MEMORY_MIN:
                    self.update_critic()
        
                # update current state
                state = next_state
    
            ################################################
            # Update target Q-function and decay epsilon            
            ################################################
            self.target_update_counter += 1
            if self.target_update_counter > self.UPDATE_TARGET_EVERY:
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.target_update_counter = 0
    
            if self.epsilon > self.EPSILON_MIN:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon  = max( self.EPSILON_MIN, self.epsilon)        
        
            ###################################################
            # Log
            ###################################################
            if LOG_DIR: 
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     episode_q0,     episode)
                summary_writer.flush()
    
                if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                    
                    ckpt_path = summary_dir + f'/agent-{episode}'
                    torch.save({'critic-weights': self.critic.state_dict(),
                                'target-weights': self.critic_target.state_dict(),
                                }, 
                               ckpt_path)
   
    def update_critic(self):

        ###############################################
        # Calculate DQN loss
        ###############################################
        # Sample minibatch
        state, action_idx, next_state, reward, not_done = self.replay_memory.sample(self.MINIBATCH_SIZE)
        
        # Calculate target
        with torch.no_grad():
            next_value = self.critic_target.forward(next_state)
            targetQ    = reward + not_done * self.DISCOUNT * next_value

        # DQN Loss (lossD)
        currentQ = self.critic(state).gather(1, action_idx)        
        loss     = F.mse_loss( currentQ, targetQ )        

        #####################################
        # Update neural network weights 
        #####################################
        loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if not self.UNCERTAIN_PARAM==None:
            self.param_optimizer.step()
            self.param_optimizer.zero_grad()

        return # end: train_step


    def load_weights(self, ckpt_dir, ckpt_idx=None):

        if not os.path.isdir(ckpt_dir):         
            raise FileNotFoundError("Directory '{}' does not exist.".format(ckpt_dir))

        if not ckpt_idx or ckpt_idx == 'latest': 
            check_points = [item for item in os.listdir(ckpt_dir) if 'agent' in item]
            check_nums   = np.array([int(file_name.split('-')[1]) for file_name in check_points])
            latest_ckpt  = f'/agent-{check_nums.max()}'  
            ckpt_path    = ckpt_dir + latest_ckpt
        else:
            ckpt_path = ckpt_dir + f'/agent-{ckpt_idx}'
            if not os.path.isfile(ckpt_path):   
                raise FileNotFoundError("Check point 'agent-{}' does not exist.".format(ckpt_idx))

        checkpoint = torch.load(ckpt_path)
        self.critic.load_state_dict(checkpoint['critic-weights'])
        self.critic_target.load_state_dict(checkpoint['target-weights'])        
        #self.replay_memory = checkpoint['replay_memory']
        
        print(f'Agent loaded weights stored in {ckpt_path}')
        
        return ckpt_path    