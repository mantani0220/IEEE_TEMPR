import numpy as np
import random
import torch

# Environment
from rl_env.energy_continue import ContinueEnergyEnv

###############################################################################
# Main func with pytorch
from agent.drdpg import DRDPGagent

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    # log_dir = 'drdpg_logs/logs'+datetime.now().strftime('%m%d%H%M')
    # writer  = SummaryWriter(log_dir=log_dir)
    
    ###########################################    
    # Environment setting
    ##########################################
    env = ContinueEnergyEnv()
    state_space       = env.observation_space
    action_space      = env.action_space
    # rho_space       = 1
    max_action        = 1
    
    ###########################################
    # DRQN setting
    ###########################################
    buffer_len           = int(100000)
    batch_size           = 8
    Actor_learning_rate  = 1e-4
    Critic_learning_rate = 1e-4
    max_epi_num          = 100
    min_epi_num          = 20
    
    episodes             = 1000 
    lookup_step          = 24
    gamma                = 0.99
    tau                  = 5e-2
    
    #noise parameter
    initial_noise           = 0.2
    noise_decay             = 0.995
    noise_min               = 0.1
    
    # DRDPG param
    random_update = True # If you want to do random update instead of sequential update
    lookup_step = 24 * 1# If you want to do random update instead of sequential update
    max_epi_len = 700
    
    # Create Q functions and optimizer
    device = torch.device("cpu")
    
    agent = DRDPGagent(state_space, action_space, max_action, buffer_len, lookup_step, Actor_learning_rate, Critic_learning_rate,
                       gamma, batch_size,tau ,initial_noise,noise_decay,noise_min, random_update,min_epi_num)
    
    LOG_DIR = 'logs/'
    
    agent.train(env,
              EPISODES      = 1000,
              LOG_DIR       = LOG_DIR,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 1,
              RESTART_EP    = None )

    
    
    
