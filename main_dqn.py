import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from agent.dqn import DQN

# Environment
from rl_env.energy_lstm import LSTMEnergyEnv


##########################################################
# Main     
##########################################################
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    start_time = datetime.now()  
    log_dir = 'logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    writer  = SummaryWriter(log_dir=log_dir)

    # Environment
    env               = LSTMEnergyEnv()
    state_space       = env.observation_space
    action_space      = env.action_space

    #########################################
    # DQN agent
    #########################################
    agent = DQN(state_space, action_space,
                          ## Learning rate
                          CRITIC_LEARN_RATE   = 5e-3,                              
                          ## DQN options
                          DISCOUNT            = 1, 
                          REPLAY_MEMORY_SIZE  = int(1e4), 
                          REPLAY_MEMORY_MIN   = 1000,
                          MINIBATCH_SIZE      = 32,                              
                          UPDATE_TARGET_EVERY = 5,
                          EPSILON_INIT        = 1,
                          EPSILON_DECAY       = 0.998, 
                          EPSILON_MIN         = 0.01, 
                          )
    
    #########################################
    # training
    #########################################
    
    LOG_DIR = 'logs/test'

    agent.train(env, 
                EPISODES      = 3000, 
                SHOW_PROGRESS = True, 
                LOG_DIR       = LOG_DIR,
                SAVE_AGENTS   = True, 
                SAVE_FREQ     = 500,
                )

