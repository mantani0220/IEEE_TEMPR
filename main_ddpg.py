import numpy as np
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# Environment
from rl_env.energy_continue import ContinueEnergyEnv

###############################################################################
from agent.ddpg  import  DDPGagent
###############################################################################
if __name__ == "__main__":
    # Set seeds
    random.seed(1)
    np.random.seed(1)
    start_time = datetime.now()
    
    log_dir = 'logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    writer  = SummaryWriter(log_dir=log_dir)
    # Environment
    rl_env      = ContinueEnergyEnv()
    state_dim   = rl_env.observation_space
    act_dim     = rl_env.action_space
    max_act     = 1
    
    agent = DDPGagent(state_dim, act_dim, max_act)
    LOG_DIR = 'logs/test'
    agent.train(rl_env,
              EPISODES      = 1000,
              LOG_DIR       = None,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 1,
              RESTART_EP    = None )

