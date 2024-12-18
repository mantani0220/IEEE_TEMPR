import numpy as np
import random
import copy
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime
# Environment
from rl_env.energy_continue import ContinueEnergyEnv

###############################################################################
# Main func with pytorch
from agent.ddpg import Actor, Critic, EpisodeMemory, EpisodeBuffer, DDPGagent,train

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    log_dir = 'logs/logs'+datetime.now().strftime('%m%d%H%M')
    writer  = SummaryWriter(log_dir=log_dir)
    
    ###########################################    
    # Environment setting
    ##########################################
    env = ContinueEnergyEnv()
    state_space       = env.observation_space
    action_space      = env.action_space
    # rho_space       = 1
    max_action        = 1
    
    ############################################
    # DDPG setting parameters
    ############################################
    batch_size              = 8
    actor_learning_rate     = 5e-4
    critic_learning_rate    = 5e-4
    buffer_len              = int(100000)
    max_epi_num             = 100
    min_epi_num             = 30
    
    episodes                = 5000
    tau                     = 1e-2
    gamma                   = 0.9
    max_step                = 1000
    
    policy_update_cnt       = 0
    target_update_period    = 4
    policy_noise            = 0.2
    noise_clip              = 0.5
    policy_freq             = 2
    policy_update_cnt       = 0
    
    # noise parameter
    initial_noise           = 0.2
    noise_decay             = 1
    noise_min               = 0.001
    
    # DRDPG param
    random_update = True # If you want to do random update instead of sequential update
    lookup_step = 24 * 1# If you want to do random update instead of sequential update
    max_epi_len = 700
    max_epi_step = max_step
    ##################################################
    
    # Create Q functions and optimizer
    device = torch.device("cpu")
    
    agent = DDPGagent(state_space, action_space, max_action, buffer_len, lookup_step, actor_learning_rate, critic_learning_rate)
    
    #Actor
    actor = Actor(state_space, action_space, max_action).to(device)
    actor_target = copy.deepcopy(actor)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer    = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
    
    # Critic
    critic = Critic(state_space, action_space).to(device)
    critic_target = copy.deepcopy(critic)
    critic_target.load_state_dict(critic.state_dict())
    
    critic_optimizer    = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)
    
    episode_memory = EpisodeMemory(random_update, max_epi_num, max_epi_len, batch_size, lookup_step)
    noise = initial_noise
    reward_list = []
    q_value_list = []
    solar_generation_data = []
    bidding_history = []  # Initialize bidding_history
    scaling_history = []  # Initialize scaling_history
    soc_history = []
    step_rewards_list = [] 
    Pc_t_history = []
    Pd_t_history = []
    penalty_history = []
    battery_penalty_history = []
    xD_t_history = []
    
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Train
    for i in range(episodes):
        obs = env.reset()
        
        env.scaling = 1.0
        env.bidding = 0.0
        done = False
        
        episode_reward = 0
        episode_reward_discount = 0
        
        episode_bidding = []  # List to collect bidding data
        episode_scaling = []  # List to collect scaling data
        episode_step_rewards = [] 
        episode_soc_history = [] 
        episode_q_values = [] 
        episode_Pc_t_history = []
        episode_Pd_t_history = []
        episode_penalty_history = []
        episode_battery_penalty_history = []
        episode_xD_t_history = []
        
        episode_record = EpisodeBuffer()
        h, c = critic.init_hidden_state(batch_size=batch_size, training=False)
        
        for t in range(max_step):

            # Get action
            a, h, c = agent.get_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                              h.to(device), c.to(device),
                                              noise)

            # Do action
            s_prime, r, done = env.step(a)
            obs_prime = s_prime
            episode_bidding.append(env.bidding)  # Collect bidding data
            episode_scaling.append(env.scaling)  # Collect scaling data
            episode_soc_history.append(env.soc)
            episode_step_rewards.append(r)
            episode_Pc_t_history.append(env.Pc_t)
            episode_Pd_t_history.append(env.Pd_t)
            # episode_penalty_history.append(env.penalty)
            # episode_battery_penalty_history.append(env.battery_penalty)
            # episode_xD_t_history.append(env.xD_t)
            
            with torch.no_grad():
                q_values, _, _ = critic.forward(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), torch.from_numpy(a).float().to(device).unsqueeze(0).unsqueeze(0), h, c)
                max_q_value = q_values.max().item()
                episode_q_values.append(max_q_value)
                
                # if t == 0:
                #    initial_q_value = max_q_value
            # make data
            done_mask = 0.0 if done else 1.0

            episode_record.put([obs, a, r, obs_prime, done_mask])

            obs = obs_prime
            
            episode_reward += r
            
            episode_reward_discount = r + gamma*episode_reward_discount
            
            if len(episode_memory) >= min_epi_num:
                train(actor, actor_target, critic, critic_target , episode_memory,
                          device,
                          actor_optimizer,
                          critic_optimizer,
                          batch_size,
                          gamma)

                if (t+1) % target_update_period == 0:
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            if done:
                break
        
        episode_memory.put(episode_record)
        
        # noise = max(noise, noise_min)  # ノイズ下限値を設定
    
        bidding_history.append(episode_bidding)  # Save episode bidding data
        scaling_history.append(episode_scaling)  # Save episode scaling data
        step_rewards_list.append(episode_step_rewards)
        soc_history.append(episode_soc_history)  # Extend the global soc_history list
        Pc_t_history.append(episode_Pc_t_history)
        Pd_t_history.append(episode_Pd_t_history)
        penalty_history.append(episode_penalty_history)
        battery_penalty_history.append(episode_battery_penalty_history)
        xD_t_history.append(episode_xD_t_history)

        
        reward_list.append(episode_reward)
        q_value_list.append((episode_q_values))
        
        print(f"Episode {i + 1}: Reward : {episode_reward}")
        print(f"Episode {i + 1}: Reward_discount : {episode_reward_discount}")
        
        # Log the reward
        # writer.add_scalar('Initial_Q_value', initial_q_value, i)
        writer.add_scalar('Rewards', episode_reward, i)
        writer.add_scalar('Q_value', max_q_value, i)
        writer.add_scalar('discount_Rewards', episode_reward_discount, i)
        writer.add_scalar('discount_Rewards', episode_reward_discount, i)
        writer.add_scalar('noise', noise, i)
        
    writer.close()
    
    # Save step rewards to a CSV file
    step_rewards_df = pd.DataFrame(step_rewards_list)
    step_rewards_df.to_csv(f'action/step_rewards_{current_datetime}.csv', index=False)
    # torch.save(Q.state_dict(),f'Q_net/Q_net_{current_datetime}.pth')
    # torch.save(Q_target.state_dict(),f'Q_net/Q_target_net_{current_datetime}.pth')
    
    bidding_df = pd.DataFrame(bidding_history)
    scaling_df = pd.DataFrame(scaling_history)
    soc_df = pd.DataFrame(soc_history)
    Pc_df = pd.DataFrame(Pc_t_history)
    Pd_df = pd.DataFrame(Pd_t_history)
    penalty_df = pd.DataFrame(penalty_history)
    battery_penalty_df = pd.DataFrame(battery_penalty_history)
    xD_t_history_df =pd.DataFrame(xD_t_history)
    
    # Save each DataFrame to CSV files
    bidding_df.to_csv(f'action/episode_bidding_{current_datetime}.csv', index=False)
    scaling_df.to_csv(f'action/episode_scaling_{current_datetime}.csv', index=False)
    soc_df.to_csv(f'action/episode_soc_{current_datetime}.csv', index=False)
    Pc_df.to_csv(f'action/episode_Pc_{current_datetime}.csv', index=False)
    Pd_df.to_csv(f'action/episode_Pd_{current_datetime}.csv', index=False)
    penalty_df.to_csv(f'action/episode_penalty_{current_datetime}.csv', index=False)
    battery_penalty_df.to_csv(f'action/episode_battery_penalty_{current_datetime}.csv', index=False)
    xD_t_history_df.to_csv(f'action/episode_battery_xDt_{current_datetime}.csv', index=False)
    