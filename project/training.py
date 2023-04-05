from epidemic_env.dynamics import *
from epidemic_env.agent import *
from epidemic_env.env import *

import numpy as np
import torch

from tqdm import tqdm

from utils import TrainResult
from utils import SEED

#######################################################################################
# SEEDING
#######################################################################################

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

#######################################################################################
# TRAINING
#######################################################################################

def train(agent, savepath, n_episodes=500, update_target_net_every=5, evaluate_every=50):
    # switch to training mode
    agent.policy_net.train()
    agent.target_net.eval()
    
    # logs
    logs = []
    eval_trace = []
    train_trace = []
    deaths = []
    
    best_eval_mean_reward = - np.inf
    
    for ep_num in tqdm(range(n_episodes)):
        logs.append([])
        
        if ep_num % evaluate_every == 0:
            mean_reward = agent.eval_model().cum_rewards.mean() # agent.eval_model() switches policy_net to eval mode
            print("mean reward: ", mean_reward)
            eval_trace.append(mean_reward)
            agent.policy_net.train() # switch policy_net back to training mode
            
            if mean_reward >= best_eval_mean_reward:
                agent.save_model(savepath)
                best_eval_mean_reward = mean_reward
                print("AGENT SAVED")

        # initialize the environment
        state, _ = agent.env.reset(SEED+ep_num)
                
        # episode
        done = False
        episode_total_reward = 0
        while not done:
            action = agent.act(state, ep_num)
            next_state, reward, done, info = agent.env.step(action)
            
            # logs
            logs[ep_num].append(info)
            episode_total_reward += reward.item()
            
            # store the transition in memory
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            
            # optimization
            agent.optimize_model()
            
        train_trace.append(episode_total_reward)
        deaths.append(info.total.dead) # variable `info` here is the info of the last episode step in the just finished episode
        
        # fully update target network
        if ep_num != 0 and ep_num % update_target_net_every == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.target_net.eval()

    # run evaluation at end of training
    mean_reward = agent.eval_model().cum_rewards.mean() # agent.eval_model() switches policy_net to eval mode
    eval_trace.append(mean_reward)
    print("mean reward: ", mean_reward)
    
    if mean_reward >= best_eval_mean_reward:
        agent.save_model(savepath)
        best_eval_mean_reward = mean_reward
        print("AGENT SAVED")    

    # process the logs
    n_confinement_days = np.array([np.array([info.action["confinement"] for info in log]).sum() * 7 for log in logs])
    n_isolation_days = np.array([np.array([info.action["isolation"] for info in log]).sum() * 7 for log in logs])
    n_hospital_days = np.array([np.array([info.action["hospital"] for info in log]).sum() * 7 for log in logs])
    n_vaccinate_days = np.array([np.array([info.action["vaccinate"] for info in log]).sum() * 7 for log in logs])
    
    deaths = np.array(deaths)
    
    eval_trace = np.array(eval_trace)
    train_trace = np.array(train_trace)
    
    train_result = TrainResult(train_trace, eval_trace, deaths, n_confinement_days, n_isolation_days, n_hospital_days, n_vaccinate_days, logs)
    
    return train_result