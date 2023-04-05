from epidemic_env.dynamics import *
from epidemic_env.agent import *
from epidemic_env.env import *

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from collections import deque

import random

from utils import Transition, EvalResult

from utils import SEED

#######################################################################################
# SEEDING
#######################################################################################

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

#######################################################################################
# AGENTS
#######################################################################################

class RussoAgent(Agent):
    def __init__(self, env, confinement_duration):
        super().__init__(env)
        
        self.env = env
        
        self.confinement_duration = confinement_duration
        self.n_weeks_in_confinement = 0
        
    def load_model(self, savepath:str):
        raise NotImplemented("`load_model` function is not and should not be implemented for a Russo agent.")
        
    def save_model(self, savepath:str):
        raise NotImplemented("`save_model` function is not and should not be implemented for a Russo agent.")
    
    def optimize_model(self):
        raise NotImplemented("`optimize_model` function is not and should not be implemented for a Russo agent.")

    def eval_model(self, n_episodes=50):
        # logs
        logs = []
        cum_rewards = []
        deaths = []
    
        # episodes
        for ep_num in range(n_episodes):
            # reset
            state, _ = self.env.reset(SEED + ep_num)
            self.reset()
            logs.append([])

        
            # start of episode
            cum_reward = 0
            done = False
            while not done:
                action = self.act(state)
                state, reward, done, info = self.env.step(action)
                cum_reward += reward.item()
                logs[ep_num].append(info)
            
            cum_rewards.append(cum_reward)
            deaths.append(info.total.dead) # deaths on the final day of the just finished episode
    
        # create the evaluation result
        n_confinement_days = np.array([np.array([info.action["confinement"] for info in log]).sum() * 7 for log in logs])
        n_isolation_days = np.array([np.array([info.action["isolation"] for info in log]).sum() * 7 for log in logs])
        n_hospital_days = np.array([np.array([info.action["hospital"] for info in log]).sum() * 7 for log in logs])
        n_vaccinate_days = np.array([np.array([info.action["vaccinate"] for info in log]).sum() * 7 for log in logs])
    
        cum_rewards = np.array(cum_rewards)
        deaths = np.array(deaths)
    
        eval_result = EvalResult(
            cum_rewards, deaths, None, 
            n_confinement_days, n_isolation_days, n_hospital_days, n_vaccinate_days, logs)
    
        return eval_result

    def reset(self):
        self.n_weeks_in_confinement = 0
        
    def act(self, obs:torch.Tensor):
        self.n_weeks_in_confinement = self.n_weeks_in_confinement % self.confinement_duration
        confine = int(obs.item() > 20_000) or (self.n_weeks_in_confinement > 0)
        self.n_weeks_in_confinement += int(confine)
        return torch.Tensor([confine])

    
class DQNAgent(Agent):
    
    class DQN(nn.Module):    
        def __init__(self, n_observations, n_actions):
            super().__init__()
        
            self.nn_layers = nn.Sequential(
                nn.Linear(n_observations, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, n_actions)
            )
        
        def forward(self, states):
            batch_size = states.shape[0]
            states = states.view(batch_size, -1)
            q_values = self.nn_layers(states)
            return q_values
    
    
    class ReplayMemory(object):
        
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            self.memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)
    
    
    def __init__(self, env, batch_size=2048, gamma=0.9, eps=0.7, lr=5e-3, buffer_size=20000):
        super().__init__(env)

        self.batch_size = batch_size
        
        # discount rate
        self.gamma = gamma
        
        # exploration
        self.eps = eps

        # environment, number of actions and observations
        self.env = env
        self.n_actions = env.action_space.n
        state, _ = env.reset(SEED)
        self.n_observations = torch.numel(state)
        
        # networks
        self.device = torch.device("cpu")
        self.policy_net = DQNAgent.DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQNAgent.DQN(self.n_observations, self.n_actions).to(self.device)
        self.policy_net.eval()
        self.target_net.eval()

        # optimizer
        self.optimizer = AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        
        # replay buffer
        self.memory = DQNAgent.ReplayMemory(buffer_size)

    def save_model(self, savepath):
        torch.save(self.policy_net.state_dict(), savepath)

    def load_model(self, savepath):
        self.policy_net.load_state_dict(torch.load(savepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()

    def reset(self):
        self.policy_net = DQNAgent.DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQNAgent.DQN(self.n_observations, self.n_actions).to(self.device)
        self.policy_net.eval()
        self.target_net.eval()
    
    def optimize_model(self):
        self.policy_net.train()
        
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute a mask of non-final states
        non_final_mask = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Compute Q-values of actions taken
        state_action_Q_batch = self.policy_net(state_batch).gather(1, action_batch)

        # Compute max(Q(s_{t+1}, .)) for the next actions s_{t+1}.
        max_next_state_Q_batch = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            max_next_state_Q_batch[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        max_next_state_Q_batch = max_next_state_Q_batch.view(-1, 1)
        
        # Compute the target
        target_batch = reward_batch + (max_next_state_Q_batch * self.gamma)

        # Compute Huber loss
        loss = self.criterion(state_action_Q_batch, target_batch)
        
        assert target_batch.size(0) == state_action_Q_batch.size(0) and target_batch.size(1) == state_action_Q_batch.size(1)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
 
    def eval_model(self, n_episodes=20, compute_qvals=False):
        self.policy_net.eval()
        
        # logs
        logs = []
        cum_rewards = []
        deaths = []
        q_values = [] if compute_qvals else None
    
        # episodes
        with torch.no_grad():
            for ep_num in range(n_episodes):
                # reset
                state, _ = self.env.reset(SEED + ep_num)
        
                logs.append([])
        
                # start of episode
                cum_reward = 0
                done = False
                while not done:
                    action = self.act(state, ep_num)
                    if compute_qvals:
                        with torch.no_grad():
                            q_values.append(self.policy_net(state).squeeze().cpu().numpy())
                    state, reward, done, info = self.env.step(action)
                    cum_reward += reward.item()
                    logs[ep_num].append(info)
            
                cum_rewards.append(cum_reward)
                deaths.append(info.total.dead) # deaths on the final day of the just finished episode
    
        # create the evaluation result
        n_confinement_days = np.array([np.array([info.action["confinement"] for info in log]).sum() * 7 for log in logs])
        n_isolation_days = np.array([np.array([info.action["isolation"] for info in log]).sum() * 7 for log in logs])
        n_hospital_days = np.array([np.array([info.action["hospital"] for info in log]).sum() * 7 for log in logs])
        n_vaccinate_days = np.array([np.array([info.action["vaccinate"] for info in log]).sum() * 7 for log in logs])
    
        cum_rewards = np.array(cum_rewards)
        deaths = np.array(deaths)
        q_values = np.array(q_values)
    
        eval_result = EvalResult(
            cum_rewards, deaths, q_values,
            n_confinement_days, n_isolation_days, n_hospital_days, n_vaccinate_days, logs)
        
        return eval_result
        
    def act(self, state, episode_num=None):
        eps_threshold = self.eps if self.policy_net.training else 0
        if random.random() >= eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(-1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)


class DQNAgentWithExplorationDecay(DQNAgent):
    def __init__(self, env, batch_size=2048, gamma=0.9, eps=0.7, lr=1e-5, buffer_size=20000, eps_min=0.2, ep_num_max=500):
        super().__init__(env, batch_size, gamma, eps, lr, buffer_size)
        
        # exploration parameters
        self.eps_min = eps_min
        self.ep_num_max = ep_num_max
          
    def act(self, state, episode_num):
        eps_threshold = max(self.eps * (self.ep_num_max - episode_num) / self.ep_num_max, self.eps_min)
        eps_threshold = eps_threshold if self.policy_net.training else 0
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(-1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        

class DQNAgentWithFactorizedQvals(DQNAgentWithExplorationDecay):
    
    class DQNWithFactorizedQvals(nn.Module):   

        def __init__(self, n_observations, n_actions):
            super().__init__()
            self.nn_layers = nn.Sequential(
                nn.Linear(n_observations, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                # n_actions * 2 because we have (false, true) for each action
                # the output is (action_1_false, action_1_true, action_2_false, action_2_true, ...)
                nn.Linear(16, n_actions * 2)
            )

        def forward(self, states):
            batch_size = states.shape[0]
            states = states.view(batch_size, -1)
            q_values = self.nn_layers(states)
            return q_values
    

    def __init__(self, env, batch_size=2048, gamma=0.9, eps=0.7, lr=1e-5, buffer_size=20000, eps_min=0.2, ep_num_max=500):
        super().__init__(env, batch_size, gamma, eps, lr, buffer_size, eps_min, ep_num_max)

        # initialize the networks
        self.policy_net = DQNAgentWithFactorizedQvals.DQNWithFactorizedQvals(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQNAgentWithFactorizedQvals.DQNWithFactorizedQvals(self.n_observations, self.n_actions).to(self.device)
        self.policy_net.eval()
        self.target_net.eval()

        # optimizer
        self.optimizer = AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        
    @staticmethod
    def __successive_max(qvals_batch, n=2):
        # computes the max among each group of n successive elements along axis 1 of a 2D tensor
        # and returns a tensor with these maximum values
        
        assert n <= qvals_batch.shape[1], "`n` cannot be larger than the number of actions"
        assert qvals_batch.shape[1] % n == 0, "`n` must divide the number of actions"
        
        # initialize the indices
        start = 0
        end   = n
        
        # generate the first column
        max_tensor, max_idx_tensor = qvals_batch[:, start:end].max(1, keepdims=True)
        start = end
        end += n
        # then the rest of the columns
        while end <= qvals_batch.shape[1]:
            max_col, max_idx_col = qvals_batch[:, start:end].max(1, keepdims=True)
            max_tensor = torch.cat((max_tensor, max_col), dim=1)
            max_idx_tensor = torch.cat((max_idx_tensor, start + max_idx_col), dim=1)
            start = end
            end += n
        
        return max_tensor, max_idx_tensor
            
    def optimize_model(self):
        self.policy_net.train()
   
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute a mask of non-final states
        non_final_mask = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Q_val is computed by sum of the Q_values of the actions taken
        
        # each action in the batch is of the form [0, 1, 0, 1] for each four actions
        # we need to select the Q_values of the actions taken and sum them
        # we have to transform the [0, 1, 0, 1] into [0, 3, 4, 7]
        action_idx_batch = torch.zeros(action_batch.shape, device=self.device, dtype=torch.long)        
        for i in range(action_batch.shape[1]):
            action_idx_batch[:, i] = action_batch[:, i] + 2 * i

        # Compute the Q-values of the actions taken.
        state_action_Q_batch = self.policy_net(state_batch).gather(1, action_idx_batch).sum(dim=1, keepdims=True)

        # Compute max(Q(s_{t+1}, .)) for the next actions s_{t+1}
        max_next_state_Q_batch = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            target_qvals =  self.target_net(non_final_next_states)
        target_max_qvals = DQNAgentWithFactorizedQvals.__successive_max(target_qvals, n=2)[0]
        target_max_qvals_sum = target_max_qvals.sum(dim=1)
        max_next_state_Q_batch[non_final_mask] = target_max_qvals_sum
        max_next_state_Q_batch = max_next_state_Q_batch.view(-1, 1)
        
        # Compute the target
        target_batch = reward_batch + (max_next_state_Q_batch * self.gamma)

        # Compute Huber loss
        loss = self.criterion(state_action_Q_batch, target_batch)
        
        assert target_batch.size(0) == state_action_Q_batch.size(0) and target_batch.size(1) == state_action_Q_batch.size(1)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def act(self, state, episode_num):
        eps_threshold = max(self.eps * (self.ep_num_max - episode_num) / self.ep_num_max, self.eps_min)
        eps_threshold = eps_threshold if self.policy_net.training else 0
        
        # epsilon-greedy
        if random.random() > eps_threshold:
            with torch.no_grad():
                qvals = self.policy_net(state)
                assert qvals.shape[1] % 2 == 0, "The number of output neurons must be even. Each couple of neurons represents an action."
                
                # the actions are of the form 0/1 for each action
                # 0 if the action should be false, 1 if the action should be true
                # even Q-value indices correspond to active actions, while even ones correspond to inactive ones
                # so if arg-successive-max is even, the action is active, otherwise  it is inactive
                action = (DQNAgentWithFactorizedQvals.__successive_max(qvals, n=2)[1]) % 2 
                action1 = action

                return action1
            
        else:
            random_action = torch.tensor([[random.randint(0, 1) for _ in range(self.n_actions)]], device=self.device, dtype=torch.long)
            return random_action
