from epidemic_env.dynamics import ModelDynamics
from epidemic_env.env import Observation

import torch
import numpy as np

from copy import deepcopy

from utils import TOGGLE_ACTIONS, TOGGLE_IDS, ACTIONS, IDLE_ACTION
from utils import SEED

#######################################################################################
# SEEDING
#######################################################################################

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

#######################################################################################
# ENVIRONMENT PREPROCESSORS
#######################################################################################

def russo_acton_preprocessor(a: torch.Tensor, dyn: ModelDynamics):
    return {
        'confinement': a.item(), 
        'isolation': False, 
        'hospital': False, 
        'vaccinate': False
    }

def russo_observation_preprocessor(obs: Observation, dyn: ModelDynamics):
    total_infected = obs.total.infected[-1] # number of total infected people at the last day of the week
    return torch.Tensor([total_infected])


def dqn_action_preprocessor(a:torch.Tensor, dyn: ModelDynamics, action_confine=1):
    action = deepcopy(IDLE_ACTION)
    if a.item() == action_confine:
        action['confinement'] = True
        
    return action
    
def dqn_observation_preprocessor(obs: Observation, dyn: ModelDynamics, scale=100):
    infected = scale * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
    dead = scale * np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities])
    confined = np.ones_like(dead) * int((dyn.get_action()['confinement']))
    return torch.pow(torch.Tensor(np.stack((infected, dead, confined))), 0.25).unsqueeze(0)


def toggleaction_dqn_action_preprocessor(a: torch.Tensor, dyn: ModelDynamics):
    switched = TOGGLE_IDS[a.item()]
    action = deepcopy(dyn.get_action()) # the previous action
    
    # if `do nothing`, return the previous action as is
    if "do-nothing" == switched:
        return action
    
    # otherwise, update and return the previous action
    action[switched] = bool((action[switched] + 1) % 2)
        
    return action
    
def toggleaction_dqn_observation_preprocessor(obs: Observation, dyn: ModelDynamics, scale=100):
    # process the observation
    infected = (scale * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities]).reshape(-1)) ** 0.25
    dead = (scale * np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities]).reshape(-1)) ** 0.25
    
    # process the action
    encoded_action = np.zeros(dyn.ACTION_CARDINALITY, dtype=np.float32)
    taken_action = dyn.get_action()
    for action, is_active in taken_action.items():
        encoded_action[TOGGLE_ACTIONS[action]] = is_active
    
    return torch.Tensor(np.hstack((encoded_action, infected, dead))).unsqueeze(0)


def multiaction_dqn_factorized_qvals_action_preprocessor(a: torch.Tensor, dyn:ModelDynamics):
    # a is a tensor of shape (1, nb_actions), for each action there is either a 0 or a 1
    # 0 if the action is not taken, 1 if the action is taken
    a = a.flatten()
    
    action = deepcopy(IDLE_ACTION)    
    for action_name, action_id in ACTIONS.items():
        action[action_name] = bool(a[action_id].item())
    
    return action
