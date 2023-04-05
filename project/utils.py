from collections import namedtuple

import torch
import numpy as np

import matplotlib.pyplot as plt

SEED = 301046

#######################################################################################
# SEEDING
#######################################################################################

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

#######################################################################################
# GLOBALS
#######################################################################################

__ALL_ACTIONS = ["confinement", "isolation", "hospital", "vaccinate", "do-nothing"]
ACTIONS = dict(zip(__ALL_ACTIONS[:-1], range(len(__ALL_ACTIONS[:-1]))))
TOGGLE_ACTIONS = dict(zip(__ALL_ACTIONS, range(len(__ALL_ACTIONS))))
TOGGLE_IDS = {v:k for k, v in TOGGLE_ACTIONS.items()}

HEALTH_STATES_CMAP = {
    "exposed": "orange",
    "infected": "magenta",
    "dead": "red",
    "suceptible": "blue",
    "recovered": "green"
}

IDLE_ACTION = dict(zip(ACTIONS.keys(), [False] * len(ACTIONS.keys())))

#######################################################################################
# NAMED TUPLES
#######################################################################################

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

EvalResult = namedtuple("EvalResult", ["cum_rewards", "deaths", "q_values"] + [action+"_days" for action in ACTIONS.keys()] + ["logs"])

TrainResult = namedtuple("TrainResult", ["train_trace", "eval_trace", "deaths"] + [action+"_days" for action in ACTIONS.keys()] + ["logs"])

#######################################################################################
# BEST MODEL
#######################################################################################

def get_best_model(models, eval_traces):
    best_model_idx = np.vstack(eval_traces).max(axis=1).argmax()
    return models[best_model_idx], best_model_idx

#######################################################################################
# DATA EXTRACTION
#######################################################################################

def extract_totals(log, health_states):
    totals = {}
    for s in health_states:
        totals[s] = []
    for info in log:
        for s in health_states:
            totals[s].append(getattr(info.total, s))
    return totals

def extract_city_totals(log, health_states, cities):
    totals = {}
    for city in cities:
        totals[city] = {}
        for s in health_states:
            totals[city][s] = []
    for info in log:
        for city_name in cities:
            city_info = info.city[city_name]
            for s in health_states:
                totals[city_name][s].append(getattr(city_info, s))
    return totals

def extract_actions(log, actions):
    extracted_actions = np.zeros((len(actions), len(log)))
    for i, info in enumerate(log):
        for k in actions.keys():
            action_id = actions[k]
            extracted_actions[action_id][i] = int(info.action[k])
    return extracted_actions

#######################################################################################
# VISUALIZATION
#######################################################################################

def hist_avg(ax, data, title):
    ymax = 50
    if title == 'deaths':
        x_range = (1000,200000)
    elif title == 'cumulative rewards': 
        x_range = (-300,300)
    elif 'days' in title:
        x_range = (0,200)
    else:
        raise ValueError(f'{title} is not a valid title')
    
    ax.set_title(title)
    ax.set_ylim(0,ymax)
    ax.vlines([np.mean(data)],0,ymax,color='red')
    ax.hist(data,bins=60,range=x_range)
    
def training_traces_scatterplot(training_traces, dots_size=2, figsize=(10, 10)):
    _, ax = plt.subplots(1, 1, figsize=figsize)
    
    for i, trace in enumerate(training_traces):
        ax.scatter(range(len(trace)), trace, s=dots_size, label="Training Trace %d"%(i+1))
        
    ax.set_title("Training Traces")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    
def avg_eval_trace_lineplot(eval_traces, eval_every=50, figsize=(10, 10)):
    avg_eval_trace = np.vstack(eval_traces).mean(0)
    
    _, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.plot(range(len(avg_eval_trace)), avg_eval_trace, c="blue")
    ax.set_title("Eval Trace")
    ax.set_xlabel(f"Evaluation Number (every {eval_every} episodes)")
    ax.set_ylabel("Cumulative Reward")

def episode_totals_plot(log, health_states=HEALTH_STATES_CMAP.keys()):
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    totals = extract_totals(log, health_states=health_states)

    for health_state, values in totals.items():
        ax.plot(values, label=health_state, color=HEALTH_STATES_CMAP[health_state])
    
    ax.set_title("Total Count of the Population per Health State")
    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend()
 
def episode_city_totals_plot(log, health_states, cities):
    fig, axs = plt.subplots(9, 1, figsize=(10, 30))
    fig.subplots_adjust(hspace=0.5)
    
    city_totals = extract_city_totals(log, health_states=health_states, cities=cities)

    for city_name, ax in zip(cities, axs):
        for health_state in health_states:
            ax.plot(city_totals[city_name][health_state], label=health_state, color=HEALTH_STATES_CMAP[health_state])
            ax.set_title(f"{city_name}")
            ax.set_xlabel("week")
            ax.set_ylabel("count")
            ax.legend()
 
def episode_actions_plot(log, actions, title=""):
    actions_taken = extract_actions(log, actions)
    
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    ax.imshow(actions_taken.astype(np.uint8), aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(actions.keys());


def show_qvals_heatmap(q_values, action_dict):
    nb_actions = len(action_dict)
    q_values_matrix = np.zeros((len(q_values), nb_actions))

    for i in range(len(q_values)):
        q_values_matrix[i] = q_values[i]

    # add legend for each action on y axis using action_dict at the bottom of the plot
    plt.legend([action_dict[i] for i in range(nb_actions)], loc='lower right')
    
    # plot the heatmap
    plt.imshow(q_values_matrix.T, cmap='coolwarm', aspect='auto')

    # show y ticks for each action
    plt.yticks(range(nb_actions))


    # use action_dict as y_ticks labels
    plt.gca().set_yticklabels([action_dict[i] for i in range(nb_actions)])







    plt.xlabel("Week")
    plt.ylabel("Action")
    plt.colorbar()
    plt.show()
