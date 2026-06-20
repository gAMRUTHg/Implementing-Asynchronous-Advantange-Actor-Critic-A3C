import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

mp.set_start_method('spawn', force = True)
DEVICE = 'cpu'

#---------------------------------------------------------------------------------------#



# PARAMETERS

ENV            = "CartPole-v1"
OBS_DIM        = 4
NUM_ACTIONS    = 2
HIDDEN_DIM     = 128
NUM_WORKERS    = 4
MAX_EPISODES   = 3000
GAMMA          = 0.99
N_STEPS        = 10
LR             = 3e-4
VALUE_COEF     = 0.5

#---------------------------------------------------------------------------------------#



# CLASS A3C

class A3C(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim):
        super().__init__()
        self.shared  = nn.Linear(obs_dim, hidden_dim)
        self.actor   = nn.Linear(hidden_dim, num_actions)
        self.critic  = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        hidden       = F.relu(self.shared(x))
        action_probs = F.softmax(self.actor(hidden), dim = -1)
        state_value  = self.critic(hidden)
        return action_probs, state_value
    def get_action(self, obs_numpy):
        obs_tensor   = torch.FloatTensor(obs_numpy)
        probs, value = self.forward(obs_tensor)
        dist         = torch.distributions.Categorical(probs)
        action       = dist.sample()
        return action.item(), dist.log_prob(action), value

#---------------------------------------------------------------------------------------#



# GRADIENTS COMPUTATION

def compute_grads(local_net, experiences, gamma, value_coef, next_obs, done):
    if done : R = 0.0
    else :
        with torch.no_grad():
            obs_t = torch.FloatTensor(next_obs)
            wasted, next_val = local_net.forward(obs_t)
            R = next_val.item()
    log_probs = [];    values    = []
    rewards   = [];    returns   = []
    grads     = [];
    for (lp, val, rew) in experiences: 
        log_probs.append(lp)
        values.append(val)
        rewards.append(rew)
    for r in reversed(rewards):
        R = r + gamma*R
        returns.append(R)
    returns.reverse()
    returns   = torch.FloatTensor(returns)
    values    = torch.cat(values).reshape(-1)
    log_probs = torch.stack(log_probs)
    advantages = returns - values.detach()
    actor_loss  = -(log_probs * advantages).mean()
    critic_loss = F.mse_loss(values, returns)
    total_loss = actor_loss + value_coef * critic_loss
    local_net.zero_grad()
    total_loss.backward()
    for param in local_net.parameters():
        if param.grad is not None: grads.append(param.grad.clone())
        else : grads.append(torch.zeros_like(param))
    return grads

#---------------------------------------------------------------------------------------#


    
# WORKER FUNCTION 

def worker_fn(worker_id, global_net, optimizer, lock, result_queue, max_episodes):
    env = gym.make(ENV)
    local_net = A3C(OBS_DIM, NUM_ACTIONS, HIDDEN_DIM)
    for episode in range(max_episodes):
        local_net.load_state_dict(global_net.state_dict())
        obs, waste = env.reset()
        done = False
        ep_reward = 0.0
        experiences = []         # [(log_probs, value, reward) for N_STEPS many times]
        while not done:
            action, log_prob, value = local_net.get_action(obs)
            next_obs, reward, terminated, truncated, waste = env.step(action)
            done = terminated or truncated
            experiences.append((log_prob, value, reward))
            ep_reward += reward
            obs = next_obs
            if len(experiences) == N_STEPS or done:
                grads = compute_grads(local_net, experiences, GAMMA, VALUE_COEF, next_obs, done)
                with lock:
                    optimizer.zero_grad()
                    for global_param, grad in zip(global_net.parameters(), grads):
                        global_param.grad = grad.clone();
                    optimizer.step()
                local_net.load_state_dict(global_net.state_dict())
                experiences = []
        result_queue.put((worker_id, episode, ep_reward))
    env.close()
    result_queue.put(None)

#---------------------------------------------------------------------------------------#



# TRAINING LOOP

def train():
    global_net = A3C(OBS_DIM, NUM_ACTIONS, HIDDEN_DIM)
    global_net.share_memory()
    optimizer = torch.optim.Adam(global_net.parameters(), lr = LR)
    lock = mp.Lock()
    result_queue = mp.Queue()
    processes = []
    for wid in range(NUM_WORKERS):
        p = mp.Process(target = worker_fn, args = (wid, global_net, optimizer, lock, result_queue, MAX_EPISODES))
        p.start(); processes.append(p)
    all_rewards = []
    done_count = 0
    while done_count < NUM_WORKERS:
        result = result_queue.get()
        if result is None:
            done_count += 1
            print(f"{done_count} workers done")
        else :
            worker_id, episode, reward = result
            all_rewards.append((worker_id, episode, reward))
            if len(all_rewards) % 200 == 0:
                recent = [r for _, _, r in all_rewards[-100:]]
                print(f"[{len(all_rewards)} episodes]  avg last 100: {np.mean(recent):.1f}")
    for p in processes: p.join()
    print("training complete")
    return global_net, all_rewards

#---------------------------------------------------------------------------------------#



# PLOT FUNCTION 

def plot_rewards(all_rewards):
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    ax = axes[0]
    for wid in range(NUM_WORKERS):
        worker_data = sorted([(ep, rew) for (w, ep, rew) in all_rewards if w == wid])
        episodes = [x[0] for x in worker_data]
        rewards  = [x[1] for x in worker_data]
        ax.plot(episodes, rewards, alpha = 0.4, label = f"worker {wid}")
    ax.set_xlabel("episode");       ax.set_ylabel("reward") 
    ax.set_title("per-worker episode rewards"); ax.legend()
    ax = axes[1]
    all_r = [rew for (_, _, rew) in all_rewards]
    window = 50
    smoothed = [np.mean(all_r[max(0, i-window) : i+1]) for i in range(len(all_r))]
    ax.plot(smoothed, color = 'blue')
    ax.axhline(y=475, color = 'red', linestyle = '--', label = 'solved threshold')
    ax.set_xlabel("episode index (all workers, arrival order)")
    ax.set_ylabel(f"smoothed reward (window={window})")
    ax.set_title("smoothed reward"); ax.legend()
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------------------------#



# MAIN FUNCTION 

if __name__ == '__main__':
    t0 = time.time()
    trained_net, all_rewards = train()
    print(f"total time: {time.time()-t0:.1f}s")
    plot_rewards(all_rewards)