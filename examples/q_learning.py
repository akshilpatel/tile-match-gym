import numpy as np
import gymnasium as gym
from copy import deepcopy
from numba import njit
from tile_match_gym.tile_match_env import TileMatchEnv
from collections import defaultdict
from tqdm import tqdm
from tile_match_gym.wrappers import ProportionRewardWrapper, OneHotWrapper
import os
import json
import pickle
import optuna

class QLearningAgent:
    def __init__(self, lr, epsilon_decay_dur, gamma, num_actions, rng):
        self.lr = lr
        self.epsilon_decay_dur = epsilon_decay_dur

        self.gamma = gamma
        self.num_actions = num_actions
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))
        self.rng = rng

    def choose_action(self, obs, effective_actions = None):
        s = self._preprocess_obs(obs)
        if self.rng.random() < self.epsilon:
            if effective_actions == None:
                return self.rng.choice(self.num_actions)
            else:
                return self.rng.choice(effective_actions)
        else:
            q_vals = self.q_table[s]
            if effective_actions != None:
                qs = q_vals[effective_actions]
                return effective_actions[self.rng.choice(np.flatnonzero(qs == qs.max()))]
            else:
                return self.rng.choice(np.flatnonzero(q_vals == q_vals.max()))
    def process_transition(self, obs, action, reward, next_obs, done):
        self.decay_epsilon()
        self.update(obs, action, reward, next_obs, done)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_dur

    def update(self, obs, action, reward, next_obs, done):
        s = self._preprocess_obs(obs)
        next_s = self._preprocess_obs(next_obs)
        q_vals = self.q_table[s]
        next_q_vals = self.q_table[next_s]
        q_target = reward + self.gamma * (1-done) * next_q_vals.max()
        self.q_table[s][action] += self.lr * ( q_target - q_vals[action])

    def _preprocess_obs(self, obs):
        board, num_moves = obs["board"], obs["num_moves_left"]
        o = board.flatten().tolist() + [num_moves]
        return tuple(o)
    

def run_episode(agent, env, obs_seen):
    obs, info = env.reset()
    done = False
    total_reward = 0
    num_effective_actions = 0
    
    obs_seen[agent._preprocess_obs(obs)] += 1
    while True:
        action = agent.choose_action(obs)
        next_obs, reward, done, _, info = env.step(action)
        obs_seen[agent._preprocess_obs(next_obs)] += 1
        agent.process_transition(obs, action, reward, next_obs, done)
        num_effective_actions += int(reward > 0)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward, num_effective_actions, obs_seen


def train(agent, env, num_episodes:int = 1000):
    epi_r = np.zeros(num_episodes)
    obs_seen = defaultdict(int)
    num_effective_actions_arr = np.zeros(num_episodes)
    print_eps = True
    for i in tqdm(range(num_episodes)):
        total_reward, num_effective_actions, obs_seen = run_episode(agent, env, obs_seen)
        if agent.epsilon > 0.1:
            agent.epsilon *= 0.9999
        elif print_eps:
            print(f"Epsilon is low at episode {i}")
            print_eps = False
        epi_r[i] = total_reward
        num_effective_actions_arr[i] = num_effective_actions
    return epi_r, num_effective_actions_arr, obs_seen, agent

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results that can be serialized as JSON
    json_results = {k: v for k, v in results.items() if isinstance(v, (list, dict, int, float, bool, str))}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(json_results, f)
    
    # Save results that cannot be serialized as JSON using pickle
    pickle_results = {k: v for k, v in results.items() if k not in json_results}
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(pickle_results, f)

if __name__ == "__main__":
    def execute_run(epsilon_decay_dur, gamma, lr, num_episodes, output_dir, seed, num_repeats):
        num_episodes=300_000
        num_moves = 10
        epsilon_decay_dur = int(num_episodes * num_moves * epsilon_decay_dur)
        r_aucs = []
        output_dir = f"results/gamma_{gamma}_lr_{lr}_eps_decay_dur_{epsilon_decay_dur}"

        for repeat in range(num_repeats):
            r_dir = output_dir + f"/repeat_{repeat}"
            env = gym.make("TileMatch-v0", num_rows=3, num_cols=3, num_colours=2, num_moves=10, colour_specials=[], colourless_specials=[], seed=seed)
            env = ProportionRewardWrapper(env)
            rng = np.random.default_rng(seed)
            agent = QLearningAgent(lr=lr, epsilon_decay_dur=epsilon_decay_dur, gamma=gamma, num_actions=env.num_actions, rng=rng)
                    
            r, eff_a, obs_seen, agent = train(agent, env, num_episodes)
            save_results({"r": r, "eff_a": eff_a, "obs_seen": obs_seen, "agent": agent}, output_dir)

            r_aucs.append(np.trapz(r))
        
        return np.mean(r_aucs)

    # Hyperparameter settings
    
    learning_rates = [0.01, 0.1, 0.25, 0.5]
    epsilon_decay_durs = [0.001, 0.1, 0.3, 0.5, 0.7, 0.9]
    num_episodes = 400_000
    num_repeats = 5

    # Run experiments
    
    

    pbounds = {
        "epsilon_decay_dur": (0.1, 0.9),
        "gamma": (0.7, 1),
        "lr": (0.01, 0.5),
        'epsilon_decay_dur': (0.001, 0.1, 0.3, 0.5, 0.7, 0.9),
        'gamma': (0.7, 0.8, 0.9, 0.95, 0.99),
        'lr': (0.01, 0.1, 0.25, 0.5),
    } # 120 combinations x 5 repeats = 600 experiments in total
    # 20mins per experiment -> 200 hours in total / 16 cores = 12.5 hours

# Metrics: H
# Effect of Episode Length
# Effect of gamma
# Effect of learning rate
# Different Epsilon 