import os
import pickle
from collections import defaultdict
import json
import gymnasium as gym
import numpy as np

from tile_match_gym.tile_match_env import TileMatchEnv
from tile_match_gym.wrappers import ProportionRewardWrapper

def run_random_episode(env, obs_seen, rng, use_effective_actions=False):
    obs, info = env.reset()
    total_reward = 0
    num_effective_actions = 0
    
    obs_seen[tuple(obs["board"].flatten().tolist() + [obs["num_moves_left"]])] += 1
    while True:
        if not use_effective_actions:
            action = env.action_space.sample()
        else:
            action = rng.choice(info["effective_actions"])



        next_obs, reward, done, _, info = env.step(action)
        if len(info["effective_actions"]) == 0 and not done:
            print(next_obs["board"], next_obs["num_moves_left"])
        obs_seen[tuple(next_obs["board"].flatten().tolist() + [next_obs["num_moves_left"]])] += 1
        
        num_effective_actions += int(reward > 0)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward, num_effective_actions, obs_seen


def run_random(env, seed:int, num_episodes:int = 1000, use_effective_actions=False):
    epi_r = np.zeros(num_episodes)
    obs_seen = defaultdict(int)
    rng = np.random.default_rng(seed)
    num_effective_actions_arr = np.zeros(num_episodes)
    
    for i in range(num_episodes):
        total_reward, num_effective_actions, obs_seen = run_random_episode(env, obs_seen, rng, use_effective_actions)
        epi_r[i] = total_reward
        num_effective_actions_arr[i] = num_effective_actions
    return epi_r, num_effective_actions_arr, obs_seen

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results that can be serialized as JSON
    r = results["r"]
    eff_a = results["eff_a"]
    obs_seen = results["obs_seen"]
    json_results = {
        "r": r.tolist(),
        "num_effective_actions": eff_a.tolist(),
        "obs_seen": {str(k): v for k, v in obs_seen.items()}
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(json_results, f)
    
    # Save results that cannot be serialized as JSON using pickle
    pickle_results = {k: v for k, v in results.items() if k not in json_results}
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(pickle_results, f)
    
def run_random_baseline(num_episodes, output_dir, num_repeats, use_effective_actions=False):
    num_moves = 5
    r_aucs = []
    output_dir = f"{output_dir}/"

    for repeat in range(num_repeats):
        print(repeat, "---------------")
        r_dir = output_dir + f"/repeat_{repeat}"
        env = gym.make("TileMatch-v0", num_rows=5, num_cols=5, num_colours=4, num_moves=num_moves, colour_specials=[], colourless_specials=[], seed=repeat)
        env = ProportionRewardWrapper(env)
        r, eff_a, obs_seen = run_random(env, repeat, num_episodes, use_effective_actions)
        save_results({"r": r, "eff_a": eff_a, "obs_seen": obs_seen}, r_dir)
        r_aucs.append(np.trapz(r))


if __name__ == "__main__":
    run_random_baseline(1000, "../../results/random_5_5_4", 5, False)
    run_random_baseline(1000, "../../results/random_5_5_4_eff_a", 5, use_effective_actions=True)