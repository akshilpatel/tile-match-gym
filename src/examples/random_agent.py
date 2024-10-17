import os
import pickle
from collections import defaultdict
import json
import gymnasium as gym
import numpy as np

from tile_match_gym.tile_match_env import TileMatchEnv
from tile_match_gym.wrappers import ProportionRewardWrapper
import pandas as pd

def run_random_episode(env, rng, use_effective_actions=False):
    obs, info = env.reset()
    total_reward = 0
    env_num_effective_actions = 0
    env_num_effective_actions += len(info["effective_actions"])
    while True:
        if not use_effective_actions:
            action = rng.integers(0, env.unwrapped.num_actions)
        else:
            action = rng.choice(info["effective_actions"])

        next_obs, reward, done, _, info = env.step(action)

        env_num_effective_actions += len(info["effective_actions"])

        total_reward += reward
        if done:
            break

    return total_reward, env_num_effective_actions


def run_random(env, seed:int, num_episodes:int = 1000, use_effective_actions=False):
    rng = np.random.default_rng(seed)
    epi_r = np.zeros(num_episodes)
    num_effective_actions_arr = np.zeros(num_episodes)
    env_num_effective_actions_arr = np.zeros_like(num_effective_actions_arr)
    for i in range(num_episodes):
        total_reward, env_num_effective_actions = run_random_episode(env, rng, use_effective_actions)
        epi_r[i] = total_reward
        env_num_effective_actions_arr[i] = env_num_effective_actions
    return epi_r, env_num_effective_actions_arr

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results that can be serialized as JSON
    (r, env_eff_a) = results

    json_results = {
        "r": r.tolist(),
        "env_num_effective_actions": env_eff_a.tolist(),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(json_results, f)


def extract_random_results(num_rows, num_cols, num_colours, num_moves, use_effective_actions):

    random_path = f"../../results/{num_rows}_{num_cols}_{num_colours}_{num_moves}"
    if use_effective_actions:
        random_path += "_effective_actions"

    random_path = f"{random_path}/results.json"
    results = json.load(open(random_path, "r"))

    epi_rewards = np.array(results["r"])
    epi_env_eff_a = np.array(results["env_num_effective_actions"])

    stats = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_colours": num_colours,
        "num_moves": num_moves,
        "use_effective_actions": use_effective_actions,
        "epi_rewards_mean": np.mean(epi_rewards),
        "epi_rewards_std": np.std(epi_rewards),
        "env_eff_a_mean": np.mean(epi_env_eff_a / num_moves),
        "env_eff_a_std": np.std(epi_env_eff_a / num_moves),
    }

    df = pd.DataFrame([stats])
    print(df.T)
    print("--------")
    return df
    
def run_random_baseline(num_episodes, num_rows, num_cols, num_colours, num_moves, use_effective_actions=False):
    output_dir = f"../../results/{num_rows}_{num_cols}_{num_colours}_{num_moves}_specials"
    if use_effective_actions:
        output_dir = output_dir + "_effective_actions"

    env = gym.make("TileMatch-v0", num_rows=num_rows, num_cols=num_cols, num_colours=num_colours, num_moves=num_moves, colour_specials=["vertical_laser"], colourless_specials=[], seed=0)
    env = ProportionRewardWrapper(env)
    results = run_random(env, 0, num_episodes, use_effective_actions)
    save_results(results, output_dir)



if __name__ == "__main__":
    combos = [
        (3, 3, 2, 5),
        (3, 3, 2, 10),
        (4, 4, 3, 5),
        (4, 4, 3, 10),
        (5, 5, 3, 5),
        (5, 5, 3, 10),
        (5, 5, 4, 5),
        (5, 5, 4, 10),
        (6, 6, 3, 5),
        (6, 6, 3, 10),
        (6, 6, 4, 5),
        (6, 6, 4, 10),
        (7, 7, 3, 5),
        (7, 7, 3, 10),
        (7, 7, 4, 5),
        (7, 7, 4, 10),
        (8, 8, 3, 5),
        (8, 8, 3, 10),
        (8, 8, 4, 5),
        (8, 8, 4, 10),
        (9, 9, 4, 5),
        (9, 9, 4, 10),
        (9, 9, 5, 5),
        (9, 9, 5, 10),
        (10, 10, 4, 10),
        (10, 10, 5, 10),
        (15, 15, 5, 10),
        (15, 15, 7, 10),
        (20, 20, 10, 10),
        (35, 35, 17, 10)
    ]
    # combos = [combos[-1]]
    for (num_rows, num_cols, num_colours, num_moves) in combos:
        print(num_rows, num_cols, num_colours, num_moves)
        run_random_baseline(3000, num_rows, num_cols, num_colours, num_moves, False)
        run_random_baseline(3000, num_rows, num_cols, num_colours, num_moves, True)


    for (num_rows, num_cols, num_colours, num_moves) in combos:
        extract_random_results(num_rows, num_cols, num_colours, num_moves, False)
        extract_random_results(num_rows, num_cols, num_colours, num_moves, True)



