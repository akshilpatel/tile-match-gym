import numpy as np

from tile_match_gym.tile_match_env import TileMatchEnv

def test_env_step():
    env = TileMatchEnv(3, 5, 3, 4, ["cookie"], ["bomb", "vertical_laser", "horizontal_laser"], seed=3)
    obs, info = env.reset()
    assert info == {'effective_actions': [4, 6, 8]}

    next_obs, reward, done, _, info = env.step(6)
    assert np.array_equal(next_obs["board"], np.array([[[2, 3, 1, 2, 1],
                                                        [2, 2, 3, 1, 2],
                                                        [3, 2, 1, 2, 3]],
                                                       [[1, 1, 1, 1, 1],
                                                        [1, 1, 1, 1, 1],
                                                        [1, 1, 1, 1, 1]]]))
    assert next_obs["num_moves_left"] == 3

    assert reward == 6
    assert not done
    assert info == {
        'is_combination_match': False,
        'num_new_specials': 0,
        'num_new_specials': 0,
        'num_specials_activated': 0,
        'shuffled': False,
        'effective_actions': [3, 10, 16, 17, 18]
        }
    

    next_obs, reward, done, _, info = env.step(16)

    assert np.array_equal(next_obs["board"], np.array([[[2, 3, 1, 3, 2],
                                                        [2, 2, 1, 2, 1],
                                                        [3, 1, 3, 3, 2]],
                                                       [[1, 1, 1, 1, 1],
                                                        [1, 1, 1, 1, 1],
                                                        [1, 1, 1, 1, 1]]]))
    
    assert next_obs["num_moves_left"] == 2
    assert reward == 18
    assert not done
    assert info == {
        'is_combination_match': False,
        'num_new_specials': 1,
        'num_specials_activated': 1,
        'shuffled': False,
        'effective_actions': [16, 17, 18, 19]
        }
    

    next_obs, reward, done, _, info = env.step(19)

    assert np.array_equal(next_obs["board"], np.array([[[1, 1, 2, 2, 1],
                                                        [2, 2, 3, 1, 2],
                                                        [1, 3, 2, 3, 1]],
                                                       [[1, 1, 1, 1, 1],
                                                        [1, 1, 1, 1, 1],
                                                        [1, 3, 4, 1, 1]]]))
    assert next_obs["num_moves_left"] == 1
    assert reward == 18
    assert info == {
        'is_combination_match': False,
        'num_new_specials': 2,
        'num_specials_activated': 0,
        'shuffled': False,
        'effective_actions': [1, 2, 4, 7, 15, 17, 19]
        }
    
    
    next_obs, reward, done, _, info = env.step(19)

    assert reward == 20
    assert np.array_equal(next_obs["board"], np.array([[[2, 2, 1, 1, 3],
                                                        [1, 3, 3, 1, 3],
                                                        [1, 3, 3, 2, 1]],

                                                       [[1, 3, 1, 1, 1],
                                                        [1, 1, 1, 1, 1],
                                                        [1, 1, 1, 1, 1]]]))
    assert done
    assert next_obs["num_moves_left"] == 0
    assert info == {
        'is_combination_match': True,
        'num_new_specials': 1,
        'num_specials_activated': 0,
        'shuffled': False,
        'effective_actions': [] # Because there are no moves left.
    }
