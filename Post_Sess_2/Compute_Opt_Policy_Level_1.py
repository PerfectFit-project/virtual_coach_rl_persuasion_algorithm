'''
Determine optimal policy for level 1 of algorithm complexity.
This means to find actions with the highest overall average effort score.
'''


import numpy as np
import pandas as pd
import pickle
import Utils as util


def compute_opt_policy_level_1(data, effort_mean, num_act):
    """Compute the optimal policy for level 1 of algorithm complexity.

    Args:
        data (list): List with samples of the form <s0, s1, a, r>.
        effort_mean (float): Mean value of effort responses.
        num_act (int): Number of possible actions.

    Returns:
        list: Optimal actions.

    """
    num_samples = len(data)

    # All effort responses
    list_of_efforts = list(np.array(data)[:, 3].astype(int))

    # Map effort responses to effort scores from 0 to 1, with the mean mapped to 0.5.
    map_to_rewards = util.get_map_effort_reward(effort_mean, 
                                                output_lower_bound = 0, 
                                                output_upper_bound = 1, 
                                                input_lower_bound = 0, 
                                                input_upper_bound = 10)
    reward_list = util.map_efforts_to_rewards(list_of_efforts, map_to_rewards)
  
    rewards = np.zeros(num_act)
    trials = np.zeros(num_act)

    for data_index in range(num_samples):
        rewards[data[data_index][2]] += reward_list[data_index]
        trials[data[data_index][2]] += 1

    # Calculate average effort score per action
    avg_reward = np.divide(rewards, trials,
                           out = np.zeros_like(rewards),
                           where=trials!=0)

    # Get as optimal policy the actions with highest average effort score
    opt_policy = [i for i in range(num_act) if avg_reward[i] == max(avg_reward)]

    return opt_policy
        

if __name__ == "__main__":

    # Load data extracted from database
    data = pd.read_csv('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/rl_samples_list_binary_exp.csv', converters={'s0': eval, 's1': eval})
    data = data.values.tolist()

    # Mean value of effort responses
    with open("W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/all_data_effort_mean", "rb") as f:
        effort_mean = pickle.load(f)

    # Compute the optimal policy
    # We have 5 actions: 4 persuasion types and the option to not persuade
    opt_policy = compute_opt_policy_level_1(data, effort_mean, num_act = 5)

    with open('W:/staff-umbrella/perfectfit/Exp0/Analysis/All_Data/Level_1_Optimal_Policy', 'wb') as f:
        pickle.dump(opt_policy, f)
