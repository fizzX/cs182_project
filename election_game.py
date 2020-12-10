import time
from election_functions import ElectionGame
from eraser import Eraser
from election_functions import get_election_results, get_reward_matrices, get_rep_states, get_dem_states
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm

# Generate Reward Functions
election_year = 2016


clean_election_results, state_list, electoral_votes = get_election_results(election_year)

republican_states = get_rep_states(clean_election_results, state_list)
democrat_states = get_dem_states(clean_election_results, state_list)

# define attacker types here
# attack_list = [state_list, republican_states, democrat_states]
# partisan_type = [False, True, True]
# partisan_type = [False, False, False]
attack_list = [state_list]
partisan_type = [False]
# attack_list = [republican_states]
# partisan_type = [False]
# attack_list = [democrat_states]
# partisan_type = [True]


# Get reward matrices
att_reward_array, def_reward_array = get_reward_matrices(clean_election_results, state_list, attack_list, electoral_votes, partisan_type)

# print(att_reward_array)
sec_game = ElectionGame(max_coverage=2, num_attacker_types=len(attack_list), att_reward=att_reward_array, def_reward= def_reward_array)

eraser_solver = Eraser(sec_game)
tic = time.perf_counter()
eraser_solver.solve()
toc = time.perf_counter()
print(eraser_solver.opt_defender_payoff)
print(eraser_solver.opt_coverage)
print(f"Completed in {toc - tic:0.4f} seconds ")



# tic = time.perf_counter()
# for i in range(51):
#     sec_game = ElectionGame(max_coverage=i+1, num_attacker_types=len(
#         attack_list), att_reward=att_reward_array, def_reward=def_reward_array)
#     eraser_solver = Eraser(sec_game)
#     eraser_solver.solve()
#     eraser_solver.opt_defender_payoff
# toc = time.perf_counter()
# print(f"Completed in {toc - tic:0.4f} seconds ")
