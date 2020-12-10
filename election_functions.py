import numpy as np
import pandas as pd
import csv

def get_election_results(election_year):
    df = pd.read_csv('1976-2016-president.csv')
    electoral_votes = pd.read_csv('electoral_votes.csv')
    state_list = list(df.state.unique())
    election_results = df[['year', 'state', 'candidate',
                        'party', 'candidatevotes', 'totalvotes']]
    selected_election_results = election_results[(election_results['year'] == election_year) &
                                                ((election_results['party'] == 'republican') | (election_results['party'] == 'democrat'))]
    df_gen = []
    for i, state in enumerate(state_list):
        evotes = int(electoral_votes[electoral_votes['state'] == state].evotes)
        df_gen.append([state, 'republican', selected_election_results[(selected_election_results['state'] == state) &
                                                                    (selected_election_results['party'] == 'republican')].candidatevotes.sum(), evotes])
        df_gen.append([state, 'democrat', selected_election_results[(selected_election_results['state'] == state) &
                                                                    (selected_election_results['party'] == 'democrat')].candidatevotes.sum(), evotes])
    clean_election_results = pd.DataFrame(df_gen, columns=['state', 'party', 'votes', 'evotes'])
    return clean_election_results, state_list, electoral_votes


def att_reward(state, election_results, electoral_votes):
    evotes = int(electoral_votes[electoral_votes['state'] == state].evotes)
    dem_votes = int(election_results[(election_results['state'] == state) & (
        election_results['party'] == 'democrat')].votes)
    rep_votes = int(election_results[(election_results['state'] == state) & (
        election_results['party'] == 'republican')].votes)
    total_votes = dem_votes + rep_votes
    margin = (max(dem_votes, rep_votes) -
              min(dem_votes, rep_votes))/total_votes
    return evotes * margin


def partisan_att_reward(state, election_results, electoral_votes):
    evotes = int(electoral_votes[electoral_votes['state'] == state].evotes)
    dem_votes = int(election_results[(election_results['state'] == state) & (
        election_results['party'] == 'democrat')].votes)
    rep_votes = int(election_results[(election_results['state'] == state) & (
        election_results['party'] == 'republican')].votes)
    total_votes = dem_votes + rep_votes
    margin = (max(dem_votes, rep_votes) -
              min(dem_votes, rep_votes))/total_votes
    return evotes/(1+margin)


def att_pos_reward(state, election_results, electoral_votes, attack_list, partisan):
    if state in attack_list:
        if partisan:
            return partisan_att_reward(state, election_results, electoral_votes)
        else:
            return att_reward(state, election_results, electoral_votes)
    else:
        return 0


def att_neg_reward(state, election_results, electoral_votes, attack_list):
    return -538/51


def def_pos_reward(state, election_results, electoral_votes, attack_list):
    if state in attack_list:
        return int(electoral_votes[electoral_votes['state'] == state].evotes)
    else:
        return 0


def def_neg_reward(state, election_results, electoral_votes, attack_list):
    if state in attack_list:
        return -1*int(electoral_votes[electoral_votes['state'] == state].evotes)
    else:
        return 0

def get_rep_states(election_results, state_list):
    rep_state_list = []
    for state in state_list:
        dem_votes = int(election_results[(election_results['state'] == state) & (
            election_results['party'] == 'democrat')].votes)
        rep_votes = int(election_results[(election_results['state'] == state) & (
            election_results['party'] == 'republican')].votes)
        if(dem_votes < rep_votes):
            rep_state_list.append(state)
    return rep_state_list


def get_dem_states(election_results, state_list):
    dem_state_list = []
    for state in state_list:
        dem_votes = int(election_results[(election_results['state'] == state) & (
            election_results['party'] == 'democrat')].votes)
        rep_votes = int(election_results[(election_results['state'] == state) & (
            election_results['party'] == 'republican')].votes)
        if(dem_votes > rep_votes):
            dem_state_list.append(state)
    return dem_state_list

def get_reward_matrices(election_results, state_list, attack_list, electoral_votes, partisan_type):
    '''Takes in election results, a list of the 51 possible targets, and a list of targets that the attacker would benefit from attacking (for partisan attackers), returns Reward Matrices for attacker(s) and defender '''
    att_type_num = len(attack_list)
    att_pos_array = np.zeros(att_type_num)
    att_neg_array = np.zeros(att_type_num)
    def_pos_array = np.zeros(att_type_num)
    def_neg_array = np.zeros(att_type_num)

    for state in state_list:
        att_pos_list = []
        att_neg_list = []
        def_pos_list = []
        def_neg_list = []
        for i in range(att_type_num):
            att_pos_list.append(att_pos_reward(
                state, election_results, electoral_votes, attack_list[i], partisan_type[i]))

            att_neg_list.append(att_neg_reward(
                state, election_results, electoral_votes, attack_list[i]))

            def_pos_list.append(def_pos_reward(
                state, election_results, electoral_votes, attack_list[i]))

            def_neg_list.append(def_neg_reward(
                state, election_results, electoral_votes, attack_list[i]))

        att_pos_array = np.vstack((att_pos_array, att_pos_list))
        att_neg_array = np.vstack((att_neg_array, att_neg_list))
        def_pos_array = np.vstack((def_pos_array, def_pos_list))    
        def_neg_array = np.vstack((def_neg_array, def_neg_list))
    
    att_reward_array = np.array([att_pos_array[1:], att_neg_array[1:]])
    def_reward_array = np.array([def_neg_array[1:], def_pos_array[1:]])

    return att_reward_array, def_reward_array


class ElectionGame:
    """
    A security game is a non-bayesian game in which the payoffs for targets
    are given by their coverage status.
    Covered targets yield higher utilities for the defender, and lower
    utilities for the attacker, whilst uncovered targets yield negative
    utilities for the defender and positive utilities for the attacker.
    """

    def __init__(self, **kwargs):
        self.num_targets = 51
        self.max_coverage = kwargs['max_coverage']
        self.num_attacker_types = kwargs['num_attacker_types']

        # for comparisons with other algos
        self.num_attacker_strategies = self.num_targets

        # uniform distribution over attacker types
        self.attacker_type_probability = np.zeros((self.num_attacker_types))
        self.attacker_type_probability += (1.0 / self.num_attacker_types)

        attacker_random = kwargs['att_reward']
        defender_random = kwargs['def_reward']

        # for attacker uncovered targets yield positive utilities, and covered targets yields negative utilities.
        self.attacker_uncovered = attacker_random[0, :, :]
        self.attacker_covered = attacker_random[1, :, :]

        # for defender uncovered targets yield negative utilities, and covered targets yield positive utilities.
        self.defender_uncovered = defender_random[0, :, :]
        self.defender_covered = defender_random[1, :, :]

        # store the type of this representation
        self.type = "compact"
