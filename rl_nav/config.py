import numpy as np
'''
    alpha_:              float, learning rate
    gamma_:              float, temporal discount factor
    epsilon_:            float, probability of random choice in an epsilon-greedy policy
    lambda_:             float, credit assigment factor, used in the eligibility trace

    total_num_timesteps: int, how many timesteps does the agent have to learn (for learning curve)
    test_every_N_trials: int, how often should we test the agents performance (for learning curve)
    num_experiments:     int, how many runs of the experiment should we use   (for learning curve)

    timestep_to_display: int, steps of learning before displaying an episode or None (for demonstration)

    environment:         np.ndarray, str '' for attainable location, 'X' for impassible obstacle
    rewards:             np.ndarray, the amount of scalar reward in each location
    test_loc:            np.ndarray, '!' for a start location to measure how many steps it takes to get reward 
'''
alpha_   = 0.01
gamma_   = 0.8
epsilon_ = 0.1
lambda_  = 0.8

total_num_timesteps = 1000
test_every_N_trials = 20
num_experiments     = 10

timestep_to_display = 100

environment = np.array([['', '', '', '', '', '', '', '', '', ''], 
                        ['', '', '', '', '', '', '', '', '', ''], 
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '','X','X','X','X','X','X', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', '']])   

rewards   =   np.array([[0,   0,  0,  0,  0,  0,  0,  0,  0,  0], 
                        [0,   0,  0,  0,  0,  0,  0,  0,  0,  0], 
                        [0,   0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [0,   0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [0,   0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [0,   0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [0,   0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [0,   0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [0,   0,  0,  0,  1,  1,  0,  0,  0,  0]])                               

test_loc   =  np.array([['', '', '', '', '!','', '', '', '', ''], 
                        ['', '', '', '', '', '', '', '', '', ''], 
                        ['', '', '', '', '', '', '', '', '', ''], 
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', ''],
                        ['', '', '', '', '', '', '', '', '', '']]) 